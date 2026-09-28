"""Базовые модели: фабрика, мешки по сидам, отбор топ-K признаков.

Здесь живут все пять моделей, которые я сравнивала. Гиперпараметры везде
«разумные дефолты», я их специально не крутила: по опыту первой версии
решения перебор параметров на таком маленьком числе позитивов больше
подгоняется под фолд, чем улучшает модель. Вместо этого я вкладывалась
в честную схему сравнения (мешки сидов, временные фолды).
"""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from src.config import SEED


def fit_with_fallback(model, X, y):
    """Обучает модель; если параллелизм недоступен — повторяет последовательно.

    В некоторых изолированных средах (песочницы без named pipes) joblib не
    может создать пул потоков/процессов и бросает PermissionError. На такой
    случай перевожу модель в однопоточный режим и повторяю fit. На качество
    это не влияет никак, только на скорость. На обычной машине эта ветка
    вообще не срабатывает.
    """
    try:
        return model.fit(X, y)
    except PermissionError:
        params = model.get_params()
        for name in ("n_jobs", "n_threads", "thread_count"):
            if name in params:
                model.set_params(**{name: 1})
        return model.fit(X, y)


def make_model(kind: str, seed: int = SEED, weight: str | None = "default", y_train: np.ndarray | None = None):
    """Одна детерминированная древесная модель.

    `weight` — схема взвешивания классов, которую протокол выбрал для этой
    конкретной модели на SELECT-фолдах. Значение "default" воспроизводит
    мою исходную захардкоженную конфигурацию из первой версии — оно нужн
    только чтобы можно было повторить старые эксперименты.
    """
    if kind == "extra_trees":
        class_weight = "balanced" if weight == "default" else weight
        # 450 деревьев: больше не пробовала, прирост после ~300 обычно мизерный.
        # min_samples_leaf=3 приглаживает шум на редких категориях.
        return ExtraTreesClassifier(
            n_estimators=450, max_features=0.7, min_samples_leaf=3,
            class_weight=class_weight, n_jobs=-1, random_state=seed,
        )
    if kind == "random_forest":
        class_weight = "balanced_subsample" if weight == "default" else weight
        return RandomForestClassifier(
            n_estimators=450, max_features=0.7, min_samples_leaf=3,
            class_weight=class_weight, n_jobs=-1, random_state=seed,
        )
    if kind == "hist_gradient":
        class_weight = "balanced" if weight == "default" else weight
        return HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=5.0, class_weight=class_weight,
            early_stopping=False, random_state=seed,
        )
    if kind == "catboost":
        auto_class_weights = "SqrtBalanced" if weight == "default" else weight
        # Неглубокие деревья (depth=6) и медленный lr: данных мало,
        # глубокие деревья сразу переобучаются. allow_writing_files=False,
        # чтобы CatBoost не оставлял свои служебные папки рядом с кодом.
        return CatBoostClassifier(
            iterations=800, depth=6, learning_rate=0.04, l2_leaf_reg=5.0,
            auto_class_weights=auto_class_weights, loss_function="Logloss",
            verbose=False, random_seed=seed, thread_count=8, allow_writing_files=False,
        )
    if kind == "lightgbm":
        if weight == "default":
            # Значение из моей первой версии решения, оставила для истории.
            scale_pos_weight = 10.5
        elif weight is None:
            scale_pos_weight = 1.0
        else:
            if y_train is None:
                raise ValueError("Схемы весов LightGBM требуют y_train")
            ratio = float(np.sum(y_train == 0)) / max(float(np.sum(y_train == 1)), 1.0)
            # "sqrt" — корень из соотношения классов, "full" — само соотношение.
            scale_pos_weight = np.sqrt(ratio) if weight == "sqrt" else ratio
        return LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=30,
            reg_lambda=5.0, reg_alpha=0.5, colsample_bytree=0.8, subsample=0.8,
            subsample_freq=1, scale_pos_weight=scale_pos_weight, verbosity=-1,
            # deterministic=True + force_col_wise: LightGBM иначе может дать
            # чуть разные результаты от запуска к запуску на одном сиде,
            # а мне нужна побайтовая воспроизводимость сабмита.
            n_jobs=8, random_state=seed, deterministic=True, force_col_wise=True,
        )
    raise ValueError(f"Неизвестная модель: {kind}")


def fit_kind_bagged(kind, X_tr, y_tr, X_pr, seeds, weight):
    """Обучает по одной модели на сид; возвращает (усреднённые скоры, скоры по сидам).

    Важный для меня принцип: кандидаты всегда сравниваются как мешки сидов,
    а не как одиночные запуски. Я проверяла — разброс ±2 п.п. между сидами
    одной и той же модели это норма, потому одиночные запуски сравнивать
    бессмысленно: отличишь шум от модели.
    """
    per_seed = []
    for seed in seeds:
        model = make_model(kind, seed=seed, weight=weight, y_train=y_tr)
        fit_with_fallback(model, X_tr, y_tr)
        per_seed.append(model.predict_proba(X_pr)[:, 1])
    return np.mean(per_seed, axis=0), per_seed


def topk_columns(X_tr, y_tr, weight_et, seeds, k):
    """Топ-k признаков по важности ExtraTrees, усреднённой по сидам.

    Важности считаются только по train-части соответствующего фолда, поэтому
    информация из валидационного блока сюда не просачивается. Это нужно для
    честного сравнения "полный набор против топ-K" в validate.py.
    """
    importances = np.zeros(X_tr.shape[1], dtype=float)
    for seed in seeds:
        model = make_model("extra_trees", seed=seed, weight=weight_et, y_train=y_tr)
        fit_with_fallback(model, X_tr, y_tr)
        importances += model.feature_importances_
    order = np.argsort(-importances)
    return list(X_tr.columns[order[:k]])
