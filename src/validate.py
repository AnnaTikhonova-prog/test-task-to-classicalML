"""Вложенная хронологическая валидация: SELECT -> EXAM -> HORIZON.

Почему я сделала именно такую схему. Реальный тест идёт 7 днями сразу после
конца train, поэтому случайные фолды тут нечестны: они перемешивают время и
завышают оценку. Отсюда структура:

* SELECT (два expanding-фолда внутри 06–16.04) — единственное место, где
  я принимала решения: схема весов классов для каждой модели, набор
  признаков (полный против топ-K) и выбор итогового кандидата. Все кандидаты
  сравниваются как мешки из нескольких сидов.
* EXAM (train <= 16.04 -> val 17–19.04) не участвует ни в одном выборе;
  здесь замороженный рецепт получает честное число с разбросом по сидам и
  бутстрап-CI, плюс диагностическая таблица всех кандидатов (проверка,
  насколько отбор на SELECT вообще стабилен).
* HORIZON (train <= 12.04 -> predict 13–19.04) повторяет форму реального
  теста: 7 дней обучения -> 7 дней предсказания.
* Стекинг обучается на out-of-fold предсказаниях временных под-фолдов;
  мета-модель для каждого оцениваемого блока обучается строго на более
  ранних днях — иначе стекинг подглядывал бы в будущее. Финальный шаг —
  обязательный FN/FP-профиль ошибок на EXAM.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.optimize import nnls

from metric import precision_at_recall, recall_at_fpr
from src.config import (
    BOOTSTRAP_ROUNDS, EXAM_TRAIN_END, EXAM_VAL, HORIZON_TRAIN_END, HORIZON_VAL,
    MODEL_KINDS, OOF_FOLDS, CLASS_WEIGHT_GRID, SEED, SEEDS_COMPARE, SEEDS_OOF,
    SEEDS_WEIGHT_SEARCH, SELECT_FOLDS, STACK_CANDIDATES, TARGET_RECALL, TOP_K,
)
from src.models import fit_kind_bagged, fit_with_fallback, topk_columns

# Порядок строк в таблицах кандидатов: сначала базовые модели, потом бленды
# (в т.ч. мой ручной рецепт из первой версии), в конце стекинг.
CANDIDATE_ORDER = [
    *MODEL_KINDS,
    "blend_prob_et_rf", "blend_prob_4", "blend_prob_5",
    "blend_rank_et_rf", "blend_rank_4", "blend_five_rank_manual",
    *STACK_CANDIDATES,
]

# Интерпретируемые признаки для профиля ошибок (FN/FP). Брала те, которые
# могу осмысленно прочитать глазами: активность, темп, интервалы.
PROFILE_COLUMNS = [
    "n_events", "active_span_sec", "events_per_active_span_hour",
    "gap_sec_mean", "gap_sec_median", "gap_under_2s_fraction",
    "hour_entropy", "session_count_30m", "zero_gap_fraction",
    "ua_headless_fraction", "ua_missing_fraction",
    "pointer_sample_fraction", "item_id_nunique", "search_query_nunique",
]


def _rank(values: np.ndarray) -> np.ndarray:
    """Процентильный ранг внутри одного блока.

    Ранги считаются отдельно в каждом блоке и никогда не склеиваются между
    фолдами — распределения скоров в разные дни разные, склеивать их это
    как сравнивать градусы в разных шкалах.
    """
    return pd.Series(values).rank(pct=True).to_numpy()


def candidate_scores(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Все кандидаты-бленды, детерминированно построенные из базовых скоров.

    Бленды — не подгонка: веса фиксированы заранее (равные доли или мой
    старый рецепт), ничего не подбирается под фолд.
    """
    et, rf, hgb, cb, lgbm = (base[kind] for kind in MODEL_KINDS)
    prob4 = np.mean([et, rf, hgb, cb], axis=0)
    out = dict(base)
    out["blend_prob_et_rf"] = (et + rf) / 2
    out["blend_prob_4"] = prob4
    out["blend_prob_5"] = np.mean([et, rf, hgb, cb, lgbm], axis=0)
    out["blend_rank_et_rf"] = (_rank(et) + _rank(rf)) / 2
    out["blend_rank_4"] = np.mean([_rank(et), _rank(rf), _rank(hgb), _rank(cb)], axis=0)
    # Ручной рецепт из первой версии решения оставила как кандидата —
    # пусть честно конкурирует с остальными, а не выигрывает "по традиции".
    out["blend_five_rank_manual"] = 0.75 * _rank(prob4) + 0.25 * _rank(lgbm)
    return out


def build_oof(day, y, X, weights, seeds):
    """Out-of-fold скоры базовых моделей на временных под-фолдах.

    Каждая кука получает предсказание от модели, обученной строго на более
    ранних днях; строки вне покрытия под-фолдов остаются NaN и в обучение
    мета-модели не попадают.
    """
    oof = pd.DataFrame(np.nan, index=X.index, columns=list(MODEL_KINDS), dtype=float)
    for train_end, val_start, val_end in OOF_FOLDS:
        tr = day.le(pd.Timestamp(train_end)).to_numpy()
        va = day.between(pd.Timestamp(val_start), pd.Timestamp(val_end)).to_numpy()
        print(f"  OOF-фолд: train<={train_end} ({int(tr.sum())}) -> {val_start}..{val_end} ({int(va.sum())})", flush=True)
        for kind in MODEL_KINDS:
            mean_pred, _ = fit_kind_bagged(kind, X.loc[tr], y[tr], X.loc[va], seeds, weights[kind])
            oof.loc[va, kind] = mean_pred
    return oof


def fit_meta_learners(oof: pd.DataFrame, y, day, max_day):
    """Мета-модели стекинга на OOF-строках строго раньше оцениваемого блока.

    max_day — последний день, который разрешено видеть мета-модели. Это
    главный предохранитель стекинга от утечки: при оценке блока 11–13.04
    мета учится только на днях <= 10.04.
    """
    mask = day.le(pd.Timestamp(max_day)).to_numpy() & oof.notna().all(axis=1).to_numpy()
    Xo = oof.loc[mask].to_numpy(dtype=float)
    yo = y[mask]
    logreg = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
    logreg.fit(Xo, yo)
    # Вторая мета-модель: неотрицательный МНК — по сути выпуклый бленд
    # без смещения; иногда устойчивее логрега на маленьких данных.
    nnls_weights, _ = nnls(Xo, yo)
    return logreg, nnls_weights, int(mask.sum())


def apply_meta(base: dict[str, np.ndarray], logreg, nnls_weights):
    """Скоры стеков: логистическая регрессия и неотрицательный МНК."""
    Xb = np.column_stack([base[kind] for kind in MODEL_KINDS])
    return {
        "stack_logreg": logreg.predict_proba(Xb)[:, 1],
        "stack_nnls": np.clip(Xb @ nnls_weights, 0.0, 1.0),
    }


def all_candidate_scores(base, logreg=None, nnls_weights=None):
    out = candidate_scores(base)
    if logreg is not None:
        out.update(apply_meta(base, logreg, nnls_weights))
    return out


# --------------------------------------------------------------------------
# Метрики и статистические помощники
# --------------------------------------------------------------------------

def metric_bundle(y_true, score):
    """Основная метрика + дополнительные, чтобы видеть картину целиком."""
    return {
        "p_at_r70": precision_at_recall(y_true, score),
        "pr_auc": average_precision_score(y_true, score),
        "roc_auc": roc_auc_score(y_true, score),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, score),
    }


def bootstrap_ci(y_true, score, rounds=BOOTSTRAP_ROUNDS, seed=SEED):
    """Перцентильный бутстрап-CI метрики P@R70 по кукам одного блока.

    Нужен потому что позитивов в блоках мало (160–408), и точечная оценка
    без интервала вводит в заблуждение — я сама на этом погорела в первой
    версии решения.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    n = len(y_true)
    values = []
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        # Вырожденная подвыборка без единого позитива: метрика не определена,
        # такие редкие итерации просто пропускаю (на интервал это не влияет).
        if y_true[idx].sum() == 0:
            continue
        values.append(precision_at_recall(y_true[idx], score[idx]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def threshold_at_recall(y_true, score, recall=TARGET_RECALL):
    """Порог скора в рабочей точке: максимум precision при recall >= целевого.

    Группы одинаковых скоров обрабатываются целиком (как в официальной
    метрике): внутри группы резать нельзя.
    """
    # mergesort стабилен: при равных скорах сохраняется исходный порядок,
    # а нам дальше важно аккуратно найти границы групп одинаковых значений.
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    y = np.asarray(y_true, dtype=int)[order]
    s = np.asarray(score, dtype=float)[order]
    ends = np.r_[s[1:] != s[:-1], True]
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    prec, rec = tp[ends] / k[ends], tp[ends] / max(int(y.sum()), 1)
    ok = rec >= recall
    if not ok.any():
        return float(s[-1]), 0.0, float(rec[-1])
    best = int(np.argmax(np.where(ok, prec, -np.inf)))
    return float(s[ends][best]), float(prec[best]), float(rec[best])


def error_profile(y_true, score, full_features: pd.DataFrame):
    """Обязательный FN/FP-профиль в рабочей точке (медианы признаков по группам).

    Хочется понимать, КОГО модель путает, а не только сколько: если FN это
    "тихие" куки, а FP это гиперактивные люди — дальше крутить модель
    бессмысленно, упёрлись в отделимость данных.
    """
    threshold, prec, rec = threshold_at_recall(y_true, score)
    y = np.asarray(y_true, dtype=int)
    flagged = np.asarray(score) >= threshold
    groups = {
        "TP (пойманные боты)": flagged & (y == 1),
        "FN (пропущенные боты)": ~flagged & (y == 1),
        "FP (люди под тревогой)": flagged & (y == 0),
        "TN (люди вне тревоги)": ~flagged & (y == 0),
    }
    print(f"  Рабочая точка: порог={threshold:.4f}, precision={prec:.4f}, recall={rec:.4f}")
    print("  Размеры групп: " + ", ".join(f"{name.split()[0]}={int(mask.sum())}" for name, mask in groups.items()))
    cols = [c for c in PROFILE_COLUMNS if c in full_features.columns]
    profile = pd.DataFrame(index=cols)
    for name, mask in groups.items():
        profile[name.split()[0]] = full_features.loc[mask, cols].median()
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4g}".format):
        print("  Медианы признаков по группам:")
        print(profile.to_string())
    return profile


# --------------------------------------------------------------------------
# Стадии протокола
# --------------------------------------------------------------------------

def select_fold_masks(day):
    """Маски двух SELECT-фолдов (expanding window, хронологически)."""
    folds = []
    for name, train_end, val_start, val_end in SELECT_FOLDS:
        tr = day.le(pd.Timestamp(train_end)).to_numpy()
        va = day.between(pd.Timestamp(val_start), pd.Timestamp(val_end)).to_numpy()
        folds.append((name, tr, va))
    return folds


def stage_select_weights(day, y, X, folds):
    """Схема весов классов подбирается для каждой модели отдельно.

    Изначально у меня везде стояли "разумные" веса из головы; проверка
    показала, что для разных моделей лучшее разное (а для HistGB — вообще
    без весов), так что это честный мини-подбор, а не дань моде.
    """
    print("\n[SELECT] Схемы весов классов (средний P@R70 по сидам и фолдам):")
    weights = {}
    for kind in MODEL_KINDS:
        scores = {}
        for weight in CLASS_WEIGHT_GRID[kind]:
            fold_metrics = []
            for _, tr, va in folds:
                mean_pred, _ = fit_kind_bagged(kind, X.loc[tr], y[tr], X.loc[va], SEEDS_WEIGHT_SEARCH, weight)
                fold_metrics.append(precision_at_recall(y[va], mean_pred))
            scores[weight] = float(np.mean(fold_metrics))
        weights[kind] = max(scores, key=scores.get)
        printable = {str(w): f"{value:.4f}" for w, value in scores.items()}
        print(f"  {kind:15s} -> {weights[kind]}   {printable}", flush=True)
    return weights


def stage_select_features(day, y, X, folds, weights):
    """Полный набор против топ-K: сравнение только на SELECT-фолдах.

    Топ-K считается по важностям ExtraTrees отдельно для train-части каждого
    фолда — чтобы список колонок не подглядывал в валидацию.
    """
    print(f"\n[SELECT] Набор признаков: полный против топ-{TOP_K} (blend_prob_5, среднее по фолдам):")
    fold_topk = [
        topk_columns(X.loc[tr], y[tr], weights["extra_trees"], SEEDS_OOF, TOP_K)
        for _, tr, _ in folds
    ]
    results = {"full": [], "top60": []}
    for fold_idx, (_, tr, va) in enumerate(folds):
        for set_name, cols in (("full", list(X.columns)), ("top60", fold_topk[fold_idx])):
            base = {}
            for kind in MODEL_KINDS:
                base[kind], _ = fit_kind_bagged(
                    kind, X.loc[tr, cols], y[tr], X.loc[va, cols], SEEDS_WEIGHT_SEARCH, weights[kind]
                )
            score = precision_at_recall(y[va], candidate_scores(base)["blend_prob_5"])
            results[set_name].append(score)
    for set_name, values in results.items():
        print(f"  {set_name:6s}: фолды={[f'{v:.4f}' for v in values]}, среднее={np.mean(values):.4f}", flush=True)
    # При равенстве берём полный набор — он проще воспроизводится.
    chosen = "top60" if np.mean(results["top60"]) > np.mean(results["full"]) else "full"
    print(f"  выбран набор признаков: {chosen}")
    return chosen, fold_topk


def stage_compare_candidates(day, y, X, folds, weights, feature_set, fold_topk):
    """Сравнение кандидатов мешками сидов только на SELECT-фолдах.

    Это самая долгая стадия: 5 базовых моделей x 5 сидов x 2 фолда плюс
    OOF-под-фолды для стекинга. Зато после неё выбор кандидата — не вопрос
    вкуса, а строчка таблицы.
    """
    print("\n[SELECT] Сравнение кандидатов (мешки по 5 сидов, P@R70 по фолдам):")
    oof = build_oof(day, y, X, weights, SEEDS_OOF)
    per_candidate: dict[str, list[float]] = {name: [] for name in CANDIDATE_ORDER}
    for fold_idx, (name, tr, va) in enumerate(folds):
        cols = list(X.columns) if feature_set == "full" else fold_topk[fold_idx]
        base = {}
        for kind in MODEL_KINDS:
            base[kind], _ = fit_kind_bagged(kind, X.loc[tr, cols], y[tr], X.loc[va, cols], SEEDS_COMPARE, weights[kind])
        val_start = day[va].min()
        # Мета-модель обучается только на днях строго раньше валидационного блока.
        logreg, nnls_weights, n_meta = fit_meta_learners(oof, y, day, val_start - pd.Timedelta(days=1))
        scores = all_candidate_scores(base, logreg, nnls_weights)
        print(f"  Фолд {name} (мета обучена на {n_meta} OOF-строках < {val_start.date()}):", flush=True)
        for candidate in CANDIDATE_ORDER:
            metric = precision_at_recall(y[va], scores[candidate])
            per_candidate[candidate].append(metric)
            print(f"    {candidate:22s} P@R70={metric:.4f}")
    summary = {name: float(np.mean(values)) for name, values in per_candidate.items()}
    print("\n[SELECT] Среднее по фолдам:")
    for name in sorted(summary, key=lambda key: -summary[key]):
        print(f"    {name:22s} {summary[name]:.4f}")
    winner = max(summary, key=summary.get)
    print(f"[SELECT] победитель: {winner}")
    return winner, oof, summary


def evaluate_block(tag, day, y, X, weights, feature_set, train_end, val_range, oof,
                   meta_max_day, seeds=SEEDS_COMPARE):
    """Обучение на всём <= train_end и оценка одного нетронутого будущего блока.

    Универсальная процедура для EXAM и HORIZON: здесь ничего не выбирается,
    только честно измеряется замороженный рецепт (плюс диагностическая
    таблица всех кандидатов — смотреть на стабильность отбора).
    """
    tr = day.le(pd.Timestamp(train_end)).to_numpy()
    va = day.between(pd.Timestamp(val_range[0]), pd.Timestamp(val_range[1])).to_numpy()
    cols = list(X.columns)
    if feature_set == "top60":
        cols = topk_columns(X.loc[tr], y[tr], weights["extra_trees"], SEEDS_OOF, TOP_K)
    print(f"\n[{tag}] train<={train_end} ({int(tr.sum())}, поз={int(y[tr].sum())}) -> "
          f"{val_range[0]}..{val_range[1]} ({int(va.sum())}, поз={int(y[va].sum())}), признаков={len(cols)}", flush=True)
    # Простейший baseline из quickstart: случайный лес на двух признаках.
    # Нужен, чтобы было с чем сравнить "сложное" решение (разница огромная).
    base_cols = [c for c in ("n_events", "item_id_nunique") if c in cols]
    baseline = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED)
    fit_with_fallback(baseline, X.loc[tr, base_cols], y[tr])
    baseline_score = baseline.predict_proba(X.loc[va, base_cols])[:, 1]
    base, base_seeds = {}, {}
    for kind in MODEL_KINDS:
        base[kind], base_seeds[kind] = fit_kind_bagged(
            kind, X.loc[tr, cols], y[tr], X.loc[va, cols], seeds, weights[kind]
        )
    logreg, nnls_weights, n_meta = fit_meta_learners(oof, y, day, meta_max_day)
    scores = all_candidate_scores(base, logreg, nnls_weights)
    scores["baseline_rf"] = baseline_score
    print(f"[{tag}] мета обучена на {n_meta} OOF-строках")
    table = {"baseline_rf": metric_bundle(y[va], baseline_score)}
    print(f"    {'baseline_rf':22s} P@R70={table['baseline_rf']['p_at_r70']:.4f}", flush=True)
    for candidate in CANDIDATE_ORDER:
        table[candidate] = metric_bundle(y[va], scores[candidate])
        print(f"    {candidate:22s} P@R70={table[candidate]['p_at_r70']:.4f} "
              f"PR-AUC={table[candidate]['pr_auc']:.4f} ROC-AUC={table[candidate]['roc_auc']:.4f} "
              f"R@1%FPR={table[candidate]['recall_at_1pct_fpr']:.4f}", flush=True)
    return scores, table, base_seeds, (tr, va)


def validate(train_meta, y, X, full_features):
    """Вложенный протокол: SELECT решает всё, EXAM и HORIZON только репортят."""
    t0 = time.perf_counter()
    day = train_meta["window_start_ts"].dt.normalize()
    folds = select_fold_masks(day)
    for name, tr, va in folds:
        print(f"Фолд {name}: train={int(tr.sum())} (поз={int(y[tr].sum())}), "
              f"val={int(va.sum())} (поз={int(y[va].sum())})")

    # Стадия 1: все решения — только на SELECT.
    weights = stage_select_weights(day, y, X, folds)
    feature_set, fold_topk = stage_select_features(day, y, X, folds, weights)
    winner, oof, select_summary = stage_compare_candidates(
        day, y, X, folds, weights, feature_set, fold_topk
    )
    print(f"[время] SELECT-стадии: {(time.perf_counter() - t0) / 60:.1f} мин", flush=True)

    # Стадия 2: EXAM — нетронутый блок; здесь замороженный рецепт получает
    # честное число с разбросом по сидам и бутстрап-интервалом.
    exam_scores, exam_table, exam_seeds, (_, exam_mask) = evaluate_block(
        "EXAM", day, y, X, weights, feature_set, EXAM_TRAIN_END, EXAM_VAL, oof,
        meta_max_day=pd.Timestamp(EXAM_VAL[0]) - pd.Timedelta(days=1),
    )
    winner_score = exam_scores[winner]
    # Разброс по сидам: тот же кандидат, но скоры каждого сида по отдельности.
    # Это ответ на вопрос "а не повезло ли нам с конкретным сидом" — если
    # разброс большой, точечной оценке верить нельзя.
    per_seed_metrics = []
    for seed_idx in range(len(SEEDS_COMPARE)):
        seed_base = {kind: exam_seeds[kind][seed_idx] for kind in MODEL_KINDS}
        logreg, nnls_weights, _ = fit_meta_learners(
            oof, y, day, pd.Timestamp(EXAM_VAL[0]) - pd.Timedelta(days=1)
        )
        seed_scores = all_candidate_scores(seed_base, logreg, nnls_weights)
        per_seed_metrics.append(precision_at_recall(y[exam_mask], seed_scores[winner]))
    # ddof=1: несмещённая оценка std, сидов всего пять, поправка заметна.
    seed_std = float(np.std(per_seed_metrics, ddof=1))
    ci_lo, ci_hi = bootstrap_ci(y[exam_mask], winner_score)
    exam_metric = precision_at_recall(y[exam_mask], winner_score)
    print(f"\n[EXAM] {winner}: P@R70={exam_metric:.4f} "
          f"(разброс по сидам +/-{seed_std:.4f}, бутстрап 95% CI [{ci_lo:.4f}, {ci_hi:.4f}])")

    # Стадия 3: HORIZON — та же форма, что у реального теста (7 дней -> 7 дней).
    horizon_scores, horizon_table, _, (_, horizon_mask) = evaluate_block(
        "HORIZON", day, y, X, weights, feature_set, HORIZON_TRAIN_END, HORIZON_VAL, oof,
        meta_max_day=pd.Timestamp(HORIZON_VAL[0]) - pd.Timedelta(days=1),
    )
    horizon_metric = precision_at_recall(y[horizon_mask], horizon_scores[winner])
    hz_ci_lo, hz_ci_hi = bootstrap_ci(y[horizon_mask], horizon_scores[winner])
    print(f"\n[HORIZON] {winner}: P@R70={horizon_metric:.4f} "
          f"(бутстрап 95% CI [{hz_ci_lo:.4f}, {hz_ci_hi:.4f}]) — оценка формы реального теста")

    # Обязательный профиль ошибок на EXAM-блоке: хочется видеть, кого путаем.
    print(f"\n[EXAM] Профиль ошибок кандидата {winner}:")
    error_profile(y[exam_mask], winner_score, full_features.loc[exam_mask])
    print(f"[время] вся валидация: {(time.perf_counter() - t0) / 60:.1f} мин", flush=True)

    return {
        "weights": weights,
        "feature_set": feature_set,
        "candidate": winner,
        "select_summary": select_summary,
        "exam_table": exam_table,
        "exam_metric": exam_metric,
        "exam_seed_std": seed_std,
        "exam_ci": (ci_lo, ci_hi),
        "horizon_metric": horizon_metric,
        "horizon_ci": (hz_ci_lo, hz_ci_hi),
        "baseline_exam": exam_table["baseline_rf"]["p_at_r70"],
        "baseline_horizon": horizon_table["baseline_rf"]["p_at_r70"],
    }
