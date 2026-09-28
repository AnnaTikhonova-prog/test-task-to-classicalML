"""Загрузка данных, финальное обучение замороженного рецепта и запись сабмита.

Здесь нет никаких решений "по ходу": рецепт (кандидат, веса, набор признаков)
приходит извне - либо из протокола валидации, либо из FALLBACK_CONFIG при
запуске с --skip-validation. Файл только добросовестно воспроизводит его
на всём train и аккуратно проверяет контракт итогового файла.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATE_COLUMNS, MODEL_KINDS, OOF_FOLDS, OOF_FINAL_EXTRA,
    SEEDS_FINAL, SEEDS_OOF, STACK_CANDIDATES, TOP_K,
)
from src.models import fit_kind_bagged, topk_columns
from src.validate import apply_meta, candidate_scores, fit_meta_learners


def load_data(data_dir: Path):
    """Читает train/test/events и проверяет базовые инварианты.

    Точные дубликаты строк событий (4863 строки — похоже, артефакт
    логирования) удаляются: они завышают счётчики собтий, не неся никакой
    информации. Совпадения только по (cookie_id, event_ts, event_name), но
    с разным содержимым, я посчитала легитимными повторами и оставила.
    """
    train = pd.read_csv(data_dir / "train.csv", parse_dates=DATE_COLUMNS)
    test = pd.read_csv(data_dir / "test.csv", parse_dates=DATE_COLUMNS)
    events = pd.read_csv(data_dir / "events.csv.gz", parse_dates=["event_ts"])
    n_before = len(events)
    # reset_index обязателен: после удаления строк в индексе дырки, а дальше
    # по коду индексы используются в масках и .loc, дырки всё ломают.
    events = events.drop_duplicates().reset_index(drop=True)
    print(f"Дубликаты событий: удалено {n_before - len(events)} из {n_before}")
    # Минимальные проверки "на дурака": лучше упасть здесь с понятным
    # сообщением, чем получить странные числа где-то в середине прогона.
    required_events = {"cookie_id", "event_ts", "event_name"}
    if not required_events.issubset(events.columns):
        raise ValueError(f"в events.csv.gz нет колонок: {sorted(required_events - set(events.columns))}")
    if train["cookie_id"].duplicated().any() or test["cookie_id"].duplicated().any():
        raise ValueError("Дубликаты cookie_id в train или test")
    if set(train["cookie_id"]) & set(test["cookie_id"]):
        raise ValueError("cookie_id пересекаются между train и test")
    if not train["target"].isin([0, 1]).all():
        raise ValueError("target должен содержать только 0 и 1")
    return train, test, events


def fit_final(train_meta, y, X, test_X, config):
    """Переобучает замороженный рецепт на всём train (мешок из 7 сидов).

    Если выбран стек, дополнительно считаем OOF-предсказания на всём train
    и учим мета-модель; если бленд — просто смешиваем скоры базовых моделей;
    если одиночная модель — обучаем только её (остальные не нужны).
    """
    # День окна куки: именно по нему режутся все временные фолды.
    day = train_meta["window_start_ts"].dt.normalize()
    weights, feature_set, candidate = config["weights"], config["feature_set"], config["candidate"]
    cols = list(X.columns)
    if feature_set == "top60":
        # На всякий случай оставила ветку сокращённого набора, хотя протокол
        # выбрал полный (см. отчёт, раздел "Отрицательные результаты").
        cols = topk_columns(X, y, weights["extra_trees"], SEEDS_OOF, TOP_K)
    print(f"\n[FINAL] кандидат={candidate}, набор={feature_set} ({len(cols)} колонок), сидов={len(SEEDS_FINAL)}")
    # Обучаем только нужные базовые модели: одиночной модели — она одна,
    # блендам и стекам — весь набор из пяти (иначе смешивать нечего).
    needs_all = candidate in STACK_CANDIDATES or candidate.startswith("blend")
    kinds = MODEL_KINDS if needs_all else (candidate,)
    base = {}
    for kind in kinds:
        base[kind], _ = fit_kind_bagged(kind, X.loc[:, cols], y, test_X.loc[:, cols], SEEDS_FINAL, weights[kind])
        print(f"  обучена {kind}", flush=True)
    if candidate in STACK_CANDIDATES:
        # OOF на всём train: четыре под-фолда SELECT-периода + блок 17–19.04,
        # тогда мета-модель видит предсказания для каждой куки.
        # Строки вне покрытия (самые ранние дни) остаются NaN и в обучение
        # мета-модели не попадают — см. fit_meta_learners.
        oof = pd.DataFrame(np.nan, index=X.index, columns=list(MODEL_KINDS), dtype=float)
        for train_end, val_start, val_end in list(OOF_FOLDS) + [OOF_FINAL_EXTRA]:
            tr = day.le(pd.Timestamp(train_end)).to_numpy()
            va = day.between(pd.Timestamp(val_start), pd.Timestamp(val_end)).to_numpy()
            for kind in MODEL_KINDS:
                mean_pred, _ = fit_kind_bagged(kind, X.loc[tr, cols], y[tr], X.loc[va, cols], SEEDS_OOF, weights[kind])
                oof.loc[va, kind] = mean_pred
        logreg, nnls_weights, n_meta = fit_meta_learners(oof, y, day, day.max())
        print(f"  мета обучена на {n_meta} OOF-строках (08–19.04)")
        scores = apply_meta(base, logreg, nnls_weights)[candidate]
    elif candidate.startswith("blend"):
        scores = candidate_scores(base)[candidate]
    else:
        scores = base[candidate]
    # Подстраховка: скор по контракту обязан лежать в [0, 1].
    return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)


def write_submission(test: pd.DataFrame, scores: np.ndarray, output: Path) -> pd.DataFrame:
    """Проверяет контракт сабмита и записывает csv.

    Проверки выглядят занудно, но именно они однажды спасают от ситуации
    "файл отправлен, а там сдвинут порядок cookie_id".
    """
    submission = pd.DataFrame({"cookie_id": test["cookie_id"], "score": scores})
    if list(submission.columns) != ["cookie_id", "score"]:
        raise AssertionError("Неожиданные колонки сабмита")
    if (
        len(submission) != len(test)
        or submission["cookie_id"].duplicated().any()
        or not submission["cookie_id"].equals(test["cookie_id"])
    ):
        raise AssertionError("В сабмите должна быть ровно одна строка на тестовую куку")
    if not submission["score"].between(0, 1).all() or submission["score"].isna().any():
        raise AssertionError("Скоры должны быть конечными значениями в [0, 1]")
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    return submission
