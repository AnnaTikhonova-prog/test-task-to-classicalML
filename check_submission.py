"""Независимая проверка submission.csv перед отправкой.

Запуск из корня проекта:
    python check_submission.py

Зачем отдельный скрипт: контракт проверяется и при записи (src/train.py),
но перед отправкой приятно перепроверить готовый файл ещё раз, уже "свежим
взглядом". Отдельно печатается число одинаковых score: официальная метрика
обрабатывает куки с равным score как одну группу, поэтому массовые ничьи
(характерны, например, для ранговых блендов) важно заметить заранее.
"""

from pathlib import Path

import pandas as pd

SUBMISSION = Path("submission.csv")
TEST = Path("data") / "test.csv"


def main() -> None:
    submission = pd.read_csv(SUBMISSION)
    test = pd.read_csv(TEST, usecols=["cookie_id"])

    # Контракт из описания задачи: ровно две колонки, одна строка на куки,
    # тот же состав и порядок cookie_id, скоры конечные и в [0, 1].
    assert list(submission.columns) == ["cookie_id", "score"], "нужны ровно две колонки: cookie_id, score"
    assert len(submission) == len(test), f"строк: {len(submission)}, ожидается {len(test)}"
    assert submission["cookie_id"].is_unique, "cookie_id должны быть уникальны"
    assert submission["cookie_id"].equals(test["cookie_id"]), "состав и порядок cookie_id должны совпадать с test.csv"
    assert submission["score"].notna().all(), "score не должен содержать NaN"
    assert submission["score"].between(0, 1).all(), "score должен быть в [0, 1]"

    # duplicated() помечает второе и дальше вхождения, поэтому n_tied это
    # именно число "лишних" кук в ничьих, а не размер всех ничейных групп.
    n_tied = int(submission["score"].duplicated().sum())
    n_groups = int(submission["score"].nunique())
    print(f"OK: {len(submission)} строк, колонки cookie_id/score, score в [0,1], без NaN")
    print(f"Одинаковые score: {n_tied} куки в ничьих ({n_groups} уникальных значений из {len(submission)})")


if __name__ == "__main__":
    main()
