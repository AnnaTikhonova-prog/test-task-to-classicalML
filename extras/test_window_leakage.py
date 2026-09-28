"""Тест на утечку: события вне окна не должны влиять на признаки.

Идея простая: строим признаки для подмножества кук дважды - на исходных
событиях и на событиях, дополненных копиями реальных строк, у которых
время сдвинуто строго ДО начала или строго ПОСЛЕ конца окна. Если
фильтрация по окну работает, матрицы признаков обязаны совпасть до
последнего байта. Если бы хоть один признак изменился - значит, будущее
просачивается в модель.

Копии берутся из реальных строк (а не заполняются NaN), чтобы сохранить
dtype колонок: иначе пивоты дали бы другие имена колонок, и тест ругался
бы на фантомную разницу (я на это один раз уже напоролась).

Запуск из корня проекта: python extras/test_window_leakage.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # корень проекта

import numpy as np
import pandas as pd

from src.config import META_COLUMNS
from src.features import build_features
from src.train import load_data


def main():
    train, test, events = load_data(Path("data"))
    meta = pd.concat([train[META_COLUMNS], test[META_COLUMNS]], ignore_index=True)
    subset = meta.head(600)
    ev_sub = events.loc[events["cookie_id"].isin(set(subset["cookie_id"]))].copy()

    feats_a = build_features(subset, ev_sub)

    # Копии реальных событий (для сохранения dtype), сдвинутые строго за окно.
    bounds = subset.set_index("cookie_id")
    fake = ev_sub.sample(n=min(1200, len(ev_sub)), random_state=0).copy()
    half = len(fake) // 2
    fake["event_ts"] = np.where(
        np.arange(len(fake)) < half,
        fake["cookie_id"].map(bounds["window_start_ts"]) - pd.Timedelta(hours=1),
        fake["cookie_id"].map(bounds["window_end_ts"]) + pd.Timedelta(hours=1),
    )
    ev_aug = pd.concat([ev_sub, fake], ignore_index=True)

    feats_b = build_features(subset, ev_aug)
    assert feats_a.equals(feats_b), "Признаки изменились от событий вне окна — утечка!"
    print(f"OK: {len(fake)} синтетических событий вне окон не изменили признаки {feats_a.shape}")


if __name__ == "__main__":
    main()
