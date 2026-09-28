"""Воспроизведение эксперимента с инверсией рейтинга (мотивация протокола).

Этот скрипт показывает, почему я ушла от сравнения "одна модель — один
запуск" к мешкам сидов. Здесь каждая модель обучается на ОДНОМ сиде, как
в моей первой версии: FIT (<=13.04) -> SELECT (14–16.04), затем
переобучение на <=16.04 и оценка на нетронутом EXAM (17–19.04). Видно,
что рейтинг моделей на SELECT и на EXAM заметно различается — то есть
одиночный запуск нельзя использовать для выбора модели, разброс между
сидами сопоставим с различием между моделями.

Запуск из корня проекта: python extras/reproduce_rank_inversion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # корень проекта

import pandas as pd

from metric import precision_at_recall
from src.config import META_COLUMNS
from src.features import build_features, select_model_features
from src.models import fit_with_fallback, make_model
from src.train import load_data

CACHE = Path("extras/_features_check.pkl")


def get_features():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    train, test, events = load_data(Path("data"))
    all_meta = pd.concat([train[META_COLUMNS], test[META_COLUMNS]], ignore_index=True)
    feats = select_model_features(build_features(all_meta, events))
    feats.to_pickle(CACHE)
    return feats


def main():
    train, _, _ = load_data(Path("data"))
    feats = get_features().iloc[: len(train)]
    day = train["window_start_ts"].dt.normalize()
    y = train["target"].to_numpy()

    fit_mask = day.le(pd.Timestamp("2026-04-13")).to_numpy()
    sel_mask = day.between(pd.Timestamp("2026-04-14"), pd.Timestamp("2026-04-16")).to_numpy()
    exam_mask = day.between(pd.Timestamp("2026-04-17"), pd.Timestamp("2026-04-19")).to_numpy()

    rows = []
    for kind in ("extra_trees", "random_forest", "hist_gradient", "catboost", "lightgbm"):
        m = make_model(kind)
        fit_with_fallback(m, feats.loc[fit_mask], y[fit_mask])
        s_sel = precision_at_recall(y[sel_mask], m.predict_proba(feats.loc[sel_mask])[:, 1])
        m2 = make_model(kind)
        fit_with_fallback(m2, feats.loc[fit_mask | sel_mask], y[fit_mask | sel_mask])
        s_exam = precision_at_recall(y[exam_mask], m2.predict_proba(feats.loc[exam_mask])[:, 1])
        rows.append((kind, s_sel, s_exam))
        print(f"{kind:15s} SELECT={s_sel:.4f}  EXAM={s_exam:.4f}", flush=True)

    print("\nРейтинг на SELECT:", [r[0] for r in sorted(rows, key=lambda r: -r[1])])
    print("Рейтинг на EXAM  :", [r[0] for r in sorted(rows, key=lambda r: -r[2])])


if __name__ == "__main__":
    main()
