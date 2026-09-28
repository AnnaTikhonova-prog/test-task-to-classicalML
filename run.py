"""Воспроизводимое решение задачи детекции ботов (daily-cookie challenge).

Python 3.12.4. Запуск одной командой из корня проекта:
    python run.py

Что делает эта команда, по шагам:
1. читает данные и выкидывает точные дубликаты событий;
2. строит признаки из событий строго внутри окна каждой куки
   [window_start_ts, window_end_ts) — ничего "из будущего" сюда не попадает;
3. прогоняет вложенную хронологическую валидацию (SELECT -> EXAM -> HORIZON)
   и печатает все промежуточные таблицы;
4. переобучает выбранный протоколом рецепт на всём train (мешок из 7 сидов);
5. пишет submission.csv и проверяет его контракт.

Подробности протокола и все числа — в REPORT.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.config import FALLBACK_CONFIG, META_COLUMNS
from src.features import build_features, select_model_features
from src.train import fit_final, load_data, write_submission
from src.validate import validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--skip-validation", action="store_true",
                        help="Пропустить протокол SELECT/EXAM и взять FALLBACK_CONFIG "
                             "(быстро пересобрать сабмит без валидации)")
    parser.add_argument("--candidate", type=str, default=None,
                        help="Переопределить кандидата из FALLBACK_CONFIG "
                             "(удобно для альтернативного сабмита)")
    parser.add_argument("--features-cache", type=Path, default=None,
                        help="Необязательный pickle-кэш матрицы признаков. "
                             "ВАЖНО: удалите кэш, если меняли build_features.")
    args = parser.parse_args()
    t0 = time.perf_counter()

    train, test, events = load_data(args.data_dir)
    # ВАЖНО: train идёт первым, test вторым. На этом порядке завязано
    # разделение матрицы признаков через iloc ниже — менять нельзя.
    all_meta = pd.concat([train[META_COLUMNS], test[META_COLUMNS]], ignore_index=True)
    print(f"Загружено: train={len(train):,}, test={len(test):,}, событий={len(events):,}")

    # Признаки считаются ~1 минуту, но при экспериментах удобно закэшировать.
    if args.features_cache and args.features_cache.exists():
        all_features = pd.read_pickle(args.features_cache)
        print(f"Признаки загружены из кэша {args.features_cache}: {all_features.shape}")
    else:
        all_features = build_features(all_meta, events)
        if args.features_cache:
            all_features.to_pickle(args.features_cache)
            print(f"Признаки сохранены в кэш {args.features_cache}")
    print(f"[время] построение признаков: {(time.perf_counter() - t0) / 60:.1f} мин", flush=True)

    # Из полной матрицы убираем экспериментальные признаки (они оставлены
    # только для ablation, в модель не идут) и делим обратно на train/test.
    # Признаки строятся на объединённых метаданных, чтобы пивоты дали
    # одинаковый набор колонок для train и test (иначе колонки разъедутся).
    model_features = select_model_features(all_features)
    train_features = model_features.iloc[:len(train)].copy()
    test_features = model_features.iloc[len(train):].copy()
    print(f"Матрица признаков: train={train_features.shape}, test={test_features.shape}")

    y = train["target"].to_numpy()
    if not args.skip_validation:
        # Основной путь: все решения принимает протокол, я ничего не подкручиваю.
        config = validate(train[META_COLUMNS], y, train_features, all_features.iloc[:len(train)])
        print("\nЗамороженная конфигурация:", config["candidate"], config["feature_set"], config["weights"])
    else:
        config = FALLBACK_CONFIG
        if args.candidate:
            config = {**FALLBACK_CONFIG, "candidate": args.candidate}
        print(f"Валидация пропущена; конфигурация: {config}")

    scores = fit_final(train[META_COLUMNS], y, train_features, test_features, config)
    submission = write_submission(test, scores, args.output)
    print(f"Выбран {config['candidate']}; записано {len(submission):,} предсказаний в {args.output}")
    print(f"[время] всего: {(time.perf_counter() - t0) / 60:.1f} мин")


if __name__ == "__main__":
    main()
