"""Построение признаков: одна строка на cookie_id из событий её окна.

Ключевое свойство корректности, ради которого весь файл и написан:
используются ТОЛЬКО события внутри индивидуального полуоткрытого окна
[window_start_ts, window_end_ts) соответствующей куки. События вне окна
(в том числе captcha_shown, который в этих данных всегда происходит уже
после конца окна, то есть является следствием блокировки) в признаки не
попадают. Это проверяется отдельным тестом extras/test_window_leakage.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Экспериментальные расширения я оставила в build_features для
# воспроизводимого ablation, но из модельной матрицы они исключаются:
# на тех же временных фолдах они снижали качество относительно ядра
# из 347 признаков (числа — в REPORT.md, раздел "Отрицательные результаты").
EXPERIMENTAL_FEATURE_PREFIXES = (
    "gap_cv", "gap_burstiness", "gap_under_", "hour_entropy", "hourly_event_count_",
    "session_events_", "session_duration_sec_", "first_half_event_fraction",
    "last_hour_event_fraction", "ua_missing_fraction", "ua_platform_conflict_fraction",
    "pointer_dominant_point_fraction", "pointer_zero_point_fraction", "page_step_",
    "page_forward_fraction", "page_sequential_fraction", "page_reset_fraction",
    "search_page_event_count", "photo_swipe_per_", "seller_page_view_per_",
    "favorite_add_per_", "contact_phone_show_per_", "contact_chat_open_per_",
    "contact_message_sent_per_", "captcha_shown_per_", "login_per_",
)


def select_model_features(features: pd.DataFrame) -> pd.DataFrame:
    """Оставляет проверенное ядро признаков; экспериментальные колонки — для ablation."""
    keep = [
        column for column in features.columns
        if not column.startswith(EXPERIMENTAL_FEATURE_PREFIXES)
    ]
    return features.loc[:, keep]


def _count_pivot(events: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """Счётчики значений колонки по кукам; пропуски осознанно не категория.

    Пропуск в событии означает "этого не наблюдалось", а не отдельное
    значение, потому строки с NaN я отсекаю до группировки.
    """
    values = events.loc[events[column].notna(), ["cookie_id", column]]
    if values.empty:
        return pd.DataFrame(index=pd.Index([], name="cookie_id"))
    result = values.groupby(["cookie_id", column], observed=True).size().unstack(fill_value=0)
    result.columns = [f"{prefix}__{str(value).strip().lower()}" for value in result.columns]
    return result


def build_features(metadata: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Строит одну строку признаков на куку из событий её собственного окна.

    Дальше идёт длинная последовательность агрегаций. Все они устроены
    одинаково: сгруппировали события окна по cookie_id -> посчитали
    статистику -> положили в колонку. В конце строки выравниваются по
    порядку cookie_id из metadata, а пропуски (= "не наблюдалось")
    заполняются нулями.
    """
    if metadata["cookie_id"].duplicated().any():
        raise ValueError("cookie_id должен быть уникален в metadata")

    # Присоединяем к каждому событию границы окна его куки и сразу режем
    # всё, что попало вне окна. Это и есть главная защита от утечки будущего.
    # validate="many_to_one" заодно проверяет, что окно у куки ровно одно:
    # если бы в metadata оказалось две строки на куку, merge бы упал здесь,
    # а не выдал тихо раздутые события.
    bounds = metadata[["cookie_id", "window_start_ts", "window_end_ts", "cookie_created_at"]]
    ev = events.merge(bounds, on="cookie_id", how="inner", validate="many_to_one")
    ev = ev.loc[
        ev["event_ts"].ge(ev["window_start_ts"])
        & ev["event_ts"].lt(ev["window_end_ts"])
    ].copy()
    ev = ev.sort_values(["cookie_id", "event_ts"], kind="mergesort")

    # В сырых данных регистр платформы пляшет (WEB/Web/web и т.п.),
    # привожу к нижнему регистру, иначе это три разных "платформы".
    ev["platform_norm"] = ev["platform"].astype("string").str.strip().str.lower()
    ev["elapsed_sec"] = (ev["event_ts"] - ev["window_start_ts"]).dt.total_seconds()
    ev["hour"] = ev["event_ts"].dt.hour
    ev["weekday"] = ev["event_ts"].dt.dayofweek
    # Интервал между соседними событиями — основа кучи признаков ниже.
    # sort=False потому что ev уже отсортирован выше (mergesort стабилен,
    # поэтому порядок событий внутри куки сохраняется, а сортировать
    # повторно — лишняя работа).
    ev["gap_sec"] = ev.groupby("cookie_id", sort=False)["event_ts"].diff().dt.total_seconds()
    # Сессия: пауза больше 30 минут = новая сессия (стандартная эвристика).
    ev["session_start"] = ev["gap_sec"].isna() | ev["gap_sec"].gt(30 * 60)
    ev["hour_bin"] = (ev["hour"] // 6).astype("int8")

    grouped = ev.groupby("cookie_id", sort=False)
    features = pd.DataFrame(index=pd.Index(ev["cookie_id"].unique(), name="cookie_id"))
    features["n_events"] = grouped.size()

    # Сколько разных значений принимали ключевые поля: боты обычно
    # либо подозрительно "однообразны", либо наоборот гоняют всё подряд.
    for column in [
        "item_id", "item_category", "item_location", "seller_type", "search_query",
        "search_page", "user_agent", "platform_norm", "event_name", "eid",
    ]:
        features[f"{column}_nunique"] = grouped[column].nunique()

    for column in ["elapsed_sec", "gap_sec", "search_page", "pointer_x", "pointer_y"]:
        for stat in ("mean", "std", "min", "median", "max"):
            features[f"{column}_{stat}"] = getattr(grouped[column], stat)()
    features["first_event_sec"] = grouped["elapsed_sec"].min()
    features["last_event_sec"] = grouped["elapsed_sec"].max()
    for quantile in (0.1, 0.9):
        features[f"gap_sec_q{int(quantile * 100)}"] = grouped["gap_sec"].quantile(quantile)
    # Энтропии распределений: считаю векторизованно, чтобы не городить
    # Python-колбэк на каждую куку (так заметно быстрее). Энтропия показывает,
    # насколько поведение "размазано": низкая энтропия = кука долбит одно и то же.
    for column in ["item_id", "item_category", "item_location", "seller_type", "search_query"]:
        counts = ev.loc[ev[column].notna()].groupby(["cookie_id", column], observed=True).size()
        probabilities = counts / counts.groupby(level=0).transform("sum")
        features[f"{column}_entropy"] = (-(probabilities * np.log(probabilities))).groupby(level=0).sum()

    features["active_span_sec"] = grouped["elapsed_sec"].max() - grouped["elapsed_sec"].min()
    features["active_hours"] = grouped["hour"].nunique()
    features["session_count_30m"] = grouped["session_start"].sum()
    # --- Экспериментальный блок: регулярность и "всплесковость" интервалов ---
    gap = ev.loc[ev["gap_sec"].notna(), ["cookie_id", "gap_sec"]]
    gap_group = gap.groupby("cookie_id", sort=False)["gap_sec"]
    gap_mean = gap_group.mean()
    gap_std = gap_group.std()
    features["gap_cv"] = gap_std / gap_mean.replace(0, np.nan)
    features["gap_burstiness"] = (gap_std - gap_mean) / (gap_std + gap_mean).replace(0, np.nan)
    features["gap_under_1s_fraction"] = gap.assign(flag=gap["gap_sec"].lt(1)).groupby("cookie_id")["flag"].mean()
    features["gap_under_2s_fraction"] = gap.assign(flag=gap["gap_sec"].lt(2)).groupby("cookie_id")["flag"].mean()
    features["gap_under_5s_fraction"] = gap.assign(flag=gap["gap_sec"].lt(5)).groupby("cookie_id")["flag"].mean()
    # Распределение по часам суток; энтропию нормую на log(24), чтобы
    # редкие и активные куки были сопоставимы между собой.
    hour_counts = ev.groupby(["cookie_id", "hour"], observed=True).size()
    hour_probs = hour_counts / hour_counts.groupby(level=0).transform("sum")
    hour_entropy = (-(hour_probs * np.log(hour_probs))).groupby(level=0).sum()
    features["hour_entropy"] = hour_entropy / np.log(24)
    # Почасовой объём событий — ловит устойчивый высокочастотный сбор данных.
    hourly = ev.assign(hour_start=ev["event_ts"].dt.floor("h")).groupby(
        ["cookie_id", "hour_start"], observed=True
    ).size()
    features["hourly_event_count_mean"] = hourly.groupby(level=0).mean()
    features["hourly_event_count_std"] = hourly.groupby(level=0).std()
    features["hourly_event_count_fano"] = (
        hourly.groupby(level=0).var() / hourly.groupby(level=0).mean().replace(0, np.nan)
    )
    # Размеры и длительности сессий (граница неактивности — 30 минут).
    ev["session_id"] = ev.groupby("cookie_id", sort=False)["session_start"].cumsum()
    session = ev.groupby(["cookie_id", "session_id"], sort=False).agg(
        session_events=("event_name", "size"),
        session_duration_sec=("elapsed_sec", lambda x: x.max() - x.min()),
    )
    for column in ("session_events", "session_duration_sec"):
        session_stats = session.groupby(level=0)[column].agg(["mean", "std", "max", "median"])
        session_stats.columns = [f"{column}_{stat}" for stat in session_stats.columns]
        features = features.join(session_stats, how="left")
    # Доля событий с нулевым интервалом — у людей так почти не бывает.
    ev["zero_gap"] = ev["gap_sec"].eq(0).astype("int8")
    features["zero_gap_fraction"] = ev.groupby("cookie_id", sort=False)["zero_gap"].mean()
    ev["search_query_length"] = ev["search_query"].astype("string").str.len()
    features["search_query_mean_length"] = ev.groupby("cookie_id", sort=False)["search_query_length"].mean()
    features["search_query_max_length"] = ev.groupby("cookie_id", sort=False)["search_query_length"].max()
    item_count = grouped["item_id"].count().replace(0, np.nan)
    features["item_repeat_fraction"] = 1 - features["item_id_nunique"] / item_count
    features["events_per_active_hour"] = features["n_events"] / grouped["hour"].nunique().clip(lower=1)
    features["events_per_active_span_hour"] = features["n_events"] / (features["active_span_sec"] / 3600).clip(lower=1 / 60)
    # Экспериментально: концентрация активности в первой половине окна
    # и в последнем часе (идея была ловить "добивание" лимитов).
    window_duration = (bounds.set_index("cookie_id")["window_end_ts"] - bounds.set_index("cookie_id")["window_start_ts"]).dt.total_seconds()
    first_half = ev.loc[ev["elapsed_sec"].le(ev["cookie_id"].map(window_duration) / 2)].groupby("cookie_id").size()
    last_hour = ev.loc[ev["event_ts"].ge(ev["cookie_id"].map(bounds.set_index("cookie_id")["window_end_ts"] - pd.Timedelta(hours=1)))].groupby("cookie_id").size()
    features["first_half_event_fraction"] = first_half / features["n_events"]
    features["last_hour_event_fraction"] = last_hour / features["n_events"]
    features["cookie_age_days"] = (
        (bounds.set_index("cookie_id")["window_start_ts"]
         - bounds.set_index("cookie_id")["cookie_created_at"]).dt.total_seconds() / 86400
    )

    # Семейства браузеров/устройств и маркеры автоматизации. Сырые строки
    # UA я в модель не подаю — только локально извлечённые признаки.
    ua = ev["user_agent"].fillna("").astype(str).str.lower()
    ua_flags = {
        "headless": r"headless|phantom|selenium|playwright|puppeteer",
        "okhttp": r"okhttp",
        "mobile": r"android|iphone|ipad|avito/",
        "linux": r"linux",
        "windows": r"windows",
        "mac": r"macintosh|mac os",
        "chrome": r"chrome/|chromium/",
        "firefox": r"firefox/",
        "yandex": r"yabrowser/",
    }
    for name, pattern in ua_flags.items():
        ev[f"ua_{name}"] = ua.str.contains(pattern, regex=True).astype("int8")
        features[f"ua_{name}_fraction"] = ev.groupby("cookie_id", sort=False)[f"ua_{name}"].mean()
    # Safari распознаётся только если в UA нет Chrome/Chromium/Android —
    # иначе почти все мобильные браузеры записывались бы как Safari.
    ev["ua_safari"] = (
        ua.str.contains(r"safari/") & ~ua.str.contains(r"chrome/|chromium/|android")
    ).astype("int8")
    features["ua_safari_fraction"] = ev.groupby("cookie_id", sort=False)["ua_safari"].mean()
    ua_version = pd.to_numeric(
        ua.str.extract(r"(?:chrome/|chromium/|firefox/|version/)(\d+)", expand=False),
        errors="coerce",
    )
    ev["ua_version"] = ua_version
    features["ua_version_mean"] = ev.groupby("cookie_id", sort=False)["ua_version"].mean()
    features["ua_version_std"] = ev.groupby("cookie_id", sort=False)["ua_version"].std()
    ua_text = ev["user_agent"].fillna("").astype(str).str.lower()
    ev["ua_missing"] = ua_text.str.strip().eq("").astype("int8")
    features["ua_missing_fraction"] = ev.groupby("cookie_id", sort=False)["ua_missing"].mean()
    # Экспериментально: конфликт между UA и заявленной платформой
    # (например, UA мобильный, а platform=web) — классический след подделки.
    ua_platform_conflict = (
        (ua_text.str.contains(r"android|iphone|ipad") & ev["platform_norm"].eq("web"))
        | (ua_text.str.contains(r"windows|macintosh|x11") & ev["platform_norm"].isin(["android", "ios", "iphone"]))
    )
    features["ua_platform_conflict_fraction"] = ua_platform_conflict.groupby(ev["cookie_id"]).mean()

    # Статистика курсора: движение и скорость между наблюдаемыми точками.
    # Координаты есть не у всех событий, поэтому считаю только по наблюденным.
    for column in ("pointer_x", "pointer_y"):
        observed = ev.loc[ev[column].notna(), ["cookie_id", column]].copy()
        observed[f"{column}_step"] = observed.groupby("cookie_id", sort=False)[column].diff().abs()
        observed_grouped = observed.groupby("cookie_id", sort=False)[f"{column}_step"]
        features[f"{column}_step_mean"] = observed_grouped.mean()
        features[f"{column}_step_std"] = observed_grouped.std()
        features[f"{column}_range"] = grouped[column].max() - grouped[column].min()
        features[f"{column}_nunique"] = grouped[column].nunique()
        features[f"{column}_observed_fraction"] = ev.groupby("cookie_id", sort=False)[column].count() / features["n_events"]
    pointer_observed = ev.dropna(subset=["pointer_x", "pointer_y"])
    pointer_pair_counts = pointer_observed.groupby(["cookie_id", "pointer_x", "pointer_y"], observed=True).size()
    pointer_pair_total = pointer_pair_counts.groupby(level=0).sum()
    features["pointer_dominant_point_fraction"] = pointer_pair_counts.groupby(level=0).max() / pointer_pair_total
    zero_point = pointer_observed["pointer_x"].eq(0) & pointer_observed["pointer_y"].eq(0)
    features["pointer_zero_point_fraction"] = zero_point.groupby(pointer_observed["cookie_id"]).mean()

    # Путь курсора: шаги, скорости, смены направления. Идея — у живого
    # человека траектория "нервная", у скрипта прямые длинные броски.
    pointer = ev.dropna(subset=["pointer_x", "pointer_y"])[
        ["cookie_id", "event_ts", "pointer_x", "pointer_y"]
    ].copy()
    if not pointer.empty:
        pointer_group = pointer.groupby("cookie_id", sort=False)
        dx = pointer_group["pointer_x"].diff()
        dy = pointer_group["pointer_y"].diff()
        dt = pointer_group["event_ts"].diff().dt.total_seconds()
        pointer["step_distance"] = np.sqrt(dx.pow(2) + dy.pow(2))
        pointer["step_speed"] = pointer["step_distance"] / dt.where(dt.gt(0))
        pointer["x_direction"] = np.sign(dx)
        pointer["y_direction"] = np.sign(dy)
        pointer_features = pointer.groupby("cookie_id", sort=False).agg(
            pointer_samples=("pointer_x", "size"),
            pointer_path_length=("step_distance", "sum"),
            pointer_step_distance_mean=("step_distance", "mean"),
            pointer_step_distance_std=("step_distance", "std"),
            pointer_speed_mean=("step_speed", "mean"),
            pointer_speed_p90=("step_speed", lambda values: values.quantile(0.9)),
            pointer_x_direction_changes=("x_direction", lambda values: values.diff().abs().gt(1).sum()),
            pointer_y_direction_changes=("y_direction", lambda values: values.diff().abs().gt(1).sum()),
        )
        features = features.join(pointer_features, how="left")
        # "Прямолинейность" пути: смещение из начала в конец к длине пути.
        displacement = np.sqrt(
            (pointer_group["pointer_x"].last() - pointer_group["pointer_x"].first()).pow(2)
            + (pointer_group["pointer_y"].last() - pointer_group["pointer_y"].first()).pow(2)
        )
        features["pointer_straightness"] = displacement / features["pointer_path_length"].replace(0, np.nan)
        features["pointer_sample_fraction"] = features["pointer_samples"] / features["n_events"]

    # Сводные таблицы счётчиков по всем перечисленным измерениям сразу.
    pivots = [
        _count_pivot(ev, "event_name", "event"),
        _count_pivot(ev, "eid", "eid"),
        _count_pivot(ev, "platform_norm", "platform"),
        _count_pivot(ev, "item_category", "category"),
        _count_pivot(ev, "item_location", "location"),
        _count_pivot(ev, "seller_type", "seller"),
        _count_pivot(ev, "hour_bin", "hour_bin"),
        _count_pivot(ev, "hour", "hour"),
    ]

    # Переходы между соседними событиями — компактный способ закинуть
    # в табличную модель немного последовательностной информации.
    # Переходы с нулевым интервалом отбрасываю (gt(0)): это пачки событий
    # с одним timestamp, порядок внутри них случаен, перехода как такового нет.
    ev["next_event"] = ev.groupby("cookie_id", sort=False)["event_name"].shift(-1)
    ev["next_gap_sec"] = ev.groupby("cookie_id", sort=False)["gap_sec"].shift(-1)
    transitions = ev.loc[
        ev["next_event"].notna() & ev["next_gap_sec"].gt(0),
        ["cookie_id", "event_name", "next_event"],
    ].copy()
    transitions["transition"] = transitions["event_name"].astype(str) + "_to_" + transitions["next_event"].astype(str)
    if not transitions.empty:
        transition_counts = transitions.groupby(["cookie_id", "transition"], observed=True).size().unstack(fill_value=0)
        transition_counts.columns = [f"transition__{str(value)}" for value in transition_counts.columns]
        pivots.append(transition_counts)

    # --- Экспериментальный блок: упорядоченная навигация по страницам ---
    features = features.copy()  # дефрагментация перед добавлением сводок
    search_pages = ev.loc[ev["search_page"].notna(), ["cookie_id", "event_ts", "search_page"]].copy()
    search_pages = search_pages.sort_values(["cookie_id", "event_ts"], kind="mergesort")
    search_pages["page_step"] = search_pages.groupby("cookie_id", sort=False)["search_page"].diff()
    page_group = search_pages.groupby("cookie_id", sort=False)
    features["page_step_mean"] = page_group["page_step"].mean()
    features["page_step_std"] = page_group["page_step"].std()
    features["page_forward_fraction"] = search_pages.assign(flag=search_pages["page_step"].gt(0)).groupby("cookie_id")["flag"].mean()
    features["page_sequential_fraction"] = search_pages.assign(flag=search_pages["page_step"].eq(1)).groupby("cookie_id")["flag"].mean()
    features["page_reset_fraction"] = search_pages.assign(flag=search_pages["page_step"].lt(0)).groupby("cookie_id")["flag"].mean()
    features["search_page_event_count"] = page_group.size()
    features = features.join(pivots, how="left").copy()
    # Нормировки действий на просмотры объявлений и на выдачу: абсолютные
    # счётчики зависят от длины сессии, а доли — нет.
    item_views = features.get("event__item_view", pd.Series(0, index=features.index)).replace(0, np.nan)
    search_views = features.get("event__search_results_view", pd.Series(0, index=features.index)).replace(0, np.nan)
    for name in ("photo_swipe", "seller_page_view", "favorite_add", "contact_phone_show",
                 "contact_chat_open", "contact_message_sent", "captcha_shown", "login"):
        event_col = f"event__{name}"
        if event_col in features:
            features[f"{name}_per_item_view"] = features[event_col] / item_views
            features[f"{name}_per_search_view"] = features[event_col] / search_views
    share_features = {}
    for prefix in ("event__", "platform__", "category__", "location__", "seller__"):
        for column in [name for name in features.columns if name.startswith(prefix)]:
            share_features[f"{column}_share"] = features[column] / features["n_events"]
    if share_features:
        features = pd.concat([features, pd.DataFrame(share_features, index=features.index)], axis=1)

    # Финал: выравниваю строки по порядку cookie_id из metadata. Пропуски
    # после агрегации означают "не наблюдалось" и осознанно заполняются
    # нулями; бесконечности (деление на ноль выше) тоже убираю.
    # float32 вместо float64: матрица вдвое легче, а точности хватает с запасом.
    features = features.reindex(metadata["cookie_id"].tolist())
    features.index = pd.Index(metadata["cookie_id"].tolist(), name="cookie_id")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    features = features.astype(np.float32)
    if not features.index.equals(pd.Index(metadata["cookie_id"], name="cookie_id")):
        raise AssertionError("Строки признаков не совпадают с metadata")
    return features
