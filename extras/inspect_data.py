"""Быстрый осмотр данных: размеры блоков, дисбаланс классов, артефакты.

Это тот самый скрипт, с которого начиналась вся работа: перед тем как
строить признаки, я хотела глазами увидеть размеры временных блоков,
дисбаланс классов и подозрительные места в событиях. Именно здесь
всплыли дубликаты строк, пляшущий регистр platform и тот факт, что
captcha_shown всегда происходит после конца окна.

Запуск из корня проекта: python extras/inspect_data.py
"""

import pandas as pd

train = pd.read_csv("data/train.csv", parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"])
test = pd.read_csv("data/test.csv", parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"])
events = pd.read_csv("data/events.csv.gz", parse_dates=["event_ts"])

print("train", train.shape, "test", test.shape, "events", events.shape)
print("target counts:\n", train["target"].value_counts())
day = train["window_start_ts"].dt.normalize()
print("train days:", day.min().date(), "->", day.max().date())
print("test days:", test["window_start_ts"].min().date(), "->", test["window_start_ts"].max().date())

# Блоки по дням: смотрю, сколько кук и позитивов приходится на каждый
# временной кусок — от этого зависело, как резать фолды валидации.
blocks = {
    "FIT   <=04-13": day.le(pd.Timestamp("2026-04-13")),
    "SEL   04-14..16": day.between(pd.Timestamp("2026-04-14"), pd.Timestamp("2026-04-16")),
    "EXAM  04-17..19": day.between(pd.Timestamp("2026-04-17"), pd.Timestamp("2026-04-19")),
}
for name, m in blocks.items():
    print(name, ": n=", int(m.sum()), " pos=", int(train.loc[m, "target"].sum()))

# Соотношение классов и его корень — оба варианта фигурируют в сетке
# весов классов (см. CLASS_WEIGHT_GRID в src/config.py).
ratio = (1 - train["target"].mean()) / train["target"].mean()
print("neg/pos ratio:", round(ratio, 2), " sqrt:", round(ratio ** 0.5, 2))

# Артефакты данных, описанные в REPORT.md (раздел про данные):
print("точных дубликатов строк событий:", int(events.duplicated().sum()))
meta = pd.concat([train[["cookie_id", "window_start_ts", "window_end_ts"]],
                  test[["cookie_id", "window_start_ts", "window_end_ts"]]])
m = events.merge(meta, on="cookie_id", how="inner")
inside = m["event_ts"].ge(m["window_start_ts"]) & m["event_ts"].lt(m["window_end_ts"])
cap = m["event_name"].eq("captcha_shown")
print("событий внутри окна: %.4f" % inside.mean())
print("captcha_shown:", int(cap.sum()), "всего, внутри окна:", int((cap & inside).sum()))
print("варианты platform:", sorted(events["platform"].dropna().unique()))
