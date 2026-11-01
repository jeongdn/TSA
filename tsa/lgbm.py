"""LightGBM 글로벌 모델: 18개 시리즈와 1~6개월 앞을 '한 모델'로 학습한다 (직접 다중 시점 방식).

한 행 = (시리즈 s, 기준월 t, 시점 h). 피처는 t 이하의 값만, 목표는 y[t+h].
기준 시점 T에서는 '목표 월 t+h <= T'인 행으로만 학습하고, t = T인 6개 행(h=1..6)을 예측한다.
시리즈마다 크기가 수십 배 달라서 목표와 래그를 '최근 12개월 평균(scale)'으로 나눈 값으로 학습한다.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config

EPS = 1.0           # 규모 하한(톤). 0으로 나누는 것을 막는다
T_MIN = 23          # 작년 12개월 평균(t-23 ~ t-12)이 있어야 해서 기준월 인덱스 23(2016-12)부터 행을 만든다

BASE_FEATURES = ([f"lag{k}" for k in range(12)]
                 + ["mean3", "mean6", "prev_year", "same_month_1y", "same_month_2y",
                    "month", "h", "product", "country", "log_scale"])
EXT_FEATURES = ["fx_chg1", "fx_chg12", "brent_chg1", "brent_chg12"]

# 랜덤 탐색 공간 (조합 972개 중 30개를 시드 고정으로 뽑는다)
SPACE = {
    "num_leaves": [7, 15, 31],
    "min_child_samples": [10, 20, 50, 100],
    "learning_rate": [0.02, 0.05, 0.1],
    "n_estimators": [200, 500, 1000],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 10.0],
}


def sample_configs(n, seed):
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        c = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        key = tuple(c.values())
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def build_rows(Y, months, info, ext):
    """가능한 모든 (s, t, h) 행을 한 번에 만든다. 목표 y[t+h]가 데이터 밖이면 NaN."""
    n_s, n_t = Y.shape
    # 외부 지표 변화율: 기준월 t의 값과 1개월·12개월 전 값만 쓴다 (모두 t 이하)
    fx, br = ext["fx"], ext["brent"]
    e = {
        "fx_chg1": np.log(fx.loc[months].to_numpy() / fx.loc[months.shift(-1)].to_numpy()),
        "fx_chg12": np.log(fx.loc[months].to_numpy() / fx.loc[months.shift(-12)].to_numpy()),
        "brent_chg1": np.log(br.loc[months].to_numpy() / br.loc[months.shift(-1)].to_numpy()),
        "brent_chg12": np.log(br.loc[months].to_numpy() / br.loc[months.shift(-12)].to_numpy()),
    }
    parts = []
    for t in range(T_MIN, n_t):
        scale = Y[:, t - 11: t + 1].mean(axis=1) + EPS          # 최근 12개월 평균
        base = {f"lag{k}": Y[:, t - k] / scale for k in range(12)}
        base["mean3"] = Y[:, t - 2: t + 1].mean(axis=1) / scale
        base["mean6"] = Y[:, t - 5: t + 1].mean(axis=1) / scale
        base["prev_year"] = Y[:, t - 23: t - 11].mean(axis=1) / scale   # 1년 전 12개월 평균 -> 추세
        for name, arr in e.items():
            base[name] = np.full(n_s, arr[t])
        for h in range(1, config.H + 1):
            d = dict(base)
            d["same_month_1y"] = Y[:, t + h - 12] / scale          # 목표 월의 1년 전 (t+h-12 <= t)
            d["same_month_2y"] = Y[:, t + h - 24] / scale          # 목표 월의 2년 전
            d["month"] = np.full(n_s, (months[t] + h).month)       # 목표 월 (1~12)
            d["h"] = np.full(n_s, h)
            d["s"] = np.arange(n_s)
            d["t"] = np.full(n_s, t)
            d["scale"] = scale
            d["log_scale"] = np.log(scale)
            d["product"] = info["product"].to_numpy()
            d["country"] = info["country"].to_numpy()
            d["y"] = Y[:, t + h] if t + h < n_t else np.full(n_s, np.nan)
            parts.append(pd.DataFrame(d))
    rows = pd.concat(parts, ignore_index=True)
    rows["product"] = rows["product"].astype(pd.CategoricalDtype(sorted(info["product"].unique())))
    rows["country"] = rows["country"].astype(pd.CategoricalDtype(sorted(info["country"].unique())))
    rows["r"] = rows["y"] / rows["scale"]       # 학습 목표: 최근 12개월 평균 대비 몇 배인가
    return rows


def make_model(params, objective="l1", alpha=None):
    kw = dict(objective=objective, random_state=config.SEED, n_jobs=config.LGB_THREADS,
              deterministic=True, force_row_wise=True, verbose=-1, **params)
    if objective == "quantile":
        kw["alpha"] = alpha
    return lgb.LGBMRegressor(**kw)


def fit_predict(rows, T, params, features, objective="l1", alpha=None):
    """기준 시점 T에서 학습하고 예측한다. 반환: (s, h, yhat) 표와 학습된 모델."""
    train = rows[rows["t"] + rows["h"] <= T]           # 목표 월이 T 이하인 행만 = 미래 값 없음
    pred = rows[rows["t"] == T]
    # L1 손실 x 규모 가중 = 톤 단위 절대 오차 합 -> WAPE 분자와 같은 것을 줄인다.
    # 가중치를 평균 1로 맞춰야 reg_lambda 같은 정규화 값이 제 의미를 가진다
    w = train["scale"] / train["scale"].mean()
    model = make_model(params, objective, alpha)
    model.fit(train[features], train["r"], sample_weight=w)
    yhat = np.clip(model.predict(pred[features]), 0.0, None) * pred["scale"].to_numpy()
    return pred[["s", "h"]].assign(yhat=yhat).reset_index(drop=True), model


def to_array(pred, n_s):
    """(s, h, yhat) 표 -> (시리즈 수, H) 배열."""
    arr = np.full((n_s, config.H), np.nan)
    arr[pred["s"].to_numpy(), pred["h"].to_numpy() - 1] = pred["yhat"].to_numpy()
    assert np.isfinite(arr).all()
    return arr
