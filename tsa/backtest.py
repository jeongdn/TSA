"""롤링 원점 백테스트 도우미: (시리즈, h) 예측 배열을 실제값이 붙은 긴 표로 바꾼다."""
import numpy as np
import pandas as pd

from . import config, lgbm


def mase_scale(Y, T):
    """MASE 분모: T까지의 데이터에서 계절 나이브(12개월 전 값)의 평균 절대 오차 (시리즈별)."""
    return np.abs(Y[:, 12: T + 1] - Y[:, : T + 1 - 12]).mean(axis=1)


def origin_scale(Y, T):
    """기준 시점의 규모 = 최근 12개월 평균. 구간 보정 점수를 나누는 데 쓴다 (LightGBM과 같은 값)."""
    return Y[:, T - 11: T + 1].mean(axis=1) + lgbm.EPS


def to_frame(model, pred, D, T):
    """pred: (시리즈 수, H) 배열 -> 한 행이 예측 하나인 표."""
    n_s, H = pred.shape
    s = np.repeat(np.arange(n_s), H)
    h = np.tile(np.arange(1, H + 1), n_s)
    q = mase_scale(D.Y, T)
    assert (q > 0).all()
    return pd.DataFrame({
        "model": model,
        "series": D.series[s],
        "product": D.products[s],
        "s": s,
        "T": T,
        "origin": D.months[T],
        "h": h,
        "target": D.months[T + h],
        "y": D.Y[s, T + h],
        "yhat": pred.ravel(),
        "q": q[s],
        "scale": origin_scale(D.Y, T)[s],
    })


def per_origin(fc):
    """부트스트랩용: 기준 시점별 절대 오차 합(A_T)과 실제값 합(Y_T)."""
    g = fc.assign(ae=(fc["y"] - fc["yhat"]).abs()).groupby("T")
    return g["ae"].sum().to_numpy(), g["y"].sum().to_numpy()


def blend(fcs, weights, name):
    """같은 (시리즈, 기준 시점, h)의 예측을 가중 평균한다. fcs: 모델 이름 -> 예측 표."""
    key = ["series", "T", "h"]
    first = next(iter(weights))
    out = fcs[first].copy()
    out["yhat"] = 0.0
    for m, w in weights.items():
        f = fcs[m].set_index(key)["yhat"].reindex(out.set_index(key).index)
        assert f.notna().all()
        out["yhat"] += w * f.to_numpy()
    out["model"] = name
    return out
