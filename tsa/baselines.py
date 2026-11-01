"""기준선 모델. 모두 Y[:, :T+1] (기준 시점 T까지의 값)만 쓴다.

Y는 (시리즈 수, 월 수) 배열, T는 기준 시점의 월 인덱스, 결과는 (시리즈 수, H) 배열.
"""
import numpy as np
from statsforecast.models import AutoETS

from . import config


def seasonal_naive(Y, T):
    """h개월 앞 = 목표 월의 1년 전 값 (h <= 12라서 항상 T 이하)."""
    return np.stack([Y[:, T + h - 12] for h in range(1, config.H + 1)], axis=1)


def mean12(Y, T):
    """참고용: 최근 12개월 평균을 모든 h에 그대로 쓴다. LightGBM의 정규화 기준과 같은 값."""
    m = Y[:, T - 11: T + 1].mean(axis=1)
    return np.repeat(m[:, None], config.H, axis=1)


def ets(Y, T):
    """시리즈마다 AutoETS를 T까지의 값으로 적합한다.
    오차·추세·계절을 더하기/곱하기/없음 중에서 AICc가 가장 낮은 조합으로 자동 선택한다.
    적합이 실패하면 계절 나이브로 대신하고 실패 횟수를 센다."""
    out = np.empty((Y.shape[0], config.H))
    fails = 0
    for i in range(Y.shape[0]):
        try:
            f = AutoETS(season_length=12).forecast(y=Y[i, : T + 1].astype(float), h=config.H)["mean"]
            if not np.all(np.isfinite(f)):
                raise ValueError("non-finite forecast")
            out[i] = f
        except Exception:
            out[i] = seasonal_naive(Y[i: i + 1], T)[0]
            fails += 1
    return np.clip(out, 0.0, None), fails
