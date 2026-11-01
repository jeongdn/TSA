"""CQR(conformalized quantile regression): 분위수 모델의 P10~P90 구간을 과거 오차로 고친다.

점수 = 실제값이 경계 밖으로 얼마나 나갔는지 (규모로 나눔, 안쪽이면 음수).
- 아래쪽 점수 s_lo = (lo - y) / scale, 위쪽 점수 s_hi = (y - hi) / scale
- 보정용 점수를 정렬해 ceil(0.9 (n+1))번째 값을 Q로 잡고, 새 구간 = [lo - Q_lo*scale, hi + Q_hi*scale]
- Q가 양수면 구간을 넓히고, 음수면 좁힌다. 아래·위를 따로 해서 각각 약 10%씩만 벗어나게 맞춘다.
보장은 '보정 점수와 새 점수가 교환 가능'할 때만 성립한다. 시계열은 그렇지 않으므로 경험적 확인으로 본다.
"""
import numpy as np

from . import config

LEVEL = 1.0 - config.ALPHA_LO    # 한쪽 경계마다 0.9


def scores(y, lo, hi, scale):
    return (lo - y) / scale, (y - hi) / scale


def threshold(s, level=LEVEL):
    """split conformal 임계값: 정렬한 점수의 ceil(level (n+1))번째. 그 순위가 n을 넘으면 무한대."""
    s = np.sort(np.asarray(s, float))
    k = int(np.ceil(level * (len(s) + 1)))
    return float(np.inf) if k > len(s) else float(s[k - 1])


def calibrate(cal):
    """cal: h, y, lo, hi, scale 열이 있는 보정용 표 -> {h: (Q_lo, Q_hi)}."""
    out = {}
    for h, g in cal.groupby("h"):
        s_lo, s_hi = scores(g["y"].to_numpy(), g["lo"].to_numpy(), g["hi"].to_numpy(), g["scale"].to_numpy())
        out[int(h)] = (threshold(s_lo), threshold(s_hi))
    return out


def apply(lo, hi, scale, q_lo, q_hi):
    return np.clip(lo - q_lo * scale, 0.0, None), hi + q_hi * scale


def online_thresholds(cal_all, T):
    """온라인 방식: 기준 시점 T에서는 '목표 월이 T 이하'인 (= 이미 실제값을 아는) 점수만 보정에 쓴다.
    cal_all: 검증 + 테스트 구간의 (T, h, y, lo, hi, scale) 표."""
    known = cal_all[cal_all["T"] + cal_all["h"] <= T]
    return calibrate(known)
