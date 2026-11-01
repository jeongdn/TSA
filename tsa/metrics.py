"""평가 지표.

- WAPE = sum|y - ŷ| / sum y : 전체 물량 대비 틀린 물량. 큰 시리즈가 더 큰 비중을 갖는다.
- Bias = sum(ŷ - y) / sum y : +면 과대 예측, -면 과소 예측.
- MASE = mean(|y - ŷ| / q) : q = 기준 시점까지 계절 나이브(12개월 전 값)의 평균 절대 오차.
  시리즈 크기와 무관하게 '계절 나이브 대비 몇 배 틀렸나'를 본다. 1보다 작으면 학습 기간의 계절 나이브보다 나음.
"""
import numpy as np
import pandas as pd


def wape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return np.abs(y - yhat).sum() / np.abs(y).sum()


def bias(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return (yhat - y).sum() / np.abs(y).sum()


def _row(g):
    se = np.abs(g["y"] - g["yhat"]) / g["q"]       # 예측 하나하나의 척도 없는 오차
    return pd.Series({"WAPE": wape(g["y"], g["yhat"]), "Bias": bias(g["y"], g["yhat"]),
                      "MASE": se.mean(), "MASE_median": se.median(), "n": len(g)})


def summarize(fc, by=None):
    """fc: 예측 표(y, yhat, q 열 필요). by 열로 묶어 지표를 계산한다."""
    cols = ["y", "yhat", "q"]
    if by is None:
        return _row(fc[cols])
    return fc.groupby(by)[cols].apply(_row)
