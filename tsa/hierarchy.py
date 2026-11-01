"""계층 조정(MinT): 하위 18개 예측의 합이 제품 합계·전체 합계 예측과 맞도록 고친다.

계층: 전체(1) -> 제품 ABS, PC(2) -> 제품 x 국가(18). 노드는 모두 21개.
S(합산 행렬, 21 x 18): 하위 18개 값에 곱하면 21개 노드 값이 된다 (y_전체노드 = S @ y_하위).

노드마다 따로 만든 기준 예측 ŷ(21개)은 합이 맞지 않는다 (하위는 LightGBM, 합계는 ETS).
조정: 하위 예측 b = G ŷ 를 만들고 모든 노드를 S b 로 다시 더한다. 그러면 합이 항상 맞는다.
- BU (상향식): 하위 기준 예측을 그대로 더한다. 합계 노드의 기준 예측은 버린다.
- OLS: W = I. 모든 노드의 오차를 같은 크기로 본다.
- WLS_struct: W = diag(노드에 속한 하위 시리즈 수).
- WLS_var: W = diag(노드별 오차 제곱 평균).
- MinT_shrink: W = 노드 간 오차 공분산의 축소 추정.
W가 있는 방법은 G = (S' W^-1 S)^-1 S' W^-1. 오차가 크거나 다른 노드와 함께 흔들리는 노드의 예측을
덜 믿도록 21개 기준 예측을 가중 결합한 것이다 (조정 뒤 오차 분산의 합이 가장 작아지는 선형 결합).
"""
import numpy as np

METHODS = ["BU", "OLS", "WLS_struct", "WLS_var", "MinT_shrink"]
PRIMARY = "MinT_shrink"          # 계획 단계에서 정한 주 방법. 나머지는 비교용


def summing_matrix(products):
    """products: 하위 시리즈의 제품 이름(18개). 반환: S(21 x 18), 합계 노드 이름 ['TOTAL', 'ABS', 'PC']."""
    products = np.asarray(products)
    prods = sorted(set(products))
    top = [np.ones(len(products))] + [(products == p).astype(float) for p in prods]
    return np.vstack(top + [np.eye(len(products))]), ["TOTAL"] + prods


def shrink_cov(E):
    """E: (표본 수, 노드 수) 기준 예측 오차. Schäfer-Strimmer 축소 추정 (R hts 패키지 MinT shrink와 같은 식).
    표본이 노드 수에 비해 적으면 표본 공분산이 불안정하므로, 대각 행렬(노드끼리 상관 없음) 쪽으로 lam만큼 당긴다.
    lam = (상관계수 추정값들의 분산 합) / (상관계수 제곱 합): 상관이 잡음에 묻힐수록 1에 가깝다."""
    n = E.shape[0]
    cov = E.T @ E / n                           # 평균을 빼지 않는다: 한쪽으로 치우친 것도 오차로 본다
    sd = np.sqrt(np.diag(cov))
    Z = E / sd
    corr = cov / np.outer(sd, sd)
    var_r = ((Z ** 2).T @ (Z ** 2) - (Z.T @ Z) ** 2 / n) / (n * (n - 1))
    off = ~np.eye(len(sd), dtype=bool)
    lam = float(np.clip(var_r[off].sum() / (corr[off] ** 2).sum(), 0.0, 1.0))
    return lam * np.diag(np.diag(cov)) + (1 - lam) * cov, lam


def w_matrix(method, S, E):
    """방법별 W와 축소 강도(MinT_shrink만). E: 검증 구간 기준 예측 오차 (표본 수 x 21). BU는 W가 없다."""
    if method == "BU":
        return None, None
    if method == "OLS":
        return np.eye(S.shape[0]), None
    if method == "WLS_struct":
        return np.diag(S.sum(axis=1)), None
    if method == "WLS_var":
        return np.diag((E ** 2).mean(axis=0)), None
    if method == "MinT_shrink":
        return shrink_cov(E)
    raise ValueError(method)


def g_matrix(S, W=None):
    """G (18 x 21): 조정된 하위 예측 = G @ 기준 예측(21개). W가 없으면 BU (하위 기준 예측만 고른다)."""
    m, n_b = S.shape
    if W is None:
        return np.hstack([np.zeros((n_b, m - n_b)), np.eye(n_b)])
    Wi_S = np.linalg.solve(W, S)                  # W^-1 S
    return np.linalg.solve(S.T @ Wi_S, Wi_S.T)    # (S' W^-1 S)^-1 S' W^-1


def reconcile(base, S, G):
    """base: (k, 21) 기준 예측 -> (k, 21) 조정 예측.
    음수가 된 하위 예측은 0으로 자른 뒤 다시 더한다 (자른 뒤에도 합은 맞는다).
    반환: 조정 예측, 0으로 자른 하위 예측의 비율."""
    b = base @ G.T
    return np.clip(b, 0.0, None) @ S.T, float((b < 0).mean())
