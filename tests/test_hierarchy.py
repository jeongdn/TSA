"""계층 조정(MinT) 단위 테스트: 합산 행렬, 합이 맞는지, 투영 성질, BU·OLS 식, 축소 추정.

실행: python -m tests.test_hierarchy   (pytest로 돌려도 된다)
"""
import numpy as np

from tsa import hierarchy

PRODUCTS = np.array(["ABS"] * 9 + ["PC"] * 9)


def test_summing_matrix():
    S, top = hierarchy.summing_matrix(PRODUCTS)
    assert S.shape == (21, 18) and top == ["TOTAL", "ABS", "PC"]
    assert (S[0] == 1).all() and S[1, :9].all() and not S[1, 9:].any() and S[2, 9:].all()
    assert np.array_equal(S[3:], np.eye(18))


def test_reconciled_forecasts_add_up():
    """어떤 방법이든 조정 뒤에는 전체 = ABS + PC, 제품 합계 = 하위 합 (0으로 자른 뒤에도)."""
    rng = np.random.default_rng(0)
    S, _ = hierarchy.summing_matrix(PRODUCTS)
    E = rng.normal(size=(72, 21)) * rng.uniform(1, 50, 21)
    base = rng.uniform(0, 500, size=(30, 21))          # 합이 맞지 않는 기준 예측
    for m in hierarchy.METHODS:
        W, _ = hierarchy.w_matrix(m, S, E)
        rec, _ = hierarchy.reconcile(base, S, hierarchy.g_matrix(S, W))
        assert np.allclose(rec[:, 0], rec[:, 1] + rec[:, 2]), m
        assert np.allclose(rec[:, 1], rec[:, 3:12].sum(axis=1)), m
        assert np.allclose(rec[:, 2], rec[:, 12:].sum(axis=1)), m
        assert (rec >= 0).all(), m


def test_coherent_input_is_unchanged():
    """이미 합이 맞는 예측은 그대로 둔다 (S G S = S: 조정은 '합이 맞는 예측' 위로의 투영)."""
    rng = np.random.default_rng(1)
    S, _ = hierarchy.summing_matrix(PRODUCTS)
    E = rng.normal(size=(72, 21))
    coherent = rng.uniform(1, 100, size=(5, 18)) @ S.T
    for m in hierarchy.METHODS:
        W, _ = hierarchy.w_matrix(m, S, E)
        rec, neg = hierarchy.reconcile(coherent, S, hierarchy.g_matrix(S, W))
        assert np.allclose(rec, coherent) and neg == 0.0, m


def test_bu_and_ols():
    S, _ = hierarchy.summing_matrix(PRODUCTS)
    base = np.random.default_rng(2).uniform(1, 100, size=(4, 21))
    rec_bu, _ = hierarchy.reconcile(base, S, hierarchy.g_matrix(S))
    assert np.allclose(rec_bu[:, 3:], base[:, 3:])                    # BU는 하위 예측을 바꾸지 않는다
    assert np.allclose(hierarchy.g_matrix(S, np.eye(21)), np.linalg.inv(S.T @ S) @ S.T)   # W = I -> OLS 식


def test_shrinkage():
    """축소 강도는 0~1. 노드끼리 상관이 없으면 대각 쪽으로 세게 당기고(강도 큼), 상관이 강하면 적게 당긴다."""
    rng = np.random.default_rng(3)
    _, lam_indep = hierarchy.shrink_cov(rng.normal(size=(5000, 5)))
    common = rng.normal(size=(5000, 1))
    W, lam_corr = hierarchy.shrink_cov(common + 0.3 * rng.normal(size=(5000, 5)))
    assert 0 <= lam_corr < 0.05 < lam_indep <= 1
    assert np.allclose(W, W.T) and np.all(np.linalg.eigvalsh(W) > 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: 통과")
