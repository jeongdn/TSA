"""원형 블록 부트스트랩으로 'LightGBM이 기준선보다 WAPE가 낮은 재표집의 비율'을 구한다.

- 재표집 단위는 기준 시점. 한 기준 시점의 18개 시리즈 x 6개 시점 오차를 통째로 묶어 둔다
  (같은 달에 여러 나라가 함께 흔들리는 상관을 보존).
- 이웃한 기준 시점은 예측 대상 월이 최대 5개월 겹쳐 오차가 닮는다. 그래서 길이 6짜리 블록으로 뽑는다.
- 짝 비교: 모든 모델에 같은 재표집을 쓴다. 실제값 합이 같으므로 WAPE 비교 = 오차 합 비교.
- 블록 수가 적으면 부트스트랩 분산이 (1 - b/n)배로 작게 나온다 -> sqrt(n/(n-b))배로 넓혀 보정한다.
"""
import numpy as np

from . import config


def block_indices(n, b, rng):
    """길이 b 블록의 시작점을 무작위로 뽑아 이어 붙인다. 끝을 넘으면 처음으로 돌아간다(원형)."""
    k = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=k)
    return ((starts[:, None] + np.arange(b)) % n).ravel()[:n]


def resample_sums(arrays, b, B, seed):
    """arrays: 이름 -> 기준 시점별 값(길이 n). 같은 재표집으로 각 배열의 합을 B번 구한다."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(arrays.values())))
    out = {k: np.empty(B) for k in arrays}
    for i in range(B):
        idx = block_indices(n, b, rng)
        for k, v in arrays.items():
            out[k][i] = v[idx].sum()
    return out


def widen(boot, hat, n, b):
    """작은 표본 보정: 관측값을 중심으로 부트스트랩 편차를 sqrt(n/(n-b))배 넓힌다."""
    return hat + np.sqrt(n / (n - b)) * (boot - hat)


def compare(A_model, A_bases, b=config.BOOT_BLOCK, B=config.BOOT_B, seed=config.SEED, correct=True):
    """A_model, A_bases[이름]: 기준 시점별 절대 오차 합.
    반환: 기준선마다 P(모델 오차 합 < 기준선 오차 합)와 상대 WAPE 감소율 분포, 그리고 P(모든 기준선보다 낮음)."""
    n = len(A_model)
    sums = resample_sums({"_model": A_model, **A_bases}, b, B, seed)
    out, win_all = {}, np.ones(B, dtype=bool)
    for name, A in A_bases.items():
        d_hat = A_model.sum() - A.sum()                      # < 0 이면 모델이 나음
        d = sums["_model"] - sums[name]
        rr_hat = 1.0 - A_model.sum() / A.sum()               # 상대 WAPE 감소율 (+면 모델이 나음)
        rr = 1.0 - sums["_model"] / sums[name]
        if correct:
            d, rr = widen(d, d_hat, n, b), widen(rr, rr_hat, n, b)
        win = d < 0
        win_all &= win
        out[name] = {
            "p_better": float(win.mean()),
            "rel_reduction": float(rr_hat),
            "rel_reduction_q10": float(np.percentile(rr, 10)),   # 단측 90% 하한: > 0 이면 P > 0.9
            "rel_reduction_q05": float(np.percentile(rr, 5)),
            "rel_reduction_q95": float(np.percentile(rr, 95)),
        }
    out["p_better_all"] = float(win_all.mean())
    return out


def compare_pairs(pairs, b=config.BOOT_BLOCK, B=config.BOOT_B, seed=config.SEED, correct=True):
    """pairs: 이름 -> (모델 오차 합, 비교 대상 오차 합) (기준 시점별). compare와 같은 계산을 짝마다 하고,
    P(모든 짝에서 모델이 나음)도 구한다. 계층 조정에서 '세 수준 모두에서 나은가'를 볼 때 쓴다."""
    n = len(next(iter(pairs.values()))[0])
    arrays = {}
    for k, (a, c) in pairs.items():
        arrays[f"{k}|model"], arrays[f"{k}|other"] = a, c
    sums = resample_sums(arrays, b, B, seed)
    out, win_all = {}, np.ones(B, dtype=bool)
    for k, (a, c) in pairs.items():
        d_hat, d = a.sum() - c.sum(), sums[f"{k}|model"] - sums[f"{k}|other"]
        rr_hat, rr = 1.0 - a.sum() / c.sum(), 1.0 - sums[f"{k}|model"] / sums[f"{k}|other"]
        if correct:
            d, rr = widen(d, d_hat, n, b), widen(rr, rr_hat, n, b)
        win = d < 0
        win_all &= win
        out[k] = {"p_better": float(win.mean()), "rel_reduction": float(rr_hat),
                  "rel_reduction_q05": float(np.percentile(rr, 5)), "rel_reduction_q95": float(np.percentile(rr, 95))}
    out["p_better_all"] = float(win_all.mean())
    return out


def ratio_interval(num, den, b=config.BOOT_BLOCK, B=config.BOOT_B, seed=config.SEED, level=0.90):
    """기준 시점별 분자·분모 합의 비율(예: 구간 적중률)의 부트스트랩 구간."""
    n = len(num)
    sums = resample_sums({"num": num, "den": den}, b, B, seed)
    hat = num.sum() / den.sum()
    boot = widen(sums["num"] / sums["den"], hat, n, b)
    lo, hi = np.percentile(boot, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(hat), float(lo), float(hi)
