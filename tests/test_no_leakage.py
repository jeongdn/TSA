"""누수 테스트: 기준 시점 T 이후의 값을 엉터리로 바꿔도 T에서 낸 예측이 똑같아야 한다.

실행: python -m tests.test_no_leakage   (pytest로 돌려도 된다)
검사하는 것
1. 상위국 선정: 선정 기간(2022-02) 이후 물량을 바꿔도 같은 나라가 뽑힌다.
2. 학습 행 필터: 't+h <= T' 필터가 목표 월 T 이하만 남긴다 (필터 자체의 재확인일 뿐이다.
   실제 학습에 미래 행이 들어가지 않는다는 근거는 3번이다).
3. 예측: 계절 나이브, 12개월 평균, ETS, LightGBM(점, q10, q90), MASE 분모, 구간 규모가
   T 이후 물량·환율·유가를 바꿔도 비트 단위로 같다. LightGBM은 동결한 설정과 다른 설정 하나로 확인한다.
4. 온라인 conformal: 목표 월이 T보다 뒤인 점수를 바꿔도 보정폭이 같다.
5. 계층 조정(7번): 합계 노드의 ETS 기준 예측이 T 이후 값과 무관하고, W의 재료인 검증 구간 오차가
   테스트 구간(2023-08 이후) 값을 바꿔도 같다.
한계: 기준 시점 두 개(검증 첫 시점, 테스트 여섯 번째 시점)에서만 확인한다.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tsa import backtest, baselines, config, conformal, hierarchy, lgbm, panel, stages


def poisoned(D, T):
    """T 이후 물량을 10배 + 12345로, 환율·유가를 달마다 다른 배수(2~9배)로 바꾼 사본.
    달마다 배수가 달라서 T 이후 두 달 사이의 변화율까지 바뀐다."""
    Y = D.Y.copy()
    Y[:, T + 1:] = Y[:, T + 1:] * 10 + 12345
    ext = D.ext.copy()
    late = ext.index > D.months[T]
    ext.loc[late] = ext.loc[late].to_numpy() * np.random.default_rng(T).uniform(2, 9, size=(late.sum(), 2))
    return Y, ext


def test_no_leakage():
    D = stages.load()

    # 1) 상위국 선정
    df = D.df.copy()
    late_part = (df["period"] > config.SELECT_END) & (df.partner != 0)
    df.loc[late_part, "kg"] *= np.random.default_rng(0).uniform(0, 50, late_part.sum())
    # World는 아주 크게 둬서 '기타'가 음수가 되지 않게 한다
    df.loc[(df["period"] > config.SELECT_END) & (df.partner == 0), "kg"] = 1e12
    _, info_p = panel.build(df)
    assert list(info_p["series"]) == list(D.info["series"]), "상위국 선정이 미래 값에 따라 바뀜"

    param_sets = [lgbm.sample_configs(1, config.SEED)[0]]
    if config.FROZEN_PATH.exists():
        param_sets.append(json.loads(config.FROZEN_PATH.read_text(encoding="utf-8"))["lgbm"]["params"])
    feats = lgbm.BASE_FEATURES + lgbm.EXT_FEATURES
    for T in [D.val_idx[0], D.test_idx[5]]:
        Yp, extp = poisoned(D, T)
        rows_p = lgbm.build_rows(Yp, D.months, D.info, extp)

        # 2) 학습 행의 목표 월
        train = D.rows[D.rows["t"] + D.rows["h"] <= T]
        assert (train["t"] + train["h"]).max() <= T

        # 3) 예측이 같은지
        for fn in (baselines.seasonal_naive, baselines.mean12):
            assert np.array_equal(fn(D.Y, T), fn(Yp, T)), fn.__name__
        assert np.array_equal(baselines.ets(D.Y, T)[0], baselines.ets(Yp, T)[0]), "ETS"
        assert np.array_equal(backtest.mase_scale(D.Y, T), backtest.mase_scale(Yp, T))
        assert np.array_equal(backtest.origin_scale(D.Y, T), backtest.origin_scale(Yp, T))
        S, _ = hierarchy.summing_matrix(D.products)
        assert np.array_equal(baselines.ets((S @ D.Y)[:3], T)[0], baselines.ets((S @ Yp)[:3], T)[0]), "합계 ETS"
        for params in param_sets:
            for obj, a in [("l1", None), ("quantile", 0.1), ("quantile", 0.9)]:
                p1, _ = lgbm.fit_predict(D.rows, T, params, feats, obj, a)
                p2, _ = lgbm.fit_predict(rows_p, T, params, feats, obj, a)
                assert np.array_equal(p1["yhat"].to_numpy(), p2["yhat"].to_numpy()), f"LightGBM {obj} {a}"
        print(f"  기준 시점 {D.months[T]}: 통과")

    # 4) 온라인 conformal: 목표 월이 T 뒤인 점수는 쓰지 않아야 한다
    rng = np.random.default_rng(1)
    cal = pd.DataFrame({"T": np.repeat(np.arange(100, 130), 6), "h": np.tile(np.arange(1, 7), 30)})
    cal["y"], cal["lo"], cal["hi"], cal["scale"] = rng.normal(10, 2, len(cal)), 8.0, 12.0, 1.0
    T = 115
    cal_p = cal.copy()
    cal_p.loc[cal_p["T"] + cal_p["h"] > T, "y"] = 1e9
    assert conformal.online_thresholds(cal, T) == conformal.online_thresholds(cal_p, T)

    # 5) 계층 조정: W의 재료(검증 구간 오차)는 테스트 구간 값을 바꿔도 같아야 한다
    val_path = config.OUT_DIR / "val_forecasts.csv.gz"
    if val_path.exists():          # val 단계를 먼저 돌린 컴퓨터에서만 확인할 수 있다
        Yc, _ = poisoned(D, D.months.get_loc(config.CUTOFF))
        errs = []
        for d in (D, SimpleNamespace(**{**vars(D), "Y": Yc})):
            HD = stages.hier(d)
            agg, _ = stages.agg_ets(HD, D.val_idx)
            bottom = stages.stored(val_path, "LGBM", len(D.series))
            errs.append(stages.val_errors(stages.hier_base(HD, D.val_idx, agg, bottom)))
        assert np.array_equal(*errs), "검증 구간 오차가 테스트 구간 값에 따라 바뀜"
        print("  계층 조정 W의 재료: 통과")
    print("누수 테스트 통과")


if __name__ == "__main__":
    test_no_leakage()
