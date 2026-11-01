"""검증 단계(val)와 테스트 단계(final).

val  : 검증 기준 시점 12개만 써서 LightGBM 설정, 외부 지표 사용 여부, 앙상블 가중치, 구간 보정폭을 고른다.
       고른 것은 frozen.json에 쓴다. 이 파일을 커밋하고 'freeze' 태그를 단 뒤에 final을 돌린다.
final: frozen.json을 읽어 테스트 기준 시점 24개를 한 번 채점한다. frozen.json이 없으면 돌지 않는다.
"""
import hashlib
import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from . import backtest, baselines, bootstrap, config, conformal, data_comtrade, data_fred, lgbm, metrics, panel

OUT = config.OUT_DIR
MODELS = ["SNaive", "ETS", "LGBM"]                       # 앙상블에 섞는 모델
GRID = [(a / 10, b / 10, round((10 - a - b) / 10, 1))    # 가중치 0.1 단위, 합 1 -> 66개
        for a in range(11) for b in range(11 - a)]
IV_COLS = ["series", "product", "s", "T", "origin", "h", "target", "y", "scale", "point", "lo", "mid", "hi",
           "reordered"]


# ---------------------------------------------------------------- 공통
def load(refresh=False):
    rows = data_comtrade.fetch_all(refresh=refresh)
    df = data_comtrade.to_frame(rows)
    check = data_comtrade.check(df)
    wide, info = panel.build(df)
    ext = data_fred.fetch(refresh=refresh)
    months = wide.index
    Y = wide.to_numpy().T.copy()                          # (시리즈, 월)
    D = SimpleNamespace(df=df, check=check, wide=wide, info=info, ext=ext, months=months, Y=Y,
                        series=info["series"].to_numpy(), products=info["product"].to_numpy())
    D.rows = lgbm.build_rows(Y, months, info, ext)
    D.val_idx = [months.get_loc(p) for p in config.VAL_ORIGINS]
    D.test_idx = [months.get_loc(p) for p in config.TEST_ORIGINS]
    return D


def content_hash(D):
    """원자료의 '값'만으로 만든 지문. 응답에 들어 있는 서버 처리 시간이나 FRED의 새 달처럼
    값과 무관한 차이에는 바뀌지 않아서, 새로 받은 자료가 동결 때와 같은 값인지 확인할 수 있다."""
    df = D.df.sort_values(["period", "hs", "partner"])[["period", "hs", "partner", "kg", "usd"]]
    text = df.astype(str).to_csv(index=False) + D.ext.loc[:config.END].astype(str).to_csv()
    return hashlib.sha256(text.encode()).hexdigest()


def run_baselines(D, origins):
    out, fails = [], 0
    for T in origins:
        out.append(backtest.to_frame("SNaive", baselines.seasonal_naive(D.Y, T), D, T))
        out.append(backtest.to_frame("Mean12", baselines.mean12(D.Y, T), D, T))
        f, n_fail = baselines.ets(D.Y, T)
        fails += n_fail
        out.append(backtest.to_frame("ETS", f, D, T))
    fc = pd.concat(out, ignore_index=True)
    return {m: g.reset_index(drop=True) for m, g in fc.groupby("model")}, fails


def run_lgbm(D, origins, params, features, name, objective="l1", alpha=None):
    """기준 시점마다 다시 학습해서 예측한다 (롤링 원점)."""
    out = []
    for T in origins:
        p, _ = lgbm.fit_predict(D.rows, T, params, features, objective, alpha)
        out.append(backtest.to_frame(name, lgbm.to_array(p, len(D.series)), D, T))
    return pd.concat(out, ignore_index=True)


def run_configs(D, origins, configs, features):
    """설정 여러 개를 프로세스 N_JOBS개로 나눠 돌린다 (설정마다 결과는 단독 실행과 같다)."""
    light = SimpleNamespace(Y=D.Y, months=D.months, series=D.series, products=D.products, rows=D.rows)
    return Parallel(n_jobs=config.N_JOBS)(
        delayed(run_lgbm)(light, origins, c, features, f"LGBM_c{i:02d}") for i, c in enumerate(configs))


def score(fc):
    return float(metrics.wape(fc["y"], fc["yhat"]))


def intervals(point, lo_fc, hi_fc):
    """점예측(P50)과 q10, q90을 한 표로. 따로 학습한 세 값이 엇갈리면 크기순으로 다시 놓아
    lo <= mid <= hi로 맞춘다. 채점한 점예측은 'point'에 그대로 남긴다."""
    key = ["series", "T", "h"]
    idx = point.set_index(key).index
    lo = lo_fc.set_index(key)["yhat"].reindex(idx).to_numpy()
    hi = hi_fc.set_index(key)["yhat"].reindex(idx).to_numpy()
    trio = np.sort(np.stack([lo, point["yhat"].to_numpy(), hi], axis=1), axis=1)
    out = point[["series", "product", "s", "T", "origin", "h", "target", "y", "scale"]].copy()
    out["point"] = point["yhat"].to_numpy()
    out["lo"], out["mid"], out["hi"] = trio[:, 0], trio[:, 1], trio[:, 2]
    out["reordered"] = (trio[:, 0] != lo) | (trio[:, 2] != hi)   # 정렬로 바뀐 행
    return out


def interval_stats(y, L, U):
    """구간 평가: 적중률, 아래·위로 벗어난 비율, 상대 폭, 상대 구간 점수(Winkler, 낮을수록 좋음)."""
    y, L, U = (np.asarray(a, float) for a in (y, L, U))
    below, above = y < L, (y > U) & ~(y < L)
    a = config.ALPHA_LO * 2                                 # 80% 구간 -> alpha = 0.2
    iscore = (U - L) + (2 / a) * (L - y) * below + (2 / a) * (y - U) * above
    return {"coverage": float(((y >= L) & (y <= U)).mean()), "below": float(below.mean()),
            "above": float(above.mean()), "width_rel": float((U - L).sum() / y.sum()),
            "interval_score_rel": float(iscore.sum() / y.sum()), "n": int(len(y))}


def model_tables(fcs):
    out = {}
    for m, fc in fcs.items():
        out[m] = {"all": metrics.summarize(fc).to_dict(),
                  "by_h": metrics.summarize(fc, "h").to_dict(orient="index"),
                  "by_product": metrics.summarize(fc, "product").to_dict(orient="index")}
    return out


def _json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------- 검증 단계
def stage_val(D):
    val = D.val_idx
    print(f"[val] 기준 시점 {D.months[val[0]]} ~ {D.months[val[-1]]} ({len(val)}개)")
    base, ets_fails = run_baselines(D, val)

    # 1) LightGBM 설정 30개 -> 검증 WAPE 최소
    configs = lgbm.sample_configs(config.N_CONFIGS, config.SEED)
    cfg_fc = run_configs(D, val, configs, lgbm.BASE_FEATURES)
    cfg_wape = [score(f) for f in cfg_fc]
    k = int(np.argmin(cfg_wape))
    print(f"[val] LightGBM 설정 {len(configs)}개 중 {k}번 선택 (검증 WAPE {cfg_wape[k]:.4f})")

    # 2) 외부 지표: 고른 설정에 환율·브렌트유 피처를 더한 것과 짝 비교. P >= 0.9일 때만 채택
    ext_fc = run_lgbm(D, val, configs[k], lgbm.BASE_FEATURES + lgbm.EXT_FEATURES, "LGBM_ext")
    ext_cmp = bootstrap.compare(backtest.per_origin(ext_fc)[0], {"without_ext": backtest.per_origin(cfg_fc[k])[0]})
    use_ext = ext_cmp["without_ext"]["p_better"] >= 0.9
    features = lgbm.BASE_FEATURES + (lgbm.EXT_FEATURES if use_ext else [])
    lgb_val = (ext_fc if use_ext else cfg_fc[k]).assign(model="LGBM")
    print(f"[val] 외부 지표 P(더 나음) = {ext_cmp['without_ext']['p_better']:.3f} -> 사용 {use_ext}")

    # 3) 앙상블 가중치: 고른 LightGBM과 두 기준선을 섞는 66가지 중 검증 WAPE 최소
    fcs = {**base, "LGBM": lgb_val}
    ens = {w: score(backtest.blend(fcs, dict(zip(MODELS, w)), "Ensemble")) for w in GRID}
    w_best = min(ens, key=ens.get)
    fcs["Ensemble"] = backtest.blend(fcs, dict(zip(MODELS, w_best)), "Ensemble")
    fcs["Ensemble_eq"] = backtest.blend(fcs, dict.fromkeys(MODELS, 1 / 3), "Ensemble_eq")
    print(f"[val] 앙상블 가중치 (SNaive, ETS, LGBM) = {w_best}")

    # 4) 분위수 모델(q10, q90)과 CQR 보정폭 (h별)
    lo = run_lgbm(D, val, configs[k], features, "q10", "quantile", config.ALPHA_LO)
    hi = run_lgbm(D, val, configs[k], features, "q90", "quantile", config.ALPHA_HI)
    iv = intervals(lgb_val, lo, hi)
    Q = conformal.calibrate(iv)

    # 5) 참고: 검증 구간 부트스트랩 (같은 구간에서 골랐으므로 낙관적)
    A = {m: backtest.per_origin(f)[0] for m, f in fcs.items()}
    val_boot = bootstrap.compare(A["LGBM"], {"SNaive": A["SNaive"], "ETS": A["ETS"]})

    manifest = {"comtrade": data_comtrade.manifest(), "fred": data_fred.manifest()}
    frozen = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "data": {"start": str(config.START), "end": str(config.END), "check": D.check,
                 "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()},
        "windows": {"select": [str(config.SELECT_END - config.SELECT_MONTHS + 1), str(config.SELECT_END)],
                    "val_origins": [str(config.VAL_ORIGINS[0]), str(config.VAL_ORIGINS[-1])],
                    "test_origins": [str(config.TEST_ORIGINS[0]), str(config.TEST_ORIGINS[-1])],
                    "cutoff": str(config.CUTOFF), "horizon": config.H},
        "series": list(D.series),
        "lgbm": {"config_index": k, "params": configs[k], "features": features, "use_ext": bool(use_ext),
                 "ext_p_better_val": ext_cmp["without_ext"]["p_better"]},
        "ensemble_weights": dict(zip(MODELS, w_best)),
        "conformal": {"level_each_side": conformal.LEVEL, "Q": {str(h): list(v) for h, v in Q.items()}},
        "bootstrap": {"block": config.BOOT_BLOCK, "block_h1": 3, "B": config.BOOT_B, "seed": config.SEED,
                      "widen": "sqrt(n/(n-b))"},
        "primary": [
            "테스트 WAPE: LGBM vs SNaive, ETS (18개 시리즈 x 24개 기준 시점 x 6개 시점 전체)",
            "P_all: 재표집 중 LGBM의 WAPE가 SNaive와 ETS 둘 다보다 낮은 비율 (블록 6, 보정)",
            "P10~P90 적중률: 검증 구간 보정(CQR static)",
        ],
        "decision_rule": "P >= 0.9이면 채택. 판단은 검증 구간에서 하고, 테스트 확률은 결과와 관계없이 그대로 보고한다.",
        "configs": [{"index": i, **c, "val_wape": cfg_wape[i]} for i, c in enumerate(configs)],
    }
    summary = {"ets_fallbacks": ets_fails, "metrics": model_tables(fcs), "ext_compare": ext_cmp,
               "ensemble_grid_best": ens[w_best], "bootstrap_val": val_boot,
               "intervals_raw": interval_stats(iv["y"], iv["lo"], iv["hi"]),
               "intervals_reordered_share": float(iv["reordered"].mean())}
    pd.concat(fcs.values(), ignore_index=True).to_csv(OUT / "val_forecasts.csv.gz", index=False)
    iv[IV_COLS].to_csv(OUT / "val_intervals.csv.gz", index=False)
    _json(summary, OUT / "val_summary.json")
    if not config.FROZEN_PATH.exists():     # 동결할 때만 원자료 지문을 남긴다 (다시 돌릴 때는 덮어쓰지 않음)
        _json(manifest, config.ROOT / "data" / "manifest.json")
        _json({"sha256": content_hash(D)}, config.ROOT / "data" / "content_hash.json")
    return frozen


# ---------------------------------------------------------------- 테스트 단계
def stage_final(D):
    if not config.FROZEN_PATH.exists():
        raise SystemExit("frozen.json이 없습니다. 먼저 `python run_all.py --stage val`로 만들고 커밋하세요.")
    F = json.loads(config.FROZEN_PATH.read_text(encoding="utf-8"))
    assert F["series"] == list(D.series), "시리즈 구성이 동결 때와 다릅니다"
    ch_path = config.ROOT / "data" / "content_hash.json"
    if ch_path.exists():      # 값 기준 지문 (새로 받아도 값이 같으면 일치)
        same_data = content_hash(D) == json.loads(ch_path.read_text(encoding="utf-8"))["sha256"]
    else:                     # 파일 기준 지문 (원래 캐시 파일과만 일치)
        manifest = {"comtrade": data_comtrade.manifest(), "fred": data_fred.manifest()}
        same_data = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() == F["data"]["manifest_sha256"]
    if not same_data:
        print("[final] 경고: 원자료 값이 동결 때와 다릅니다 (Comtrade 수정 등). 숫자가 RESULTS.md와 다를 수 있습니다.")
    test = D.test_idx
    print(f"[final] 기준 시점 {D.months[test[0]]} ~ {D.months[test[-1]]} ({len(test)}개)")
    params, features = F["lgbm"]["params"], F["lgbm"]["features"]

    base, ets_fails = run_baselines(D, test)
    fcs = {**base, "LGBM": run_lgbm(D, test, params, features, "LGBM")}
    fcs["Ensemble"] = backtest.blend(fcs, F["ensemble_weights"], "Ensemble")
    fcs["Ensemble_eq"] = backtest.blend(fcs, dict.fromkeys(MODELS, 1 / 3), "Ensemble_eq")

    # ---- 확률 예측: 원래 분위수 / 검증 구간 보정(static) / 온라인 보정
    lo = run_lgbm(D, test, params, features, "q10", "quantile", config.ALPHA_LO)
    hi = run_lgbm(D, test, params, features, "q90", "quantile", config.ALPHA_HI)
    iv = intervals(fcs["LGBM"], lo, hi)
    Q = {int(h): v for h, v in F["conformal"]["Q"].items()}
    iv["L_static"], iv["U_static"] = conformal.apply(
        iv["lo"], iv["hi"], iv["scale"], iv["h"].map(lambda h: Q[h][0]), iv["h"].map(lambda h: Q[h][1]))
    iv_val = pd.read_csv(OUT / "val_intervals.csv.gz")
    cal_all = pd.concat([iv_val[["T", "h", "y", "lo", "hi", "scale"]], iv[["T", "h", "y", "lo", "hi", "scale"]]])
    iv["L_online"], iv["U_online"] = np.nan, np.nan
    for T in test:
        Qt = conformal.online_thresholds(cal_all, T)       # 목표 월 <= T 인 점수만
        m = iv["T"] == T
        L, U = conformal.apply(iv.loc[m, "lo"], iv.loc[m, "hi"], iv.loc[m, "scale"],
                               iv.loc[m, "h"].map(lambda h: Qt[h][0]), iv.loc[m, "h"].map(lambda h: Qt[h][1]))
        iv.loc[m, "L_online"], iv.loc[m, "U_online"] = L, U

    # ---- 진단 (선택에는 쓰지 않음): 설정 30개 전부, 외부 지표 반대 버전
    cfgs = [{k: c[k] for k in lgbm.SPACE} for c in F["configs"]]
    diag = run_configs(D, test, cfgs, lgbm.BASE_FEATURES)
    alt_features = lgbm.BASE_FEATURES if F["lgbm"]["use_ext"] else lgbm.BASE_FEATURES + lgbm.EXT_FEATURES
    alt = run_lgbm(D, test, params, alt_features, "LGBM_alt")

    # ---- 결과 묶기
    R = {"frozen_created": F["created"], "same_data_as_freeze": same_data, "ets_fallbacks_test": ets_fails,
         "intervals_reordered_share_test": float(iv["reordered"].mean()), "metrics": model_tables(fcs)}
    # 예측 대상 월이 모든 h에서 겹치는 구간만으로 본 시점별 WAPE (시점 효과와 기간 효과 분리)
    c0, c1 = config.CUTOFF + config.H, config.TEST_ORIGINS[-1] + 1
    R["common_targets"] = [str(c0), str(c1)]
    R["by_h_common"] = {m: metrics.summarize(fc[(fc.target >= c0) & (fc.target <= c1)], "h")["WAPE"].to_dict()
                        for m, fc in fcs.items()}
    R["by_series"] = {m: metrics.summarize(fc, "series")[["WAPE", "Bias", "MASE"]].to_dict(orient="index")
                      for m, fc in fcs.items() if m in ("SNaive", "ETS", "LGBM")}
    vol = fcs["LGBM"].groupby("series")["y"].sum()
    R["series_share_test"] = (vol / vol.sum()).to_dict()
    no_oth = {m: score(fc[~fc.series.str.endswith("-OTH")]) for m, fc in fcs.items()}
    R["wape_without_others"] = no_oth

    # 부트스트랩: 블록 6 + 보정 (주), 그 밖은 민감도
    A = {m: backtest.per_origin(f)[0] for m, f in fcs.items()}
    bases = {"SNaive": A["SNaive"], "ETS": A["ETS"]}
    R["bootstrap"] = {
        "LGBM": bootstrap.compare(A["LGBM"], bases),
        "Ensemble": bootstrap.compare(A["Ensemble"], bases),
        "LGBM_b3": bootstrap.compare(A["LGBM"], bases, b=3),
        "LGBM_b6_raw": bootstrap.compare(A["LGBM"], bases, correct=False),
    }
    h1 = {m: backtest.per_origin(f[f.h == 1])[0] for m, f in fcs.items()}
    R["bootstrap"]["LGBM_h1_b3"] = bootstrap.compare(h1["LGBM"], {"SNaive": h1["SNaive"], "ETS": h1["ETS"]},
                                                     b=F["bootstrap"]["block_h1"])

    # 구간 평가
    ivr = {}
    for kind, (Lc, Uc) in {"raw": ("lo", "hi"), "static": ("L_static", "U_static"),
                           "online": ("L_online", "U_online")}.items():
        hit = ((iv["y"] >= iv[Lc]) & (iv["y"] <= iv[Uc])).astype(float)
        num = hit.groupby(iv["T"]).sum().to_numpy()
        den = hit.groupby(iv["T"]).size().to_numpy().astype(float)
        cov, cl, ch = bootstrap.ratio_interval(num, den)
        ivr[kind] = {"all": {**interval_stats(iv["y"], iv[Lc], iv[Uc]), "coverage_ci90": [cl, ch]},
                     "by_h": {int(h): interval_stats(g["y"], g[Lc], g[Uc]) for h, g in iv.groupby("h")},
                     "by_product": {p: interval_stats(g["y"], g[Lc], g[Uc]) for p, g in iv.groupby("product")},
                     "series_coverage_min_max": [
                         float(min(((g["y"] >= g[Lc]) & (g["y"] <= g[Uc])).mean() for _, g in iv.groupby("series"))),
                         float(max(((g["y"] >= g[Lc]) & (g["y"] <= g[Uc])).mean() for _, g in iv.groupby("series")))]}
    R["intervals"] = ivr
    R["conformal_Q"] = F["conformal"]["Q"]

    # 선택 편향: 설정 30개의 검증 WAPE vs 테스트 WAPE
    vw = np.array([c["val_wape"] for c in F["configs"]])
    tw = np.array([score(f) for f in diag])
    k = F["lgbm"]["config_index"]
    shift = np.log(tw) - np.log(vw)                         # 설정마다 검증 -> 테스트 변화 (로그 비율)
    opt = shift[k] - np.median(shift)                       # 고른 설정이 '보통 설정'보다 더 나빠진 정도
    R["selection_bias"] = {
        "winner": k, "val_winner": float(vw[k]), "test_winner": float(tw[k]),
        "val_median": float(np.median(vw)), "test_median": float(np.median(tw)),
        "val_best": float(vw.min()), "test_best": float(tw.min()), "test_best_index": int(tw.argmin()),
        "winner_test_rank": int((tw < tw[k]).sum()) + 1, "n_configs": int(len(tw)),
        "optimism_log": float(opt), "optimism_pct": float(100 * (np.exp(opt) - 1)),
        "median_shift_pct": float(100 * (np.exp(np.median(shift)) - 1)),
        "spearman_val_test": float(spearmanr(vw, tw)[0]),
        "oracle_gap_pp": float(100 * (tw[k] - tw.min())),
        "val_wape": vw.tolist(), "test_wape": tw.tolist(),
    }
    vs = json.loads((OUT / "val_summary.json").read_text(encoding="utf-8"))
    R["val_vs_test"] = {m: {"val": vs["metrics"][m]["all"]["WAPE"], "test": R["metrics"][m]["all"]["WAPE"]}
                        for m in fcs if m in vs["metrics"]}
    R["ext"] = {"use_ext": F["lgbm"]["use_ext"], "p_better_val": F["lgbm"]["ext_p_better_val"],
                "test_wape_chosen": score(fcs["LGBM"]), "test_wape_alternative": score(alt)}

    # 피처 중요도 (마지막 테스트 기준 시점에서 학습한 모델, gain)
    _, model = lgbm.fit_predict(D.rows, test[-1], params, features)
    gain = pd.Series(model.booster_.feature_importance("gain"), index=features)
    R["feature_gain_share"] = (gain / gain.sum()).sort_values(ascending=False).to_dict()

    pd.concat(fcs.values(), ignore_index=True).to_csv(OUT / "test_forecasts.csv.gz", index=False)
    iv.to_csv(OUT / "test_intervals.csv.gz", index=False)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(diag, ignore_index=True).to_csv(config.CACHE_DIR / "diag_configs_test.csv.gz", index=False)
    _json(R, OUT / "results.json")
    return R
