"""질의응답 에이전트의 도구 두 개. 저장된 실행 결과 파일만 읽는다 (모델을 다시 돌리지 않는다).

- get_forecast(series, origin='latest', h=None): 시리즈 하나의 1~6개월 앞 예측
- get_backtest(series=None, h=None): 테스트 구간 백테스트 성능
숫자는 사람이 읽는 단위로 반올림해서 돌려준다: 물량 톤(소수 1자리), 비율 %(소수 1자리), 확률 소수 3자리.
에이전트가 계산하지 않고 인용만 하도록, 자주 묻는 값(감소율, 오차율, 전년 대비, 이긴 기준 시점 수)은 미리 계산해 넣는다.
먼저 python run_all.py로 outputs/의 결과 파일을 만들어 두어야 한다.
"""
import json
from functools import lru_cache

import pandas as pd

from . import config
from .metrics import wape

OUT = config.OUT_DIR
TOP = ["TOTAL", "ABS", "PC"]
MODEL_KO = {"SNaive": "계절 나이브", "ETS": "ETS", "LGBM": "LightGBM", "Ensemble": "앙상블"}
METHOD_KO = {"Base": "조정 전 (합계를 ETS로 직접 예측)", "BU": "BU (하위 LightGBM 예측의 합)", "MinT_shrink": "MinT"}
LEVEL_KO = {"total": "전체", "product": "제품 합계", "bottom": "하위 18개"}
COUNTRY_KO = {"CHN": "중국", "HKG": "홍콩", "TUR": "튀르키예", "VNM": "베트남", "USA": "미국", "MEX": "멕시코",
              "THA": "태국", "IND": "인도", "HUN": "헝가리", "JPN": "일본", "OTH": "상위 8개국 밖의 나머지 전부"}


class ToolError(ValueError):
    """잘못된 인자. 메시지를 모델에 그대로 돌려줘 스스로 고치게 한다."""


def _pct(x):
    v = round(100 * float(x), 1)
    return 0.0 if v == 0 else v              # -0.0 대신 0.0


def _t(x):
    return round(float(x), 1)


def _p(x):
    return round(float(x), 3)


@lru_cache(maxsize=None)
def _json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _csv(name):
    return pd.read_csv(OUT / name)


def _frozen_mint():
    return json.loads(config.FROZEN_MINT_PATH.read_text(encoding="utf-8"))


def series_names():
    return TOP + _csv("series_info.csv")["series"].tolist()


def _series(s):
    s = str(s or "").strip().upper()
    if s not in series_names():
        raise ToolError(f"알 수 없는 시리즈입니다: '{s}'. 가능한 값: {', '.join(series_names())}")
    return s


def _h(h):
    """h 인자 -> 정수 목록. 없으면 1~6 모두."""
    if h in (None, ""):
        return list(range(1, config.H + 1))
    try:
        h = int(h)
    except (TypeError, ValueError):
        raise ToolError("h는 1~6 사이 정수입니다") from None
    if not 1 <= h <= config.H:
        raise ToolError("h는 1~6 사이 정수입니다")
    return [h]


def _notes(series, method):
    if method != "BU":
        return "forecast_t는 MinT로 합계가 맞도록 조정한 점예측이다."
    if series in TOP:
        return "forecast_t는 이 합계에 속한 하위 시리즈의 LightGBM 예측을 더한 값이다 (BU). 합계 노드에는 구간이 없다."
    return ("forecast_t는 LightGBM 점예측(P50)이고, 합계 노드는 이 값들을 더해 만든다 (BU). "
            "p10_t ~ p90_t는 LightGBM 분위수를 검증 구간 오차로 보정(CQR)한 80% 구간이다.")


# ---------------------------------------------------------------- get_forecast
def get_forecast(series=None, origin="latest", h=None):
    s, hs = _series(series), _h(h)
    origin = str(origin or "latest").strip().lower()
    method = _frozen_mint()["adopted"]
    rows = []
    if origin == "latest":
        f = _csv("forecast_latest.csv.gz")
        f = f[(f.series == s) & f.h.isin(hs)]
        for r in f.itertuples():
            row = {"h": int(r.h), "target": r.target, "forecast_t": _t(r.point),
                   "same_month_last_year_actual_t": _t(r.same_month_last_year),
                   "change_vs_last_year_pct": _pct(r.point / r.same_month_last_year - 1)
                   if r.same_month_last_year > 0 else None}
            if s not in TOP:
                row.update(p10_t=_t(r.p10), p90_t=_t(r.p90))
            rows.append(row)
        return {"series": s, "origin": str(f.origin.iloc[0]), "unit": "톤",
                "kind": "미래 예측 (이 달들의 실제값은 아직 Comtrade에 없다)", "rows": rows, "notes": _notes(s, method)}

    origins = [str(p) for p in config.TEST_ORIGINS]
    if origin not in origins:
        raise ToolError(f"origin은 'latest'이거나 테스트 기준 시점 {origins[0]} ~ {origins[-1]} 중 하나('YYYY-MM')입니다")
    m = _csv("mint_forecasts.csv.gz")
    g = m[(m.base == "LGBM") & (m.model == method) & (m.series == s) & (m.origin == origin) & m.h.isin(hs)]
    iv = _csv("test_intervals.csv.gz")
    for r in g.sort_values("h").itertuples():
        row = {"h": int(r.h), "target": r.target, "forecast_t": _t(r.yhat), "actual_t": _t(r.y),
               "abs_error_pct_of_actual": _pct(abs(r.y - r.yhat) / r.y)}
        if s not in TOP:
            k = iv[(iv.series == s) & (iv.origin == origin) & (iv.h == r.h)].iloc[0]
            row.update(p10_t=_t(k.L_static), p90_t=_t(k.U_static),
                       actual_inside_p10_p90=bool(k.L_static <= k.y <= k.U_static))
        rows.append(row)
    return {"series": s, "origin": origin, "unit": "톤",
            "kind": "테스트 구간 백테스트 (그 기준 시점까지의 자료만으로 낸 예측과 실제값)", "rows": rows,
            "notes": _notes(s, method)}


# ---------------------------------------------------------------- get_backtest
def _reconciliation():
    M, FM = _json("mint_results.json"), _frozen_mint()
    mt, b = M["metrics"]["LGBM"], M["bootstrap"]["LGBM"]["MinT_vs_BU"]
    return {
        "adopted_method": FM["adopted"],
        "decision": f"검증 구간 2겹 교차 적합에서 P(MinT가 전체·제품·하위 모두 BU보다 나음) = "
                    f"{FM['gate']['p_better_all']:.3f}, 기준 0.9 -> {FM['adopted']} 채택",
        "test_wape_pct": {LEVEL_KO[k]: {METHOD_KO[m]: _pct(mt[m]["by_level"][k]["WAPE"]) for m in METHOD_KO}
                          for k in LEVEL_KO},
        "test_p_mint_better_than_bu": {LEVEL_KO[k]: _p(b[k]["p_better"]) for k in LEVEL_KO},
        "test_mint_bias_pct": _pct(mt["MinT_shrink"]["by_level"]["bottom"]["Bias"]),
    }


def _overall(R):
    m, b, iv = R["metrics"], R["bootstrap"]["LGBM"], R["intervals"]
    fc = _csv("test_forecasts.csv.gz")
    A = fc.assign(ae=(fc.y - fc.yhat).abs()).groupby(["model", "origin"])["ae"].sum().unstack(0)
    return {
        "window": f"테스트 구간: 기준 시점 {config.TEST_ORIGINS[0]} ~ {config.TEST_ORIGINS[-1]} ({len(A)}개), "
                  f"1~6개월 앞, 하위 시리즈 18개",
        "models": {MODEL_KO[k]: {"wape_pct": _pct(m[k]["all"]["WAPE"]), "bias_pct": _pct(m[k]["all"]["Bias"]),
                                 "mase": round(m[k]["all"]["MASE"], 3)} for k in MODEL_KO},
        "lightgbm_vs_baselines": {
            "p_better_than_ets": _p(b["ETS"]["p_better"]),
            "p_better_than_seasonal_naive": _p(b["SNaive"]["p_better"]),
            "p_better_than_both": _p(b["p_better_all"]),
            "wape_reduction_vs_ets_pct": _pct(b["ETS"]["rel_reduction"]),
            "wape_reduction_vs_ets_ci90_pct": [_pct(b["ETS"]["rel_reduction_q05"]), _pct(b["ETS"]["rel_reduction_q95"])],
            "wape_reduction_vs_seasonal_naive_pct": _pct(b["SNaive"]["rel_reduction"]),
            "wape_reduction_vs_seasonal_naive_ci90_pct": [_pct(b["SNaive"]["rel_reduction_q05"]),
                                                          _pct(b["SNaive"]["rel_reduction_q95"])],
            "origins_lightgbm_better_than_ets": int((A["ETS"] > A["LGBM"]).sum()),
            "origins_lightgbm_better_than_seasonal_naive": int((A["SNaive"] > A["LGBM"]).sum()),
            "n_origins": int(len(A))},
        "h1_only": {"lightgbm_wape_pct": _pct(m["LGBM"]["by_h"]["1"]["WAPE"]),
                    "ets_wape_pct": _pct(m["ETS"]["by_h"]["1"]["WAPE"]),
                    "p_lightgbm_better_than_ets": _p(R["bootstrap"]["LGBM_h1_b3"]["ETS"]["p_better"])},
        "intervals_p10_p90": {"target_coverage_pct": 80,
                              "coverage_before_calibration_pct": _pct(iv["raw"]["all"]["coverage"]),
                              "coverage_cqr_pct": _pct(iv["static"]["all"]["coverage"]),
                              "coverage_cqr_ci90_pct": [_pct(x) for x in iv["static"]["all"]["coverage_ci90"]]},
        "reconciliation": _reconciliation(),
        "notes": "P는 테스트 기준 시점을 길이 6 블록으로 다시 뽑는 부트스트랩 10,000회에서 WAPE가 더 낮게 나온 비율이다. "
                 "미래에 더 나을 확률이 아니며, 서로 겹치지 않는 6개월 블록은 4개뿐이다.",
    }


def _by_h(R, h):
    m, iv = R["metrics"], R["intervals"]
    out = {"h": h, "models_wape_pct": {MODEL_KO[k]: _pct(m[k]["by_h"][str(h)]["WAPE"]) for k in MODEL_KO},
           "coverage_cqr_pct": _pct(iv["static"]["by_h"][str(h)]["coverage"]),
           "coverage_before_calibration_pct": _pct(iv["raw"]["by_h"][str(h)]["coverage"])}
    if h == 1:
        out["p_lightgbm_better_than_ets"] = _p(R["bootstrap"]["LGBM_h1_b3"]["ETS"]["p_better"])
    return out


def _node(R, s, h):
    if s in TOP:                        # 합계 노드: 계층 조정 결과
        mt = _json("mint_results.json")["metrics"]["LGBM"]
        if h is None:
            raw = {m: mt[m]["by_node"][s]["WAPE"] for m in METHOD_KO}
        else:
            mf = _csv("mint_forecasts.csv.gz")
            g = mf[(mf.base == "LGBM") & (mf.series == s) & (mf.h == h)]
            raw = {m: wape(g[g.model == m]["y"], g[g.model == m]["yhat"]) for m in METHOD_KO}
        return {"series": s, "h": h if h else "1~6", "methods_wape_pct": {METHOD_KO[m]: _pct(v) for m, v in raw.items()},
                "best_method": METHOD_KO[min(raw, key=raw.get)], "adopted_method": _frozen_mint()["adopted"]}
    country = s.split("-", 1)[1]
    if h is None:
        raw = {k: R["by_series"][k][s] for k in ("SNaive", "ETS", "LGBM")}
        models = {MODEL_KO[k]: {"wape_pct": _pct(v["WAPE"]), "bias_pct": _pct(v["Bias"]), "mase": round(v["MASE"], 3)}
                  for k, v in raw.items()}
        best = min(raw, key=lambda k: raw[k]["WAPE"])
    else:
        fc = _csv("test_forecasts.csv.gz")
        g = fc[(fc.series == s) & (fc.h == h)]
        raw = {k: wape(g[g.model == k]["y"], g[g.model == k]["yhat"]) for k in ("SNaive", "ETS", "LGBM")}
        models = {MODEL_KO[k]: {"wape_pct": _pct(v)} for k, v in raw.items()}
        best = min(raw, key=raw.get)
    iv = _csv("test_intervals.csv.gz")
    g = iv[(iv.series == s) & (iv.h.isin(_h(h)))]
    return {"series": s, "country": COUNTRY_KO.get(country, country), "h": h if h else "1~6",
            "test_volume_share_pct": _pct(R["series_share_test"][s]), "models": models, "best_model": MODEL_KO[best],
            "coverage_cqr_pct": _pct(((g.y >= g.L_static) & (g.y <= g.U_static)).mean())}


def get_backtest(series=None, h=None):
    R = _json("results.json")
    if series in (None, ""):
        return _overall(R) if h in (None, "") else _by_h(R, _h(h)[0])
    return _node(R, _series(series), None if h in (None, "") else _h(h)[0])


# ---------------------------------------------------------------- 모델에 보내는 도구 설명
FUNCS = {"get_forecast": get_forecast, "get_backtest": get_backtest}
ARGS = {"get_forecast": {"series", "origin", "h"}, "get_backtest": {"series", "h"}}


def call(name, args):
    if name not in FUNCS:
        raise ToolError(f"없는 도구입니다: {name}. 쓸 수 있는 도구: {', '.join(FUNCS)}")
    extra = set(args) - ARGS[name]
    if extra:
        raise ToolError(f"{name}에 없는 인자입니다: {', '.join(sorted(extra))}")
    return FUNCS[name](**args)


def specs():
    """OpenAI 호환 API의 tools 형식."""
    info = _csv("series_info.csv").drop_duplicates("country")
    countries = ", ".join(f"{c} {COUNTRY_KO.get(c, n)}" for c, n in zip(info["country"], info["name"]))
    names = ", ".join(series_names()[len(TOP):])
    series_desc = (f"TOTAL(ABS와 PC 전체 합계), ABS, PC(제품 합계), 또는 '제품-국가' 하위 시리즈: {names}. "
                   f"국가 코드: {countries}")
    t0, t1 = config.TEST_ORIGINS[0], config.TEST_ORIGINS[-1]
    return [
        {"type": "function", "function": {
            "name": "get_forecast",
            "description": f"시리즈 하나의 1~6개월 앞 예측(톤). origin='latest'면 마지막 자료 월({config.END}) 기준 미래 "
                           f"예측, 'YYYY-MM'({t0} ~ {t1})이면 그 기준 시점의 테스트 구간 예측과 실제값.",
            "parameters": {"type": "object", "properties": {
                "series": {"type": "string", "description": series_desc},
                "origin": {"type": "string", "description": "'latest' 또는 'YYYY-MM'. 기본값 'latest'"},
                "h": {"type": "integer", "description": "몇 개월 앞인지 (1~6). 없으면 1~6 모두"}},
                "required": ["series"]}}},
        {"type": "function", "function": {
            "name": "get_backtest",
            "description": f"테스트 구간(기준 시점 {t0} ~ {t1}) 백테스트 성능: 모델별 WAPE·편향·MASE, 부트스트랩 P, "
                           "P10~P90 적중률, 계층 조정 결과. 인자가 없으면 전체 요약.",
            "parameters": {"type": "object", "properties": {
                "series": {"type": "string", "description": "시리즈 하나로 좁힐 때. " + series_desc},
                "h": {"type": "integer", "description": "몇 개월 앞(1~6)으로 좁힐 때"}},
                "required": []}}},
    ]
