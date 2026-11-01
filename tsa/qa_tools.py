"""질의응답 에이전트의 도구 두 개. 저장된 실행 결과 파일만 읽는다 (모델을 다시 돌리지 않는다).

- get_forecast(series, origin='latest', h=None): 시리즈 하나의 1~6개월 앞 예측
- get_backtest(series=None, h=None): 테스트 구간 백테스트 성능
숫자는 사람이 읽는 단위로 반올림해서 돌려준다: 물량 톤(소수 1자리), 비율 %(소수 1자리), 확률 소수 3자리.
에이전트가 계산하지 않고 인용만 하도록, 자주 묻는 값(감소율, 오차율, 전년 대비, 여러 달 합계)은 미리 계산해 넣는다.
작은 모델이 숫자를 다른 뜻으로 붙이지 않게, 결과마다 그 숫자들을 설명하는 사실 문장(summary_ko)을 함께 준다.
먼저 python run_all.py로 outputs/의 결과 파일을 만들어 두어야 한다.
"""
import json
import re
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
FIRST, LAST = str(config.END + 1), str(config.END + config.H)       # 미래 예측이 있는 달
HINT_H = "h는 1~6 사이 정수이거나 [1, 2, 3] 같은 목록입니다. 1~6이 모두 필요하면 h를 비우세요."


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
    """h 인자 -> 정수 목록. 없으면 1~6 모두. 3, '3', [1, 2, 3], '1~3', '1,2,3'을 받는다."""
    if h is None or h == "" or h == []:
        return list(range(1, config.H + 1))
    if isinstance(h, str):
        m = re.fullmatch(r"\s*(\d+)\s*[~-]\s*(\d+)\s*", h)
        items = list(range(int(m.group(1)), int(m.group(2)) + 1)) if m else re.split(r"[,\s]+", h.strip())
    elif isinstance(h, (list, tuple)):
        items = list(h)
    else:
        items = [h]
    try:
        hs = sorted({int(x) for x in items})
    except (TypeError, ValueError):
        raise ToolError(HINT_H) from None
    if not hs or not all(1 <= x <= config.H for x in hs):
        raise ToolError(HINT_H)
    return hs


def _best(raw, names):
    """WAPE가 가장 낮은 이름 (반올림 전 값으로 고른다)."""
    return names[min(raw, key=raw.get)]


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
        f = f[(f.series == s) & f.h.isin(hs)].sort_values("h")
        for r in f.itertuples():
            row = {"h": int(r.h), "target": r.target, "forecast_t": _t(r.point),
                   "same_month_last_year_actual_t": _t(r.same_month_last_year),
                   "change_vs_last_year_pct": _pct(r.point / r.same_month_last_year - 1)
                   if r.same_month_last_year > 0 else None}
            if s not in TOP:
                row.update(p10_t=_t(r.p10), p90_t=_t(r.p90))
            rows.append(row)
        out = {"series": s, "origin": str(f.origin.iloc[0]), "unit": "톤",
               "kind": "미래 예측 (이 달들의 실제값은 아직 Comtrade에 없다)",
               "available_targets": f"{FIRST} ~ {LAST}", "rows": rows}
        summary = (f"{s}의 {out['origin']} 기준 미래 예측: "
                   + ", ".join(f"{r['target']} {r['forecast_t']:,.1f}톤" for r in rows)
                   + ". 각 값은 그 달 한 달의 예측이다.")
        if len(rows) > 1:
            out["rows_total_t"] = _t(f["point"].sum())
            summary += f" 이 {len(rows)}개월의 합계는 {out['rows_total_t']:,.1f}톤이다."
        summary += (" 작년 같은 달 대비: " + ", ".join(f"{r['target']} {r['change_vs_last_year_pct']:+.1f}%" for r in rows
                                                if r["change_vs_last_year_pct"] is not None) + ".")
        if s not in TOP:
            summary += (" 80% 구간(P10~P90): "
                        + ", ".join(f"{r['target']} {r['p10_t']:,.1f} ~ {r['p90_t']:,.1f}톤" for r in rows) + ".")
        summary += f" 미래 예측이 있는 달은 {FIRST} ~ {LAST}뿐이다."
        out.update(summary_ko=summary, notes=_notes(s, method))
        return out

    origins = [str(p) for p in config.TEST_ORIGINS]
    if origin not in origins:
        raise ToolError(f"origin은 'latest'(미래 예측 {FIRST} ~ {LAST})이거나 테스트 기준 시점 "
                        f"{origins[0]} ~ {origins[-1]} 중 하나('YYYY-MM')입니다")
    m = _csv("mint_forecasts.csv.gz")
    g = m[(m.base == "LGBM") & (m.model == method) & (m.series == s) & (m.origin == origin) & m.h.isin(hs)]
    iv = _csv("test_intervals.csv.gz")
    parts = []
    for r in g.sort_values("h").itertuples():
        row = {"h": int(r.h), "target": r.target, "forecast_t": _t(r.yhat), "actual_t": _t(r.y),
               "abs_error_pct_of_actual": _pct(abs(r.y - r.yhat) / r.y)}
        text = (f"{r.target}({int(r.h)}개월 앞) 예측 {row['forecast_t']:,.1f}톤, 실제 {row['actual_t']:,.1f}톤, "
                f"오차 {row['abs_error_pct_of_actual']}%")
        if s not in TOP:
            k = iv[(iv.series == s) & (iv.origin == origin) & (iv.h == r.h)].iloc[0]
            inside = bool(k.L_static <= k.y <= k.U_static)
            row.update(p10_t=_t(k.L_static), p90_t=_t(k.U_static), actual_inside_p10_p90=inside)
            text += f", 실제값은 80% 구간 {'안' if inside else '밖'}"
        rows.append(row)
        parts.append(text)
    return {"series": s, "origin": origin, "unit": "톤",
            "kind": "테스트 구간 백테스트 (그 기준 시점까지의 자료만으로 낸 예측과 실제값)", "rows": rows,
            "summary_ko": f"{origin} 기준 {s}: " + "; ".join(parts) + ".", "notes": _notes(s, method)}


# ---------------------------------------------------------------- get_backtest
def _reconciliation():
    M, FM = _json("mint_results.json"), _frozen_mint()
    mt, b = M["metrics"]["LGBM"], M["bootstrap"]["LGBM"]["MinT_vs_BU"]
    w = {k: {m: _pct(mt[m]["by_level"][k]["WAPE"]) for m in METHOD_KO} for k in LEVEL_KO}
    decision = (f"검증 구간 2겹 교차 적합에서 P(MinT가 전체·제품·하위 모두 BU보다 나음) = "
                f"{FM['gate']['p_better_all']:.3f}, 기준 0.9 -> {FM['adopted']} 채택")
    return {
        "adopted_method": FM["adopted"], "decision": decision,
        "test_wape_pct": {LEVEL_KO[k]: {METHOD_KO[m]: v for m, v in w[k].items()} for k in LEVEL_KO},
        "test_p_mint_better_than_bu": {LEVEL_KO[k]: _p(b[k]["p_better"]) for k in LEVEL_KO},
        "test_mint_bias_pct": _pct(mt["MinT_shrink"]["by_level"]["bottom"]["Bias"]),
        "summary_ko": (f"합계 맞추기는 {FM['adopted']}를 채택했다 ({decision}). 테스트 WAPE: 전체는 조정 전 "
                       f"{w['total']['Base']}%, BU {w['total']['BU']}%, MinT {w['total']['MinT_shrink']}%이고, "
                       f"하위 18개는 BU {w['bottom']['BU']}%, MinT {w['bottom']['MinT_shrink']}%이다."),
    }


def _overall(R):
    m, b, iv = R["metrics"], R["bootstrap"]["LGBM"], R["intervals"]
    fc = _csv("test_forecasts.csv.gz")
    A = fc.assign(ae=(fc.y - fc.yhat).abs()).groupby(["model", "origin"])["ae"].sum().unstack(0)
    wp = {k: _pct(m[k]["all"]["WAPE"]) for k in MODEL_KO}
    r_e, r_s = _pct(b["ETS"]["rel_reduction"]), _pct(b["SNaive"]["rel_reduction"])
    ci_e = [_pct(b["ETS"]["rel_reduction_q05"]), _pct(b["ETS"]["rel_reduction_q95"])]
    won_e, won_s, n = int((A["ETS"] > A["LGBM"]).sum()), int((A["SNaive"] > A["LGBM"]).sum()), int(len(A))
    h1 = {k: _pct(m[k]["by_h"]["1"]["WAPE"]) for k in ("LGBM", "ETS")}
    p_h1 = _p(R["bootstrap"]["LGBM_h1_b3"]["ETS"]["p_better"])
    cov = {"raw": _pct(iv["raw"]["all"]["coverage"]), "cqr": _pct(iv["static"]["all"]["coverage"]),
           "ci": [_pct(x) for x in iv["static"]["all"]["coverage_ci90"]]}
    rec = _reconciliation()
    h1_text = ("같다" if h1["LGBM"] == h1["ETS"] else
               ("LightGBM이 더 낮다" if h1["LGBM"] < h1["ETS"] else "ETS가 더 낮다"))
    return {
        "window": f"테스트 구간: 기준 시점 {config.TEST_ORIGINS[0]} ~ {config.TEST_ORIGINS[-1]} ({n}개), "
                  f"1~6개월 앞, 하위 시리즈 18개",
        "models": {MODEL_KO[k]: {"wape_pct": wp[k], "bias_pct": _pct(m[k]["all"]["Bias"]),
                                 "mase": round(m[k]["all"]["MASE"], 3)} for k in MODEL_KO},
        "lightgbm_vs_baselines": {
            "p_better_than_ets": _p(b["ETS"]["p_better"]),
            "p_better_than_seasonal_naive": _p(b["SNaive"]["p_better"]),
            "p_better_than_both": _p(b["p_better_all"]),
            "wape_reduction_vs_ets_pct": r_e, "wape_reduction_vs_ets_ci90_pct": ci_e,
            "wape_reduction_vs_seasonal_naive_pct": r_s,
            "wape_reduction_vs_seasonal_naive_ci90_pct": [_pct(b["SNaive"]["rel_reduction_q05"]),
                                                          _pct(b["SNaive"]["rel_reduction_q95"])],
            "origins_lightgbm_better_than_ets": won_e, "origins_lightgbm_better_than_seasonal_naive": won_s,
            "n_origins": n},
        "h1_only": {"lightgbm_wape_pct": h1["LGBM"], "ets_wape_pct": h1["ETS"], "p_lightgbm_better_than_ets": p_h1},
        "intervals_p10_p90": {"target_coverage_pct": 80, "coverage_before_calibration_pct": cov["raw"],
                              "coverage_cqr_pct": cov["cqr"], "coverage_cqr_ci90_pct": cov["ci"]},
        "reconciliation": rec,
        "summary_ko": [
            f"테스트 WAPE(낮을수록 좋음): LightGBM {wp['LGBM']}%, ETS {wp['ETS']}%, 계절 나이브 {wp['SNaive']}%, "
            f"앙상블 {wp['Ensemble']}%.",
            f"LightGBM의 WAPE는 ETS 대비 상대적으로 {abs(r_e)}% {'낮고' if r_e > 0 else '높고'} "
            f"(90% 구간 {ci_e[0]}% ~ {ci_e[1]}%), 계절 나이브 대비 상대적으로 {abs(r_s)}% {'낮다' if r_s > 0 else '높다'}.",
            f"기준 시점 {n}개 중 LightGBM이 ETS보다 나은 기준 시점은 {won_e}개, 계절 나이브보다 나은 기준 시점은 {won_s}개다.",
            f"블록 부트스트랩에서 LightGBM의 WAPE가 ETS와 계절 나이브 둘 다보다 낮게 나온 비율은 {_p(b['p_better_all'])}이다 "
            "(미래에 나을 확률이 아니다).",
            f"1개월 앞만 보면 WAPE가 LightGBM {h1['LGBM']}%, ETS {h1['ETS']}%로 {h1_text}. "
            f"P(LightGBM이 ETS보다 나음) = {p_h1}.",
            f"P10~P90 구간 적중률(목표 80%): 보정 전 {cov['raw']}%, CQR 보정 후 {cov['cqr']}% "
            f"(90% 구간 {cov['ci'][0]}% ~ {cov['ci'][1]}%).",
            rec["summary_ko"],
        ],
        "notes": "P는 테스트 기준 시점을 길이 6 블록으로 다시 뽑는 부트스트랩 10,000회에서 WAPE가 더 낮게 나온 비율이다. "
                 "미래에 더 나을 확률이 아니며, 서로 겹치지 않는 6개월 블록은 4개뿐이다.",
    }


def _by_h(R, h):
    m, iv = R["metrics"], R["intervals"]
    raw = {k: m[k]["by_h"][str(h)]["WAPE"] for k in MODEL_KO}
    wp = {k: _pct(v) for k, v in raw.items()}
    cov = _pct(iv["static"]["by_h"][str(h)]["coverage"])
    out = {"h": h, "models_wape_pct": {MODEL_KO[k]: v for k, v in wp.items()},
           "coverage_cqr_pct": cov, "coverage_before_calibration_pct": _pct(iv["raw"]["by_h"][str(h)]["coverage"])}
    summary = (f"{h}개월 앞 테스트 WAPE: " + ", ".join(f"{MODEL_KO[k]} {v}%" for k, v in wp.items())
               + f". 가장 낮은 모델: {_best(raw, MODEL_KO)}. P10~P90 적중률(CQR 보정 후) {cov}%.")
    if h == 1:
        out["p_lightgbm_better_than_ets"] = _p(R["bootstrap"]["LGBM_h1_b3"]["ETS"]["p_better"])
        summary += f" P(LightGBM이 ETS보다 나음) = {out['p_lightgbm_better_than_ets']}."
    out["summary_ko"] = summary
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
        wp = {METHOD_KO[m]: _pct(v) for m, v in raw.items()}
        adopted = _frozen_mint()["adopted"]
        when = f"{h}개월 앞" if h else "1~6개월 앞 전체"
        return {"series": s, "h": h if h else "1~6", "methods_wape_pct": wp,
                "best_method": _best(raw, METHOD_KO), "adopted_method": adopted,
                "summary_ko": (f"{s} 합계의 테스트 WAPE({when}): " + ", ".join(f"{k} {v}%" for k, v in wp.items())
                               + f". 가장 낮은 방법: {_best(raw, METHOD_KO)}. 채택한 방법: {adopted}."),
                "note": "합계 노드에는 P10~P90 구간이 없다. 구간 적중률은 인자 없는 get_backtest()의 intervals_p10_p90에 있다."}
    country = s.split("-", 1)[1]
    if h is None:
        raw_all = {k: R["by_series"][k][s] for k in ("SNaive", "ETS", "LGBM")}
        raw = {k: v["WAPE"] for k, v in raw_all.items()}
        models = {MODEL_KO[k]: {"wape_pct": _pct(v["WAPE"]), "bias_pct": _pct(v["Bias"]), "mase": round(v["MASE"], 3)}
                  for k, v in raw_all.items()}
    else:
        fc = _csv("test_forecasts.csv.gz")
        g = fc[(fc.series == s) & (fc.h == h)]
        raw = {k: wape(g[g.model == k]["y"], g[g.model == k]["yhat"]) for k in ("SNaive", "ETS", "LGBM")}
        models = {MODEL_KO[k]: {"wape_pct": _pct(v)} for k, v in raw.items()}
    iv = _csv("test_intervals.csv.gz")
    g = iv[(iv.series == s) & (iv.h.isin(_h(h)))]
    cov = _pct(((g.y >= g.L_static) & (g.y <= g.U_static)).mean())
    share = _pct(R["series_share_test"][s])
    name = COUNTRY_KO.get(country, country)
    when = f"{h}개월 앞" if h else "1~6개월 앞 전체"
    return {"series": s, "country": name, "h": h if h else "1~6", "test_volume_share_pct": share, "models": models,
            "best_model": _best(raw, MODEL_KO), "coverage_cqr_pct": cov,
            "summary_ko": (f"{s}({name}) 테스트 WAPE({when}): "
                           + ", ".join(f"{k} {v['wape_pct']}%" for k, v in models.items())
                           + f". 가장 좋은 모델: {_best(raw, MODEL_KO)}. 테스트 물량 비중 {share}%, "
                             f"P10~P90 적중률(CQR 보정 후) {cov}%.")}


def get_backtest(series=None, h=None):
    R = _json("results.json")
    hs = None if h is None or h == "" or h == [] else _h(h)
    if series in (None, ""):
        if hs is None:
            return _overall(R)
        return _by_h(R, hs[0]) if len(hs) == 1 else {"by_h": [_by_h(R, x) for x in hs]}
    s = _series(series)
    if hs is None or len(hs) == 1:
        return _node(R, s, None if hs is None else hs[0])
    return {"series": s, "by_h": [_node(R, s, x) for x in hs]}


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
    h_desc = "몇 개월 앞인지: 1~6 중 하나, 또는 [1, 2, 3] 같은 목록. 없으면 1~6 모두"
    t0, t1 = config.TEST_ORIGINS[0], config.TEST_ORIGINS[-1]
    return [
        {"type": "function", "function": {
            "name": "get_forecast",
            "description": f"예측 물량(톤)을 돌려준다. origin='latest'(기본값)면 마지막 자료 월({config.END}) 기준 "
                           f"{FIRST} ~ {LAST}의 미래 예측. origin='YYYY-MM'({t0} ~ {t1})이면 그 과거 기준 시점에서 낸 "
                           "예측과 실제값. 모델 성능 질문에는 쓰지 않는다 (get_backtest).",
            "parameters": {"type": "object", "properties": {
                "series": {"type": "string", "description": series_desc},
                "origin": {"type": "string", "description": "'latest' 또는 'YYYY-MM'. 기본값 'latest'"},
                "h": {"type": "array", "items": {"type": "integer"}, "description": h_desc}},
                "required": ["series"]}}},
        {"type": "function", "function": {
            "name": "get_backtest",
            "description": f"테스트 구간(기준 시점 {t0} ~ {t1}) 백테스트 성능. 인자 없이 부르면 전체 요약: 모델별 "
                           "WAPE·편향·MASE, 기준선 대비 개선, 부트스트랩 P, 1개월 앞 비교, P10~P90 적중률, 합계 맞추기"
                           "(BU·MinT) 결정. h만 주면 그 시점의 모델별 WAPE, series를 주면 그 시리즈의 모델별 성능.",
            "parameters": {"type": "object", "properties": {
                "series": {"type": "string", "description": "시리즈 하나로 좁힐 때. " + series_desc},
                "h": {"type": "array", "items": {"type": "integer"}, "description": "몇 개월 앞(1~6)으로 좁힐 때"}},
                "required": []}}},
    ]
