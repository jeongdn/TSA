"""도구 결과(JSON) -> 사실 문장 목록. 3차 에이전트에서 답의 문장은 모두 여기서 코드로 만든다.

LLM은 이 문장들 중 질문에 답하는 것의 id만 고르고(final_answer), 글은 직접 쓰지 않는다.
그래서 숫자와 대상(제품·국가·달·모델)의 짝이 틀릴 수 없다. 남는 위험은 '엉뚱한 문장을 고르는 것'인데,
문장마다 대상을 이름과 코드로 밝혀 두므로 고른 문장이 질문과 다르면 읽는 사람이 바로 알 수 있다.
문장마다 뜻이 드러나는 키(예: '2026-04:예측', '최저모델')를 붙인다. 번호만 주었을 때 작은 모델이 몇 번째가
무엇인지 헷갈려 엉뚱한 달을 골랐다 (이미 본 질문으로 돌렸을 때).
문장은 qa_tools가 돌려준 값(반올림한 값)만 쓴다. 비교(가장 낮은 모델, 가장 많은 달)도 여기서 계산한다.
"""
from . import config

COUNTRY_KO = {"CHN": "중국", "HKG": "홍콩", "TUR": "튀르키예", "VNM": "베트남", "USA": "미국", "MEX": "멕시코",
              "THA": "태국", "IND": "인도", "HUN": "헝가리", "JPN": "일본"}
SCOPE = ("이 결과는 UN Comtrade의 한국 ABS·PC 수출 통계로 만들었다. 다른 제품(예: PP, PE)이나 롯데케미칼 판매 자료는 "
         "없다.")
RANGE = f"미래 예측이 있는 달은 {config.END + 1} ~ {config.END + config.H}뿐이다. 그 밖의 기간 예측은 없다."


# ---------------------------------------------------------------- 조사 (받침에 따라 은/는, 이다/다, 과/와, 이라/라)
def _batchim(word):
    """마지막 글자에 받침 소리가 있는가. 'A(B)'는 괄호 앞 낱말로, 영문자는 읽는 소리(엘·엠·엔·알)로 본다."""
    w = word.strip()
    if w.endswith(")") and "(" in w:
        w = w[: w.rfind("(")].strip()
    ch = w[-1] if w else "가"
    if "가" <= ch <= "힣":
        return (ord(ch) - 0xAC00) % 28 != 0
    if ch.isdigit():
        return ch in "013678"
    return ch.upper() in "LMNR"


def eun(w):
    return w + ("은" if _batchim(w) else "는")


def ida(w):
    return w + ("이다" if _batchim(w) else "다")


def gwa(w):
    return w + ("과" if _batchim(w) else "와")


def ira(w):
    return w + ("이라" if _batchim(w) else "라")


# ---------------------------------------------------------------- 공통
def label(s):
    """시리즈 이름 -> 답에 그대로 들어갈 이름 (대상을 헷갈리지 않게 코드도 함께 쓴다)."""
    if s == "TOTAL":
        return "ABS+PC 전체 합계(TOTAL)"
    if s in ("ABS", "PC"):
        return f"{s} 전체 합계({s})"
    prod, c = s.split("-", 1)
    if c == "OTH":
        return f"{prod}-기타 국가({s}, 상위 8개국 밖 나머지)"
    return f"{prod}-{COUNTRY_KO.get(c, c)}({s})"


def _key(text):
    return text.replace(" ", "")


def _list(d, fmt="{}%"):
    return ", ".join(f"{k} {fmt.format(v)}" for k, v in d.items())


def _signed(v):
    """부호 있는 비율: +1.3%, -0.3%, 0은 0.0%."""
    return "0.0%" if v == 0 else f"{v:+.1f}%"


def _tie(best, d):
    others = [k for k, v in d.items() if k != best and v == d[best]]
    return f" (소수 첫째 자리까지는 {', '.join(others)}도 같다)" if others else ""


def _lgbm_vs_ets(va, vb, what):
    """LightGBM(va)과 ETS(vb)의 WAPE 비교 문장."""
    if va == vb:
        return f"{what} LightGBM과 ETS의 WAPE는 {va}%로 같다."
    lo = "LightGBM" if va < vb else "ETS"
    return f"{what} LightGBM과 ETS 중 WAPE가 낮은(더 나은) 쪽은 {ida(lo)} (LightGBM {va}%, ETS {vb}%)."


def _more_less(x):
    return "많다(증가)" if x > 0 else "적다(감소)" if x < 0 else "같다"


# ---------------------------------------------------------------- get_backtest
def _overall(r):
    m, b, h1, iv, rec = r["models"], r["lightgbm_vs_baselines"], r["h1_only"], r["intervals_p10_p90"], r["reconciliation"]
    wp = {k: v["wape_pct"] for k, v in m.items()}
    best = min(wp, key=wp.get)
    out = [("모델별WAPE", f"테스트 WAPE(기준 시점 {config.TEST_ORIGINS[0]} ~ {config.TEST_ORIGINS[-1]}, 1~6개월 앞 전체, "
                         f"낮을수록 좋음): {_list(wp)}."),
           ("최저모델", f"네 모델 중 테스트 WAPE가 가장 낮은 모델은 {ida(f'{best}({wp[best]}%)')}{_tie(best, wp)}.")]
    for base, key in [("ETS", "ets"), ("계절 나이브", "seasonal_naive")]:
        rr, ci = b[f"wape_reduction_vs_{key}_pct"], b[f"wape_reduction_vs_{key}_ci90_pct"]
        out.append((f"LightGBM-{_key(base)}대비", f"LightGBM의 테스트 WAPE는 {base}보다 상대적으로 {abs(rr)}% "
                    f"{'낮다' if rr > 0 else '높다'} (90% 구간 {ci[0]}% ~ {ci[1]}%)."))
    out += [("기준시점비교", f"테스트 기준 시점 {b['n_origins']}개 중 LightGBM의 오차가 ETS보다 작았던 기준 시점은 "
                         f"{b['origins_lightgbm_better_than_ets']}개, 계절 나이브보다 작았던 기준 시점은 "
                         f"{b['origins_lightgbm_better_than_seasonal_naive']}개다."),
            ("부트스트랩", f"블록 부트스트랩 재표집 중 LightGBM의 WAPE가 ETS와 계절 나이브 둘 다보다 낮았던 비율은 "
                       f"{b['p_better_than_both']}이다 (미래에 더 나을 확률이 아니다)."),
            ("모델별편향", "편향(예측 합계와 실제 합계의 차이, +는 많게, -는 적게 예측): "
             + ", ".join(f"{k} {_signed(v['bias_pct'])}" for k, v in m.items()) + ".")]
    for k, v in m.items():
        bp = v["bias_pct"]
        if bp == 0:
            text = f"{eun(k)} 테스트 기간 전체로 치우침이 없었다 (편향 0.0%)."
        else:
            text = (f"{eun(k)} 테스트 기간 전체로 실제보다 {abs(bp)}% {'많게' if bp > 0 else '적게'} 예측했다"
                    + (" (치우침이 아주 작다)." if abs(bp) < 1 else "."))
        out.append((f"편향-{_key(k)}", text))
    out.append(("모델별MASE", "MASE(같은 시리즈의 계절 나이브 오차 대비, 낮을수록 좋음): "
                + ", ".join(f"{k} {v['mase']}" for k, v in m.items()) + "."))
    p = h1["p_lightgbm_better_than_ets"]
    out.append(("1개월앞-LightGBM과ETS", _lgbm_vs_ets(h1["lightgbm_wape_pct"], h1["ets_wape_pct"], "1개월 앞만 보면")
                + f" 재표집 중 LightGBM이 ETS보다 나았던 비율은 {p}이다" + (" (확실한 차이가 아니다)." if p < 0.9 else ".")))
    gap = iv["coverage_cqr_minus_target_pp"]
    out += [("적중률", f"P10~P90 구간의 적중률(목표 {iv['target_coverage_pct']}%)은 보정 전 "
                    f"{iv['coverage_before_calibration_pct']}%, CQR 보정 후 {iv['coverage_cqr_pct']}%(90% 구간 "
                    f"{iv['coverage_cqr_ci90_pct'][0]}% ~ {iv['coverage_cqr_ci90_pct'][1]}%)다."),
            ("적중률-목표차이", f"CQR 보정 후 적중률은 목표보다 {abs(gap)}%p "
             + ("높다. 구간이 필요보다 넓다는 뜻이다." if gap > 0 else "낮다. 구간이 필요보다 좁다는 뜻이다." if gap < 0
                else "차이가 없다."))]
    if "gate_p_better_all" in rec:
        g = rec["gate_p_better_all"]
        out.append(("합계맞추기-결정", f"합계 맞추기는 {rec['adopted_method']}를 채택했다. 검증 구간 2겹 교차 적합의 재표집 중 "
                    f"MinT가 전체·제품·하위 모두에서 BU보다 나았던 비율이 {ira(str(g))} 채택 기준 0.9"
                    + ("에 못 미쳤기 때문이다." if g < 0.9 else "를 넘었기 때문이다.")))
    else:
        out.append(("합계맞추기-결정", f"합계 맞추기는 {rec['adopted_method']}를 채택했다. 이유: {rec['decision']}."))
    for lv, d in rec["test_wape_pct"].items():
        out.append((f"합계맞추기-{_key(lv)}", f"합계 맞추기 방법별 테스트 WAPE({lv}): {_list(d)}."))
    out += [("합계맞추기-재표집", f"재표집 중 MinT가 BU보다 나았던 비율(테스트): {_list(rec['test_p_mint_better_than_bu'], '{}')}."),
            ("자료범위", SCOPE)]
    return out


def _by_h(r):
    h, wp = r["h"], r["models_wape_pct"]
    best = r.get("best_model") or min(wp, key=wp.get)
    out = [(f"{h}개월앞-모델별WAPE", f"{h}개월 앞 테스트 WAPE(낮을수록 좋음): {_list(wp)}."),
           (f"{h}개월앞-최저모델", f"{h}개월 앞에서 WAPE가 가장 낮은 모델은 {ida(best)}{_tie(best, wp)}."),
           (f"{h}개월앞-LightGBM과ETS", _lgbm_vs_ets(wp["LightGBM"], wp["ETS"], f"{h}개월 앞에서")),
           (f"{h}개월앞-적중률", f"{h}개월 앞 P10~P90 적중률(목표 {r.get('target_coverage_pct', 80)}%)은 보정 전 "
                              f"{r['coverage_before_calibration_pct']}%, CQR 보정 후 {r['coverage_cqr_pct']}%다.")]
    if "p_lightgbm_better_than_ets" in r:
        p = r["p_lightgbm_better_than_ets"]
        out.append(("1개월앞-재표집", f"1개월 앞에서 재표집 중 LightGBM이 ETS보다 나았던 비율은 {p}이다"
                    + (" (확실한 차이가 아니다)." if p < 0.9 else ".")))
    return out


def _node(r):
    s, lab = r["series"], label(r["series"])
    when = "1~6개월 앞 전체" if r["h"] == "1~6" else f"{r['h']}개월 앞"
    tag = s if r["h"] == "1~6" else f"{s}-{r['h']}개월앞"
    if "methods_wape_pct" in r:                       # 합계 노드: 합계 맞추기 방법 비교
        return [(f"{tag}-방법별WAPE", f"{lab}의 테스트 WAPE({when}): {_list(r['methods_wape_pct'])}."),
                (f"{tag}-최저방법", f"{lab}에서 WAPE가 가장 낮은 방법은 {ida(r['best_method'])}. 채택한 합계 맞추기 "
                                  f"방법은 {ida(r['adopted_method'])}."),
                (f"{tag}-구간없음", f"{lab} 같은 합계 노드에는 P10~P90 구간이 없다.")]
    wp = {k: v["wape_pct"] for k, v in r["models"].items()}
    out = [(f"{tag}-모델별WAPE", f"{lab}의 테스트 WAPE({when}, 낮을수록 좋음): {_list(wp)}."),
           (f"{tag}-최저모델", f"{lab}에서 WAPE가 가장 낮은 모델은 {ida(r['best_model'])} ({when})."),
           (f"{tag}-LightGBM과ETS", _lgbm_vs_ets(wp["LightGBM"], wp["ETS"], f"{lab}의 {when}에서"))]
    if all("bias_pct" in v for v in r["models"].values()):
        out.append((f"{tag}-모델별편향", f"{lab}의 편향({when}, +는 많게, -는 적게 예측): "
                    + ", ".join(f"{k} {_signed(v['bias_pct'])}" for k, v in r["models"].items()) + "."))
    out += [(f"{tag}-물량비중", f"{lab}의 테스트 물량은 하위 시리즈 전체 물량의 {r['test_volume_share_pct']}%다."),
            (f"{tag}-적중률", f"{lab}의 P10~P90 적중률(CQR 보정 후, {when})은 {r['coverage_cqr_pct']}%다.")]
    return out


def backtest(r):
    if "by_h" in r:
        return [t for x in r["by_h"] for t in (_node(x) if "series" in x else _by_h(x))]
    if "models_wape_pct" in r:
        return _by_h(r)
    if "series" in r:
        return _node(r)
    return _overall(r)


# ---------------------------------------------------------------- get_forecast
def forecast(r):
    lab, rows = label(r["series"]), r["rows"]
    out = []
    if rows and "actual_t" in rows[0]:               # 과거 기준 시점: 예측과 실제
        for x in rows:
            t = x["target"]
            out.append((f"{t}-예측과실제", f"{r['origin']} 기준으로 낸 {lab}의 {t}({x['h']}개월 앞) 예측은 "
                        f"{x['forecast_t']:,.1f}톤, 실제는 {x['actual_t']:,.1f}톤이었다 (오차는 실제값의 "
                        f"{x['abs_error_pct_of_actual']}%)."))
            if "p10_t" in x:
                out.append((f"{t}-구간안밖", f"{lab}의 {t} 실제값은 {r['origin']} 기준 80% 예측 구간"
                            f"({x['p10_t']:,.1f} ~ {x['p90_t']:,.1f}톤) "
                            f"{'안에 들어왔다' if x['actual_inside_p10_p90'] else '밖이었다'}."))
        return out + [("자료범위", SCOPE)]
    for x in rows:                                    # 미래 예측
        t = x["target"]
        out.append((f"{t}-예측", f"{lab}의 {t} 수출 예측({r['origin']} 기준)은 {x['forecast_t']:,.1f}톤이다."))
        if "p10_t" in x:
            out.append((f"{t}-구간", f"{lab}의 {t} 예측의 80% 구간(P10~P90)은 {x['p10_t']:,.1f} ~ {x['p90_t']:,.1f}톤이다."))
        c = x.get("change_vs_last_year_pct")
        if c is not None:
            out.append((f"{t}-작년대비", f"{lab}의 {t} 예측은 작년 같은 달 실제({x['same_month_last_year_actual_t']:,.1f}톤)"
                        f"보다 {abs(c)}% {_more_less(c)}."))
    if len(rows) > 1:
        first, last = rows[0]["target"], rows[-1]["target"]
        hi = max(rows, key=lambda x: x["forecast_t"])
        lo = min(rows, key=lambda x: x["forecast_t"])
        out += [(f"{first}~{last}-합계", f"{lab}의 {first} ~ {last} {len(rows)}개월 예측 합계는 "
                                        f"{r['rows_total_t']:,.1f}톤이다."),
                (f"{first}~{last}-최대최소", f"{lab}의 {first} ~ {last} 예측 중 가장 많은 달은 {hi['target']}"
                 f"({hi['forecast_t']:,.1f}톤), 가장 적은 달은 {lo['target']}({lo['forecast_t']:,.1f}톤)이다.")]
    return out + [("예측범위", RANGE), ("자료범위", SCOPE)]


def build(name, result):
    """도구 이름과 결과 -> [(뜻이 드러나는 키, 사실 문장)] (오류 결과면 빈 목록). 키는 한 결과 안에서 겹치지 않는다."""
    if "error" in result:
        return []
    return forecast(result) if name == "get_forecast" else backtest(result)
