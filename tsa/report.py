"""그림과 RESULTS.md를 만든다. 숫자는 모두 outputs/의 실행 결과 파일에서 읽는다 (손으로 적은 숫자 없음)."""
import json
import math
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

OUT = config.OUT_DIR
FIG = OUT / "figures"
# 색: dataviz 기본 범주 팔레트 1~4번 (검사기 통과). 청록·노랑은 배경 대비가 낮아 선 끝에 직접 라벨을 붙인다
COLORS = {"LGBM": "#2a78d6", "ETS": "#eb6834", "SNaive": "#1baf7a", "Ensemble": "#eda100"}
IV_COLORS = {"raw": "#2a78d6", "static": "#eb6834", "online": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAME = {"SNaive": "계절 나이브", "Mean12": "12개월 평균", "ETS": "ETS", "LGBM": "LightGBM",
        "Ensemble": "앙상블(검증 가중치)", "Ensemble_eq": "앙상블(1/3씩)"}
IV_NAME = {"raw": "분위수 (보정 전)", "static": "CQR (검증 구간 보정)", "online": "CQR (온라인 보정)"}
METHOD_NAME = {"Base": "조정 전 (합이 안 맞음)", "BU": "BU (하위 예측의 합)", "OLS": "OLS", "WLS_struct": "WLS (구조)",
               "WLS_var": "WLS (분산)", "MinT_shrink": "MinT (축소 추정)"}
LEVEL_KO = {"total": "전체", "product": "제품 합계", "bottom": "하위 18개"}


def _load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _style():
    plt.rcParams.update({
        "font.family": ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"],
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "savefig.dpi": 160,
        "legend.frameon": False,
    })


def _end_labels(ax, ends, x, gap):
    """선 끝 라벨이 겹치지 않게 세로로 조금씩 민다."""
    items = sorted(ends, key=lambda e: e[1])
    placed = []
    for label, y in items:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.text(x, y, label, va="center", ha="left", fontsize=9, color=INK)


# ---------------------------------------------------------------- 그림
def fig_wape_by_h(R, ax):
    hs = range(1, config.H + 1)
    ends = []
    for m in ["LGBM", "ETS", "SNaive", "Ensemble"]:
        v = [100 * R["metrics"][m]["by_h"][str(h)]["WAPE"] for h in hs]
        ax.plot(hs, v, color=COLORS[m], lw=2, marker="o", ms=5, label=NAME[m])
        ends.append((NAME[m], v[-1]))
    lo, hi = ax.get_ylim()
    _end_labels(ax, ends, config.H + 0.12, (hi - lo) * 0.05)
    ax.set_xlim(0.7, config.H + 1.6)
    ax.set_xticks(list(hs))
    ax.set_xlabel("몇 개월 앞 (h)")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("테스트 WAPE: 시점별", loc="left", fontsize=11)
    ax.legend(loc="center left", fontsize=8.5)


def fig_coverage_by_h(R, ax):
    hs = range(1, config.H + 1)
    ax.axhline(80, color=INK2, lw=1, ls="--")
    ax.text(0.75, 80.6, "목표 80%", fontsize=8.5, color=INK2)
    for kind in ["raw", "static", "online"]:
        v = [100 * R["intervals"][kind]["by_h"][str(h)]["coverage"] for h in hs]
        ax.plot(hs, v, color=IV_COLORS[kind], lw=2, marker="o", ms=5, label=IV_NAME[kind])
    ax.set_xlim(0.7, config.H + 0.3)
    ax.set_xticks(list(hs))
    ax.set_xlabel("몇 개월 앞 (h)")
    ax.set_ylabel("P10~P90 적중률 (%)")
    ax.set_title("테스트 구간 적중률: 시점별", loc="left", fontsize=11)
    ax.legend(loc="lower left", fontsize=8.5)


def fig_origin_diff(fc):
    """기준 시점별 WAPE 차이 (기준선 - LightGBM). 위로 가면 LightGBM이 나은 시점."""
    fc = fc.assign(ae=(fc.y - fc.yhat).abs())
    g = fc.groupby(["model", "origin"])
    w = (g["ae"].sum() / g["y"].sum()).unstack(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, base in zip(axes, ["ETS", "SNaive"]):
        d = 100 * (w[base] - w["LGBM"])
        ax.bar(range(len(d)), d.to_numpy(), color=COLORS["LGBM"], width=0.75)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(0, len(d), 3))
        ax.set_xticklabels([str(x) for x in d.index[::3]], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{NAME[base]} WAPE - LightGBM WAPE (%p)", loc="left", fontsize=10.5)
        ax.set_xlabel("기준 시점")
    axes[0].set_ylabel("+ 이면 LightGBM이 나음 (%p)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_origin_diff.png")
    plt.close(fig)


def fig_selection(R):
    sb = R["selection_bias"]
    v, t = 100 * np.array(sb["val_wape"]), 100 * np.array(sb["test_wape"])
    k, o = sb["winner"], sb["test_best_index"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(v, t, s=36, color=COLORS["LGBM"], edgecolor=SURFACE, linewidth=1.5, zorder=3, label="설정 30개")
    ax.scatter([v[k]], [t[k]], s=90, facecolor="none", edgecolor=INK, linewidth=1.8, zorder=4)
    ax.annotate("검증에서 고른 설정", (v[k], t[k]), xytext=(8, 8), textcoords="offset points", fontsize=9)
    if o != k:
        ax.scatter([v[o]], [t[o]], s=90, facecolor="none", edgecolor=INK2, linewidth=1.2, ls="--", zorder=4)
        ax.annotate("테스트 1등 (사후)", (v[o], t[o]), xytext=(8, -14), textcoords="offset points", fontsize=9,
                    color=INK2)
    ax.set_xlabel("검증 WAPE (%)")
    ax.set_ylabel("테스트 WAPE (%)")
    ax.set_title(f"검증 순위가 테스트에서도 유지될까? (Spearman {sb['spearman_val_test']:.2f})", loc="left",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_selection_bias.png")
    plt.close(fig)


def fig_fan(iv, wide, series=("ABS-CHN", "PC-CHN")):
    """1개월 앞 예측과 보정한 P10~P90 띠 (테스트 구간)."""
    fig, axes = plt.subplots(1, len(series), figsize=(10, 3.4))
    for ax, s in zip(axes, series):
        act = wide[s].loc[pd.Period("2021-01", "M"):]
        x = act.index.to_timestamp()
        ax.plot(x, act.to_numpy(), color=INK2, lw=1.6, label="실제")
        d = iv[(iv.series == s) & (iv.h == 1)].sort_values("target")
        tx = pd.PeriodIndex(d["target"], freq="M").to_timestamp()
        ax.fill_between(tx, d["L_static"], d["U_static"], color=COLORS["LGBM"], alpha=0.18, lw=0,
                        label="P10~P90 (CQR)")
        ax.plot(tx, d["point"], color=COLORS["LGBM"], lw=2, label="LightGBM 1개월 앞")
        ax.set_title(f"{s} (톤/월)", loc="left", fontsize=10.5)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_fan.png")
    plt.close(fig)


def figures(R):
    FIG.mkdir(parents=True, exist_ok=True)
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    fig_wape_by_h(R, axes[0])
    fig_coverage_by_h(R, axes[1])
    fig.tight_layout()
    fig.savefig(FIG / "fig_portfolio.png")
    plt.close(fig)
    fc = pd.read_csv(OUT / "test_forecasts.csv.gz")
    fig_origin_diff(fc[fc.model.isin(["LGBM", "ETS", "SNaive"])])
    fig_selection(R)
    iv = pd.read_csv(OUT / "test_intervals.csv.gz")
    wide = pd.read_csv(config.PROCESSED_DIR / "panel_tonnes.csv", index_col=0)
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    fig_fan(iv, wide)


# ---------------------------------------------------------------- RESULTS.md
def _p(x, d=1):
    return f"{100 * x:.{d}f}%"


def _pp(x, d=1):
    v = round(100 * x, d)                # 편향처럼 부호가 있는 비율
    return f"{0.0:.{d}f}%" if v == 0 else f"{v:+.{d}f}%"


def _git(ref):
    try:
        return subprocess.run(["git", "rev-parse", "--short", ref], cwd=config.ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "n/a"


def origin_wins(fc):
    """기준 시점별 승패: LightGBM의 기준 시점 WAPE가 기준선보다 낮은 횟수, ETS 대비 길이 6 원형 블록 승수."""
    fc = fc.assign(ae=(fc.y - fc.yhat).abs())
    A = fc.groupby(["model", "origin"])["ae"].sum().unstack(0)
    out = {b: (int((A[b] > A["LGBM"]).sum()), int(len(A))) for b in ("ETS", "SNaive")}
    d = (A["ETS"] - A["LGBM"]).to_numpy()
    n, k = len(d), config.BOOT_BLOCK
    blocks = [d[[(i + j) % n for j in range(k)]].sum() for i in range(n)]
    out["ETS_blocks"] = (int(sum(x > 0 for x in blocks)), n)
    return out


def results_md(R, V, F, wins=None):
    m, b, sb, iv = R["metrics"], R["bootstrap"], R["selection_bias"], R["intervals"]
    L = []
    w = L.append
    w("# RESULTS\n")
    w("> 이 파일은 `python run_all.py --stage report`가 `outputs/results.json`, `outputs/val_summary.json`, "
      "`frozen.json`에서 자동으로 만든다. 손으로 고친 숫자는 없다.\n")
    w(f"- 데이터: UN Comtrade, 한국 수출 HS 390330(ABS)·390740(PC), {F['data']['start']} ~ {F['data']['end']}, "
      f"시리즈 {len(F['series'])}개 (제품별 상위 8개국 + 기타)")
    w(f"- 검증 기준 시점 {F['windows']['val_origins'][0]} ~ {F['windows']['val_origins'][1]} (12개), "
      f"테스트 기준 시점 {F['windows']['test_origins'][0]} ~ {F['windows']['test_origins'][1]} (24개), 1~6개월 앞")
    w(f"- 동결: frozen.json 작성 {F['created']}, 태그 `freeze` = 커밋 `{_git('freeze^{commit}')}`. "
      "테스트 채점은 그 뒤 한 번.")
    if not R.get("same_data_as_freeze", True):
        w("- **경고: 이 결과는 동결 때와 다른 원자료로 계산됐다.**")
    w("")

    # 1. 핵심
    lg, et, sn = m["LGBM"]["all"], m["ETS"]["all"], m["SNaive"]["all"]
    bl = b["LGBM"]
    w("## 1. 핵심 숫자 (테스트, 미리 정한 주 지표)\n")
    w("| 항목 | 값 |\n|---|---|")
    w(f"| LightGBM WAPE | {_p(lg['WAPE'])} |")
    w(f"| ETS WAPE | {_p(et['WAPE'])} |")
    w(f"| 계절 나이브 WAPE | {_p(sn['WAPE'])} |")
    w(f"| ETS 대비 WAPE 상대 감소 | {_p(bl['ETS']['rel_reduction'])} "
      f"(90% 구간 {_p(bl['ETS']['rel_reduction_q05'])} ~ {_p(bl['ETS']['rel_reduction_q95'])}) |")
    w(f"| 계절 나이브 대비 WAPE 상대 감소 | {_p(bl['SNaive']['rel_reduction'])} "
      f"(90% 구간 {_p(bl['SNaive']['rel_reduction_q05'])} ~ {_p(bl['SNaive']['rel_reduction_q95'])}) |")
    w(f"| P(LightGBM < ETS) | {bl['ETS']['p_better']:.3f} |")
    w(f"| P(LightGBM < 계절 나이브) | {bl['SNaive']['p_better']:.3f} |")
    w(f"| **P(LightGBM < 둘 다)** | **{bl['p_better_all']:.3f}** |")
    s = iv["static"]["all"]
    w(f"| P10~P90 적중률 (CQR, 검증 구간 보정) | {_p(s['coverage'])} "
      f"(90% 구간 {_p(s['coverage_ci90'][0])} ~ {_p(s['coverage_ci90'][1])}, 목표 80%) |")
    w(f"| P10~P90 적중률 (보정 전 분위수) | {_p(iv['raw']['all']['coverage'])} |\n")
    verdict = "넘었다" if bl["p_better_all"] >= 0.9 else "넘지 못했다"
    w(f"판정: 미리 정한 기준 P >= 0.9를 테스트에서 **{verdict}** (P = {bl['p_better_all']:.3f}). "
      f"검증 구간 값은 {V['bootstrap_val']['p_better_all']:.3f}였다 (같은 구간에서 고른 모델이라 낙관적인 값).\n")
    # 좋지 않았던 점: 조건을 코드로 판정해서 적는다 (결과가 바뀌면 문장도 바뀐다)
    bad, share = [], R["series_share_test"]
    p1 = b["LGBM_h1_b3"]["ETS"]["p_better"]
    if p1 < 0.9:
        bad.append(f"1개월 앞(h=1)만 보면 ETS 대비 P = {p1:.3f}로 기준 0.9에 못 미친다 "
                   f"(WAPE LightGBM {_p(m['LGBM']['by_h']['1']['WAPE'])}, ETS {_p(m['ETS']['by_h']['1']['WAPE'])}).")
    worse = [s for s in share if R["by_series"]["LGBM"][s]["WAPE"] > R["by_series"]["ETS"][s]["WAPE"]]
    if worse:
        bad.append("ETS가 LightGBM보다 나은 시리즈: " + ", ".join(
            f"{s} (ETS {_p(R['by_series']['ETS'][s]['WAPE'])}, LightGBM {_p(R['by_series']['LGBM'][s]['WAPE'])}, "
            f"테스트 물량 비중 {_p(share[s])})" for s in sorted(worse, key=share.get, reverse=True)) + ".")
    if sb["spearman_val_test"] < 0.3:
        bad.append(f"설정 30개의 검증·테스트 순위 상관이 {sb['spearman_val_test']:.3f}이고, 검증에서 고른 설정은 "
                   f"테스트에서 {sb['n_configs']}개 중 {sb['winner_test_rank']}위다.")
    ci = iv["static"]["all"]["coverage_ci90"]
    if not ci[0] <= 0.8 <= ci[1]:
        bad.append(f"CQR(검증 구간 보정) 적중률 {_p(iv['static']['all']['coverage'])}의 90% 구간 "
                   f"({_p(ci[0])} ~ {_p(ci[1])})이 목표 80%를 포함하지 않는다.")
    if R["ext"]["test_wape_alternative"] < R["ext"]["test_wape_chosen"]:
        bad.append(f"외부 지표를 {'뺀' if R['ext']['use_ext'] else '넣은'} 쪽의 테스트 WAPE"
                   f"({_p(R['ext']['test_wape_alternative'])})가 동결한 쪽({_p(R['ext']['test_wape_chosen'])})보다 "
                   "낮았다. 검증 규칙대로 정한 것이라 바꾸지 않았다.")
    if bad:
        w("좋지 않았던 점:\n")
        for x in bad:
            w(f"- {x}")
        w("")
    w("P는 기준 시점을 길이 6 블록으로 다시 뽑아 얻은 '오차 합 차이'의 분포를 작은 표본 보정으로 "
      "sqrt(24/18)배 넓힌 뒤, LightGBM의 WAPE가 더 낮게 나온 비율이다. 보정하지 않은 비율은 4절의 "
      "'보정 없음' 행에 있다. 미래에 더 나을 확률이 아니다.\n")

    # 2. 모델별
    w("## 2. 모델별 테스트 지표\n")
    w("| 모델 | WAPE | 편향 (+ 과대) | MASE 평균 | MASE 중앙값 | 검증 WAPE |\n|---|---|---|---|---|---|")
    for k in ["SNaive", "Mean12", "ETS", "LGBM", "Ensemble", "Ensemble_eq"]:
        a = m[k]["all"]
        vv = R["val_vs_test"].get(k, {}).get("val")
        w(f"| {NAME[k]} | {_p(a['WAPE'])} | {_pp(a['Bias'])} | {a['MASE']:.3f} | {a['MASE_median']:.3f} | "
          f"{_p(vv) if vv is not None else '-'} |")
    ew = F["ensemble_weights"]
    w(f"\n앙상블 가중치(검증에서 선택): 계절 나이브 {ew['SNaive']}, ETS {ew['ETS']}, LightGBM {ew['LGBM']}. "
      f"'기타' 2개 시리즈를 뺀 16개 국가 시리즈 WAPE: "
      + ", ".join(f"{NAME[k]} {_p(R['wape_without_others'][k])}" for k in ["SNaive", "ETS", "LGBM"]) + ".\n")

    # 3. 시점별 / 제품별
    w("## 3. 시점별 WAPE\n")
    w("| h | " + " | ".join(NAME[k] for k in ["SNaive", "ETS", "LGBM", "Ensemble"]) + " |\n|---|---|---|---|---|")
    for h in range(1, config.H + 1):
        w(f"| {h} | " + " | ".join(_p(m[k]["by_h"][str(h)]["WAPE"]) for k in ["SNaive", "ETS", "LGBM", "Ensemble"])
          + " |")
    c0, c1 = R["common_targets"]
    w(f"\n예측 대상 월을 {c0} ~ {c1}로 맞춘 시점별 WAPE (모든 h가 같은 달을 예측하는 구간):\n")
    w("| h | " + " | ".join(NAME[k] for k in ["SNaive", "ETS", "LGBM"]) + " |\n|---|---|---|---|")
    for h in range(1, config.H + 1):
        w(f"| {h} | " + " | ".join(_p(R["by_h_common"][k][str(h)]) for k in ["SNaive", "ETS", "LGBM"]) + " |")
    w("\n제품별:\n")
    w("| 제품 | " + " | ".join(NAME[k] for k in ["SNaive", "ETS", "LGBM"]) + " |\n|---|---|---|---|")
    for p in config.PRODUCTS.values():
        w(f"| {p} | " + " | ".join(f"{_p(m[k]['by_product'][p]['WAPE'])} (편향 {_pp(m[k]['by_product'][p]['Bias'])})"
                                    for k in ["SNaive", "ETS", "LGBM"]) + " |")

    # 4. 부트스트랩
    w("\n## 4. 블록 부트스트랩 (테스트 기준 시점 24개, 10,000회)\n")
    w("| 비교 | 설정 | P(나음) vs ETS | P(나음) vs 계절 나이브 | P(둘 다) |\n|---|---|---|---|---|")
    rows = [("LightGBM", "LGBM", "블록 6, 보정 (주)"), ("앙상블", "Ensemble", "블록 6, 보정"),
            ("LightGBM", "LGBM_b3", "블록 3, 보정 (민감도)"), ("LightGBM", "LGBM_b6_raw", "블록 6, 보정 없음 (민감도)"),
            ("LightGBM, h=1만", "LGBM_h1_b3", "블록 3, 보정")]
    for label, key, desc in rows:
        x = b[key]
        w(f"| {label} | {desc} | {x['ETS']['p_better']:.3f} | {x['SNaive']['p_better']:.3f} | {x['p_better_all']:.3f} |")
    if wins:
        w(f"\n기준 시점별로 보면 LightGBM의 WAPE가 ETS보다 낮은 기준 시점은 {wins['ETS'][0]}/{wins['ETS'][1]}개, "
          f"계절 나이브보다 낮은 기준 시점은 {wins['SNaive'][0]}/{wins['SNaive'][1]}개다. "
          f"연속 6개 기준 시점을 묶은 원형 블록 {wins['ETS_blocks'][1]}개 중 LightGBM의 오차 합이 ETS보다 작은 블록은 "
          f"{wins['ETS_blocks'][0]}개다. "
          + ("그래서 어떤 블록 조합을 뽑아도 LightGBM이 이기고 P가 1에 가깝게 나온다. "
             if wins["ETS_blocks"][0] == wins["ETS_blocks"][1] else "")
          + f"다만 서로 겹치지 않는 {config.BOOT_BLOCK}개월 블록은 {config.N_TEST // config.BOOT_BLOCK}개뿐이라, "
          "P가 1에 가깝다고 '확실하다'로 읽으면 안 된다. 개선 폭의 90% 구간(1절)을 함께 본다.")

    # 5. 선택 편향
    w("\n## 5. 선택 편향 (LightGBM 설정 30개)\n")
    w("| 항목 | 값 |\n|---|---|")
    w(f"| 고른 설정의 검증 WAPE -> 테스트 WAPE | {_p(sb['val_winner'])} -> {_p(sb['test_winner'])} |")
    w(f"| 30개 설정 중앙값: 검증 -> 테스트 | {_p(sb['val_median'])} -> {_p(sb['test_median'])} |")
    w(f"| 기간 효과(중앙 설정의 검증->테스트 변화) | {sb['median_shift_pct']:+.1f}% |")
    w(f"| 선택 편향(고른 설정이 기간 효과보다 더 나빠진 정도) | {sb['optimism_pct']:+.1f}% |")
    w(f"| 검증·테스트 순위 상관 (Spearman) | {sb['spearman_val_test']:.3f} |")
    w(f"| 고른 설정의 테스트 순위 | {sb['winner_test_rank']} / {sb['n_configs']} |")
    w(f"| 테스트로 골랐다면 (사후 1등) | {_p(sb['test_best'])} (정직한 값보다 {sb['oracle_gap_pp']:.2f}%p 좋아 보임) |")
    w("\n선택 편향 = [ln(테스트/검증)의 고른 설정 값] - [그 값의 30개 중앙값]. 한 번의 분할에서 나온 값이라 잡음이 크다.\n")

    # 6. 외부 지표
    e = R["ext"]
    w("## 6. 외부 지표 (원/달러 환율, 브렌트유)\n")
    w(f"- 검증 구간 P(넣은 쪽이 나음) = {e['p_better_val']:.3f} -> {'사용' if e['use_ext'] else '사용 안 함'} (기준 0.9)")
    alt_name = "넣지 않은" if e["use_ext"] else "넣은"
    w(f"- 테스트 WAPE: 고른 쪽 {_p(e['test_wape_chosen'])}, {alt_name} 쪽 {_p(e['test_wape_alternative'])} "
      "(사후 확인용, 선택에 쓰지 않음)\n")

    # 7. 구간
    w("## 7. 확률 예측 (P10~P90, 목표 적중률 80%)\n")
    w("| 방법 | 적중률 | 90% 구간 | 아래로 벗어남 | 위로 벗어남 | 상대 폭 | 상대 구간 점수 | 시리즈별 적중률 범위 |\n"
      "|---|---|---|---|---|---|---|---|")
    for kind in ["raw", "static", "online"]:
        a = iv[kind]["all"]
        mn, mx = iv[kind]["series_coverage_min_max"]
        w(f"| {IV_NAME[kind]} | {_p(a['coverage'])} | {_p(a['coverage_ci90'][0])} ~ {_p(a['coverage_ci90'][1])} | "
          f"{_p(a['below'])} | {_p(a['above'])} | {_p(a['width_rel'])} | {a['interval_score_rel']:.3f} | "
          f"{_p(mn, 0)} ~ {_p(mx, 0)} |")
    w("\n시점별 적중률:\n")
    w("| h | " + " | ".join(IV_NAME[k] for k in ["raw", "static", "online"]) + " |\n|---|---|---|---|")
    for h in range(1, config.H + 1):
        w(f"| {h} | " + " | ".join(_p(iv[k]["by_h"][str(h)]["coverage"]) for k in ["raw", "static", "online"]) + " |")
    w("\n제품별 적중률 (CQR, 검증 구간 보정): "
      + ", ".join(f"{p} {_p(iv['static']['by_product'][p]['coverage'])}" for p in config.PRODUCTS.values()) + ".")
    w("상대 폭 = sum(U - L) / sum(실제값). 상대 구간 점수 = Winkler 점수 합 / 실제값 합 (낮을수록 좋음).\n")

    # 8. 시리즈별
    w("## 8. 시리즈별 테스트 WAPE\n")
    w("| 시리즈 | 테스트 물량 비중 | 계절 나이브 | ETS | LightGBM |\n|---|---|---|---|---|")
    share = R["series_share_test"]
    for s in sorted(share, key=share.get, reverse=True):
        w(f"| {s} | {_p(share[s])} | " + " | ".join(_p(R["by_series"][k][s]["WAPE"]) for k in ["SNaive", "ETS", "LGBM"])
          + " |")

    # 9. 기타
    fg = R["feature_gain_share"]
    top = sorted(fg, key=fg.get, reverse=True)[:8]
    w("\n## 9. 그 밖에\n")
    w("- 고른 LightGBM 설정: " + ", ".join(f"{k}={v}" for k, v in F["lgbm"]["params"].items()))
    w("- 피처 중요도(gain 비중, 마지막 테스트 기준 시점 모델): " + ", ".join(f"{k} {_p(fg[k])}" for k in top))
    w(f"- ETS 적합 실패로 계절 나이브를 대신 쓴 횟수: 검증 {V['ets_fallbacks']}, 테스트 {R['ets_fallbacks_test']}")
    w(f"- q10·P50·q90을 크기순으로 다시 놓은 행의 비율: 검증 {_p(V['intervals_reordered_share'])}, "
      f"테스트 {_p(R['intervals_reordered_share_test'])} ('보정 전' 구간은 이 정렬 뒤의 q10~q90)")
    qtxt = ", ".join(f"h{h}: ({v[0]:+.3f}, {v[1]:+.3f})" for h, v in F["conformal"]["Q"].items())
    w(f"- CQR 보정폭 (Q_lo, Q_hi), 규모 대비: {qtxt}. 양수면 구간을 넓히고 음수면 좁힌다.")
    w("\n그림: `outputs/figures/` (fig_portfolio, fig_origin_diff, fig_selection_bias, fig_fan)")
    return "\n".join(L) + "\n"


def mint_md(M, MV, FM):
    """10절: 계층 조정. mint_results.json(테스트), mint_val.json(검증), frozen_mint.json(방법 선택)에서 만든다."""
    L = []
    w = L.append
    lv = ["total", "product", "bottom"]
    g, mt, b = FM["gate"], M["metrics"]["LGBM"], M["bootstrap"]["LGBM"]
    w("\n## 10. 계층 조정 (7번: 제품 합계·전체 합계 맞추기)\n")
    w("- 계층: 전체 -> ABS, PC -> 하위 18개 (노드 21개). 합계 노드의 기준 예측은 합계 시리즈에 직접 적합한 AutoETS, "
      "하위는 동결한 LightGBM이다. 따로 만든 예측이라 그대로는 합이 맞지 않는다.")
    gap = M["coherence_gap_base"]["LGBM"]
    w(f"- 조정 전 불일치 (테스트, sum|합계 노드 예측 - 하위 예측의 합| / sum 실제값): 전체 {_p(gap['TOTAL'])}, "
      f"ABS {_p(gap['ABS'])}, PC {_p(gap['PC'])}")
    w(f"- 방법 선택 (검증 구간): 후보 MinT(축소 추정), 기본값 BU. frozen_mint.json 작성 {FM['created']}, 태그 "
      f"`freeze-mint` = 커밋 `{_git('freeze-mint^{commit}')}`, 테스트 채점은 그 뒤 한 번. 검증 기준 시점 12개를 앞뒤 6개로 "
      f"나눈 2겹 교차 적합에서 P(MinT가 전체·제품·하위 모두에서 BU보다 나음) = {g['p_better_all']:.3f} "
      f"-> **{FM['adopted']} 채택** (기준 0.9).")
    ins, cf = MV["in_sample"], MV["cross_fit"]
    w(f"- W를 같은 검증 구간 오차로 추정하고 같은 구간에서 평가하면 MinT의 하위 WAPE가 "
      f"{_p(ins['MinT_shrink']['bottom']['WAPE'])} (BU {_p(ins['BU']['bottom']['WAPE'])})로 좋아 보였지만, "
      f"교차 적합에서는 {_p(cf['MinT_shrink']['bottom']['WAPE'])}로 BU보다 나빴다.\n")
    w("| 방법 (테스트, 하위 기준 예측 = LightGBM) | 전체 WAPE | 제품 합계 WAPE | 하위 18개 WAPE | 편향 (전체 / 하위) "
      "| 0으로 자른 하위 예측 |\n|---|---|---|---|---|---|")
    neg = M["neg_share"]["LGBM"]
    for m in ["Base", "BU", "OLS", "WLS_struct", "WLS_var", "MinT_shrink"]:
        t = mt[m]["by_level"]
        mark = " **(채택)**" if m == FM["adopted"] else ""
        w(f"| {METHOD_NAME[m]}{mark} | " + " | ".join(_p(t[x]["WAPE"]) for x in lv)
          + f" | {_pp(t['total']['Bias'])} / {_pp(t['bottom']['Bias'])} | {_p(neg[m], 2) if m in neg else '-'} |")
    x = b["MinT_vs_BU"]
    w("\n- 테스트 P(MinT < BU): " + ", ".join(f"{LEVEL_KO[k]} {x[k]['p_better']:.3f}" for k in lv)
      + f", 세 수준 모두 {x['p_better_all']:.3f}. 하위에서 MinT의 WAPE가 BU보다 상대적으로 "
      f"{_p(-x['bottom']['rel_reduction'])} 높았다 (90% 구간 {_p(-x['bottom']['rel_reduction_q95'])} ~ "
      f"{_p(-x['bottom']['rel_reduction_q05'])}). MinT 편향 {_pp(mt['MinT_shrink']['by_level']['bottom']['Bias'])}: "
      "검증 구간(수출 감소기)의 과대 예측 오차로 만든 W가 테스트 구간 예측을 아래로 끌어내린 것으로 보인다.")
    y = b["BU_vs_Base"]
    better = [k for k in ("total", "product") if y[k]["p_better"] >= 0.9]
    w("- P(BU < 조정 전): " + ", ".join(f"{LEVEL_KO[k]} {y[k]['p_better']:.3f}" for k in ("total", "product"))
      + (f". {'·'.join(LEVEL_KO[k] for k in better)}에서는 합계를 직접 ETS로 예측하는 것보다 하위 LightGBM 예측을 "
         "더하는 쪽이 나았다 (P >= 0.9)." if better else ". 차이가 기준(0.9)을 넘지 않았다."))
    node = mt["Base"]["by_node"], mt["BU"]["by_node"]
    w(f"- 노드별 (조정 전 -> BU): 전체 {_p(node[0]['TOTAL']['WAPE'])} -> {_p(node[1]['TOTAL']['WAPE'])}, "
      f"ABS {_p(node[0]['ABS']['WAPE'])} -> {_p(node[1]['ABS']['WAPE'])}, "
      f"PC {_p(node[0]['PC']['WAPE'])} -> {_p(node[1]['PC']['WAPE'])}")
    me, be = M["metrics"]["ETS"], M["bootstrap"]["ETS"]["MinT_vs_BU"]
    w(f"- 참고 (하위 기준 예측도 ETS인 고전적 설정): 하위 WAPE BU {_p(me['BU']['by_level']['bottom']['WAPE'])}, "
      f"MinT {_p(me['MinT_shrink']['by_level']['bottom']['WAPE'])}, 제품 합계 BU "
      f"{_p(me['BU']['by_level']['product']['WAPE'])}, MinT {_p(me['MinT_shrink']['by_level']['product']['WAPE'])}, "
      f"P(MinT < BU, 세 수준 모두) = {be['p_better_all']:.3f}.")
    w(f"- MinT 축소 강도 λ = {M['lambda']['LGBM']:.3f} (1이면 상관을 버리고 대각만 씀). 오차 72개가 기준 시점과 h로 "
      "서로 겹쳐 실제 정보보다 많아 보였고, 그래서 노드 간 상관을 거의 그대로 믿은 것으로 본다.")
    w("- P10~P90 구간은 하위 시리즈에만 있다 (분위수는 더해지지 않아서 합계 노드 구간은 만들지 않았다).")
    return "\n".join(L) + "\n"


# 에이전트 평가 실행. 있는 파일만 11절에 싣고, 질문별 상세 표는 뜻 검토를 한 실행만 싣는다
AGENT_RUNS = [   # (결과 파일, 짧은 이름, 설명, 질문별 상세 표를 실을지)
    ("agent_eval.json", "개발 1", "처음 에이전트, 개발용 질문 10개 (실행 전에 고정)", True),
    ("agent_eval_dev2.json", "개발 2", "1번 고친 에이전트, 같은 개발용 질문 (고칠 때 본 질문이라 낙관적)", False),
    ("agent_eval_dev3.json", "개발 3", "2번 고친 에이전트, 같은 개발용 질문 (고칠 때 본 질문이라 낙관적)", False),
    ("agent_eval_dev4.json", "개발 4", "3번 고친 에이전트(2차 동결본), 같은 개발용 질문 (고칠 때 본 질문이라 낙관적)", True),
    ("agent_eval_test_v1.json", "테스트·처음", "처음 에이전트(개발 1과 같은 코드, 질문 파일만 바꿔 별도 worktree에서), 테스트 "
                                              "질문 10개 (동결본 결과를 본 뒤 비교용으로 정해 돌림)", True),
    ("agent_eval_test.json", "테스트·동결본", "2차 동결본(태그 `freeze-agent`), 테스트 질문 10개 (독립 에이전트가 작성, 고친 "
                                         "사람은 실행 전까지 보지 않음, 한 번 실행)", True),
    ("agent_eval_v3dev4.json", "3차·개발용", "3차 동결본, 개발용 질문 10개 (3차를 고칠 때 본 질문이라 낙관적)", False),
    ("agent_eval_v3t1d.json", "3차·테스트", "3차 동결본, 1차 테스트 질문 10개 (3차를 고칠 때 본 질문이라 낙관적)", False),
    ("agent_eval_t2_v1.json", "테스트2·처음", "처음 에이전트, 새 테스트 질문 10개 (테스트2, 독립 에이전트가 작성, 한 번)", True),
    ("agent_eval_t2_v2.json", "테스트2·2차", "2차 동결본(`freeze-agent`), 같은 테스트2 질문 (한 번)", True),
    ("agent_eval_t2_v3.json", "테스트2·3차", "3차 동결본(`freeze-agent-v3`), 같은 테스트2 질문 (한 번)", True),
]
# 같은 질문으로 돌린 두 실행의 짝 비교 (짧은 이름)
AGENT_PAIRS = [("테스트·처음", "테스트·동결본"), ("테스트2·처음", "테스트2·2차"), ("테스트2·처음", "테스트2·3차"),
               ("테스트2·2차", "테스트2·3차")]


def mcnemar(A, B, key="faithful", review="semantic_review"):
    """같은 질문에서 A만 뜻 검토를 통과한 개수, B만 통과한 개수, 맥니마 정확 검정의 양쪽 p값.
    결과가 갈린 질문만 쓰고, 그 안에서 'A만 통과'의 개수가 이항분포(n, 0.5)를 따른다고 보고 계산한다.
    key: 'faithful'(통과) 또는 'claims_faithful'(주장이 모두 맞음). review: 'semantic_review'(원래 검토) 또는
    'semantic_review_blind'(가린 재검토)."""
    a, b = A[review]["by_id"], B[review]["by_id"]
    assert set(a) == set(b), "두 실행의 질문이 다르다"
    only_a = sum(a[k][key] and not b[k][key] for k in a)
    only_b = sum(b[k][key] and not a[k][key] for k in a)
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p


def agent_md(runs):
    """11절: 질의응답 에이전트 평가. runs = [(짧은 이름, 설명, ask.py --eval이 쓴 JSON)]."""
    A0 = runs[0][2]
    blind = any(A.get("semantic_review_blind") for _, _, A, _ in runs)
    L = ["\n## 11. 질의응답 에이전트 (7번)\n",
         f"- 모델 `{A0['model']}` (OpenAI 호환 API, {A0['endpoint_host']}). 질문 파일은 모두 실행 전에 커밋해 고정했다.",
         "- 숫자 검사: 답변의 숫자가 도구 결과나 질문에 있는가 (부호는 비교하지 않음). 걸리면 알려 주고 다시 쓰게 한다. "
         "버전별: 처음 에이전트·개발 2는 한 번 다시 쓰기, 개발 3부터 두 번까지. 기간 불일치·한자 검사는 개발 2부터, "
         "끝까지 걸린 문장을 빼고 보여 주는 안전장치는 개발 3부터, 범위 밖 질문 검사는 개발 4(2차 동결본)부터.",
         "- 3차: 모델은 코드가 도구 결과로 만든 사실 문장의 id만 고르고, 답은 고른 문장을 그대로 잇는다. 숫자 검사에 걸려도 "
         "다시 쓰게 하지 않고 걸린 문장을 뺀다. 보호 장치: 질문한 기간이 모델이 부른 도구 결과에 없거나(기준 시점을 잘못 넣으면 "
         "범위 안의 질문도 걸린다), 질문에 목록에 있는 다른 제품 이름(PP 등)이 있고 ABS·PC 이야기가 없으면 코드가 '답할 수 "
         "없음'과 범위 안내로 바꾼다. 이 문장은 숫자 검사를 건너뛰고 통과로 센다.",
         "- 숫자 검사 칸은 버전마다 잰 방식이 다르다. 처음 에이전트 줄은 기간 검사가 없던 그때의 검사이고, 3차 줄은 보호 "
         "장치가 바꾼 뒤의 답 기준이다. 같은 검사로 다시 본 값은 WORKLOG 16절에 있다.",
         "- 뜻 검토: 기록 원본만 받은 검토 에이전트(Claude, 고친 세션과 같은 모델이라 완전히 독립은 아님)가 두 관점(주장별 "
         "사실 확인 / 질문에 답했는지·도구 사용)으로 보여 준 답을 판정했다. 통과 = 두 검토자 모두 모든 주장이 그 답변의 도구 "
         "결과와 맞다고 보고, '질문에 답했는지' 관점 검토자가 질문에 답했다고 본 경우. '질문에 답함'과 '알맞은 도구·인자'는 "
         "그 검토자 한 명의 판정이다. 검토자는 답변 파일 이름으로 버전을 알 수 있었고, 실행마다 새로 띄웠다. 읽는 파일도 "
         "막지 않아서, 일부 검토자는 이전 실행의 판정 파일과 git 기록을 봤다.\n",
         "| 실행 | 에이전트와 질문 | 숫자 검사 통과 (첫 답변 / 최종) | 보여 준 답의 숫자 검사 | 기대한 도구를 부름 | "
         "뜻 검토 통과 |" + (" 가린 재검토 통과 |" if blind else "") + " 가장 긴 요청 (토큰) |",
         "|---|---|---|---|---|---|" + ("---|" if blind else "") + "---|"]
    for short, desc, A, _ in runs:
        n, sr, sb = A["n_questions"], A.get("semantic_review"), A.get("semantic_review_blind")
        sem = f"{sr['faithful']}/{n}" if sr else "-"
        tok = f"{A['max_prompt_tokens']:,}" if A.get("max_prompt_tokens") else "-"
        shown = A.get("delivered_ok", A["final_ok"])        # 안전장치가 없던 실행은 모델 답을 그대로 보여 줬다
        cell = f" {sb['faithful']}/{n} |" if sb else (" - |" if blind else "")
        L.append(f"| {short} | {desc} | {A['first_try_ok']}/{n} / {A['final_ok']}/{n} | {shown}/{n} | "
                 f"{A['tool_choice_ok']}/{n} | {sem} |" + cell + f" {tok} |")
    L.append("")
    L.append("질문이 실행마다 10개뿐이라 몇 개 차이는 우연일 수 있다. 버전을 공정하게 비교하는 것은 같은 질문으로 돌린 "
             "'테스트' 두 줄과 '테스트2' 세 줄이다. 개발 실행과 3차의 '개발용'·'테스트' 줄은 고칠 때 본 질문이라 낙관적이다. "
             "테스트2 질문은 3차를 동결하기 전에 독립 에이전트가 따로 썼고, 에이전트를 고친 세션은 세 버전을 한 번씩 돌릴 "
             "때까지 질문을 열지 않았다 (기록일 뿐 확인할 장치는 없다). 다만 테스트2 질문 칸은 1차 테스트를 따라 정했고, 3차는 "
             "1차 테스트 질문을 보며 고쳤으므로 테스트2는 3차에 유리할 수 있다.")
    by = {short: A for short, _, A, _ in runs}
    pairs = [(x, y) for x, y in AGENT_PAIRS
             if x in by and y in by and by[x].get("semantic_review") and by[y].get("semantic_review")]
    if pairs:
        L.append("")
        L.append("같은 질문 짝 비교 (뜻 검토 통과 기준, 맥니마 정확 검정, 양쪽):")
        L.append("")
        for x, y in pairs:
            only_x, only_y, p = mcnemar(by[x], by[y])
            L.append(f"- {x} {by[x]['semantic_review']['faithful']}/{by[x]['n_questions']} vs "
                     f"{y} {by[y]['semantic_review']['faithful']}/{by[y]['n_questions']}: "
                     f"{x}만 통과 {only_x}개, {y}만 통과 {only_y}개, p = {p:.3f}")
        L.append("")
        L.append("참고 (테스트2 결과를 본 뒤 고른 지표라 미리 정한 비교가 아님): '주장이 모두 맞음'만 기준으로 같은 짝 비교.")
        L.append("")
        for x, y in pairs:
            only_x, only_y, p = mcnemar(by[x], by[y], "claims_faithful")
            L.append(f"- {x} {by[x]['semantic_review']['claims_faithful']}/{by[x]['n_questions']} vs "
                     f"{y} {by[y]['semantic_review']['claims_faithful']}/{by[y]['n_questions']}: "
                     f"{x}만 맞음 {only_x}개, {y}만 맞음 {only_y}개, p = {p:.3f}")
    if blind:
        L.append("")
        L.append("가린 재검토: 같은 답을 버전을 알려 주는 항목을 빼고 질문 묶음별로 섞어 다시 검토했다 (계획은 실행 전에 "
                 "WORKLOG 16절에 적음). 주 지표는 위의 원래 검토이고, 이것은 검토 조건을 바꿔도 결론이 유지되는지 보는 확인이다. "
                 "가림은 완전하지 않다. 처음 에이전트의 도구 결과에는 summary_ko가 없고 3차 답은 모양이 달라 버전 무리는 알아볼 수 "
                 "있었고, 검토자마다 자기 묶음의 답 전부가 든 파일을 받았다. 두 관점 지시문과 판정 규칙은 원래 검토와 같지만, 가린 "
                 "재검토에는 숫자 검사 기록과 '숫자 검사가 못 보는 것을 찾으라'는 안내가 없었다.")
        L.append("")
        same_all = n_all = 0
        for short, _, A, _ in runs:
            if not A.get("semantic_review_blind"):
                continue
            o, b = A["semantic_review"]["by_id"], A["semantic_review_blind"]["by_id"]
            same = sum(o[k]["faithful"] == b[k]["faithful"] for k in o)
            only_o = sum(o[k]["faithful"] and not b[k]["faithful"] for k in o)
            only_b = sum(b[k]["faithful"] and not o[k]["faithful"] for k in o)
            same_all, n_all = same_all + same, n_all + len(o)
            L.append(f"- {short}: 원래 {A['semantic_review']['faithful']}/{len(o)}, 가린 재검토 "
                     f"{A['semantic_review_blind']['faithful']}/{len(o)}. 통과 판정이 같은 질문 {same}/{len(o)} "
                     f"(원래만 통과 {only_o}, 가린 재검토만 통과 {only_b})")
        L.append(f"- 모두 합쳐 통과 판정이 같은 답 {same_all}/{n_all}")
        bpairs = [(x, y) for x, y in pairs if by[x].get("semantic_review_blind") and by[y].get("semantic_review_blind")]
        if bpairs:
            L.append("")
            L.append("가린 재검토 기준 같은 질문 짝 비교 (맥니마 정확 검정, 양쪽):")
            L.append("")
            for x, y in bpairs:
                only_x, only_y, p = mcnemar(by[x], by[y], review="semantic_review_blind")
                L.append(f"- {x} {by[x]['semantic_review_blind']['faithful']}/{by[x]['n_questions']} vs "
                         f"{y} {by[y]['semantic_review_blind']['faithful']}/{by[y]['n_questions']}: "
                         f"{x}만 통과 {only_x}개, {y}만 통과 {only_y}개, p = {p:.3f}")
            L.append("")
            L.append("참고 (결과를 본 뒤 고른 지표): 가린 재검토 기준 '주장이 모두 맞음' 짝 비교.")
            L.append("")
            for x, y in bpairs:
                only_x, only_y, p = mcnemar(by[x], by[y], "claims_faithful", review="semantic_review_blind")
                L.append(f"- {x} {by[x]['semantic_review_blind']['claims_faithful']}/{by[x]['n_questions']} vs "
                         f"{y} {by[y]['semantic_review_blind']['claims_faithful']}/{by[y]['n_questions']}: "
                         f"{x}만 맞음 {only_x}개, {y}만 맞음 {only_y}개, p = {p:.3f}")
    for short, desc, A, detail in runs:
        if not detail:
            continue
        n, sr, sb = A["n_questions"], A.get("semantic_review"), A.get("semantic_review_blind")
        L.append("")
        L.append(f"### {short}: {desc}")
        L.append("")
        L.append(f"실행이 끝난 시각 {A['created']}" + (f", 에이전트 커밋 `{A['agent_commit']}`" if A.get("agent_commit") else "")
                 + f". 답변 속 숫자 {A['numbers_first_try']}개 중 근거 없는 숫자 {A['unsupported_first_try']}개 (첫 답변)"
                 + (f", 기간 불일치·한자 문제 {A['issues_first_try']}건 (첫 답변)" if "issues_first_try" in A else "") + "."
                 + (f" {'숫자 검사에 걸려' if A.get('agent_version') == 'v3' else '다시 써도 걸려서 안전장치로'} 문장을 뺀 "
                    f"답변 {A['redacted_answers']}개 (보여 준 답의 숫자 검사 통과 {A['delivered_ok']}/{n})."
                    if "redacted_answers" in A else ""))
        kinds = A.get("answer_kinds", {})
        if any(k != "text" for k in kinds):
            names = {"final": "final_answer로 고름", "text_ids": "글에 적은 id를 읽음",
                     "guard": "질문한 기간이 도구 결과에 없거나 자료 밖 제품이라 보호 장치가 '답할 수 없음'으로 바꿈",
                     "none": "고르지 못함('도구 결과로 확인할 수 있는 답을 만들지 못했습니다')"}
            L.append("3차 답이 만들어진 방식: " + ", ".join(f"{names.get(k, k)} {v}" for k, v in kinds.items()) + ".")
        if sr:
            L.append(f"뜻 검토 항목별: 주장이 모두 맞음 {sr['claims_faithful']}/{n}, 질문에 답함 {sr['answers_question']}/{n}, "
                     f"알맞은 도구·인자 {sr['right_tool_and_args']}/{n}, 한국어로만 답함 {sr['korean_only']}/{n}. "
                     f"보여 준 답이 숫자 검사는 통과했지만 뜻 검토에서 걸린 답변 {sr['passed_numbers_but_unfaithful']}개.")
        if sb:
            L.append(f"가린 재검토 항목별: 주장이 모두 맞음 {sb['claims_faithful']}/{n}, 질문에 답함 {sb['answers_question']}/{n}, "
                     f"알맞은 도구·인자 {sb['right_tool_and_args']}/{n}, 한국어로만 답함 {sb['korean_only']}/{n}.")
        red = "redacted_answers" in A
        L += ["", "| 질문 | 부른 도구 | 첫 답변 | 최종 | 첫 답변에서 걸린 것 |" + (" 뺀 문장 |" if red else "")
              + (" 뜻 검토 |" if sr else "") + (" 가린 재검토 |" if sb else ""),
              "|---|---|---|---|---|" + ("---|" if red else "") + ("---|" if sr else "") + ("---|" if sb else "")]
        for q in A["questions"]:
            caught = ", ".join(q["unsupported_first"] + q.get("issues_first", [])) or "-"
            row = (f"| {q['question']} | {', '.join(q['called_tools']) or '-'} | {'통과' if q['first_try_ok'] else '실패'} | "
                   f"{'통과' if q['final_ok'] else '실패'} | {caught} |" + (f" {q['redacted']} |" if red else ""))
            for review in (sr, sb):
                if review:
                    v = review["by_id"].get(q["id"], {})
                    row += f" {'맞음' if v.get('faithful') else '문제: ' + v.get('summary_ko', '-')} |"
            L.append(row)
    L.append("\n숫자 검사는 '답변의 숫자가 도구 결과 어딘가에 있는가'만 본다. 맞는 항목과 짝지었는지까지는 확인하지 않아서 "
             "뜻 검토를 따로 했다. 검토자도 AI라 답변 전문과 판정 전문을 outputs/cache/에 남겼다 (커밋하지 않음, 이 PC에만 "
             "있음). 검토 지시문과 집계 스크립트는 tools/agent_eval/에 있다.")
    return "\n".join(L) + "\n"


def summary_table(R):
    """README·SUMMARY에 넣는 표 1개: 모델별 테스트 WAPE·편향·MASE + 부트스트랩 + 구간 적중률."""
    m, bl, iv = R["metrics"], R["bootstrap"]["LGBM"], R["intervals"]
    L = ["| 모델 | 테스트 WAPE | 편향 | MASE |", "|---|---|---|---|"]
    for k in ["SNaive", "ETS", "LGBM", "Ensemble"]:
        a = m[k]["all"]
        L.append(f"| {NAME[k]} | {_p(a['WAPE'])} | {_pp(a['Bias'])} | {a['MASE']:.3f} |")
    L.append("")
    L.append(f"- P(LightGBM의 WAPE < ETS) = {bl['ETS']['p_better']:.3f}, "
             f"P(< 계절 나이브) = {bl['SNaive']['p_better']:.3f}, P(둘 다) = {bl['p_better_all']:.3f} "
             "(블록 부트스트랩, 블록 6, 10,000회, 작은 표본 보정)")
    L.append(f"- P10~P90 적중률(목표 80%): 보정 전 {_p(iv['raw']['all']['coverage'])} -> "
             f"CQR 보정 후 {_p(iv['static']['all']['coverage'])}")
    return "\n".join(L)


def inject(path, name, text):
    """파일 안의 <!-- AUTO:name:START --> ~ <!-- AUTO:name:END --> 사이를 생성된 내용으로 바꾼다."""
    if not path.exists():
        return
    s = path.read_text(encoding="utf-8")
    a, b = f"<!-- AUTO:{name}:START -->", f"<!-- AUTO:{name}:END -->"
    if a in s and b in s:
        head, rest = s.split(a, 1)
        _, tail = rest.split(b, 1)
        path.write_text(f"{head}{a}\n{text}\n{b}{tail}", encoding="utf-8")


def build():
    R, V, F = _load("results.json"), _load("val_summary.json"), json.loads(config.FROZEN_PATH.read_text(encoding="utf-8"))
    fc_path = OUT / "test_forecasts.csv.gz"
    wins = origin_wins(pd.read_csv(fc_path)) if fc_path.exists() else None
    text = results_md(R, V, F, wins)
    if (OUT / "mint_results.json").exists() and config.FROZEN_MINT_PATH.exists():
        FM = json.loads(config.FROZEN_MINT_PATH.read_text(encoding="utf-8"))
        text += mint_md(_load("mint_results.json"), _load("mint_val.json"), FM)
    runs = [(short, desc, _load(f), detail) for f, short, desc, detail in AGENT_RUNS if (OUT / f).exists()]
    if runs:
        text += agent_md(runs)
    (config.ROOT / "RESULTS.md").write_text(text, encoding="utf-8")
    for doc in ("README.md", "SUMMARY.md"):
        inject(config.ROOT / doc, "summary_table", summary_table(R))
    print("RESULTS.md를 만들었습니다.")
    needed = [OUT / "test_forecasts.csv.gz", OUT / "test_intervals.csv.gz", config.PROCESSED_DIR / "panel_tonnes.csv"]
    missing = [p.name for p in needed if not p.exists()]
    if missing:   # 예측 파일은 커밋하지 않으므로 새로 받은 저장소에서는 --stage final을 먼저 돌려야 한다
        print(f"그림은 건너뜁니다: {missing} 없음. `python run_all.py`로 전체를 먼저 실행하세요.")
        return
    figures(R)
    print("그림을 만들었습니다.")
