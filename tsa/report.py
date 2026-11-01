"""그림과 RESULTS.md를 만든다. 숫자는 모두 outputs/의 실행 결과 파일에서 읽는다 (손으로 적은 숫자 없음)."""
import json
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
IV_NAME = {"raw": "분위수 그대로", "static": "CQR (검증 구간 보정)", "online": "CQR (온라인 보정)"}


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
    ax.legend(loc="upper left", fontsize=8.5)


def fig_coverage_by_h(R, ax):
    hs = range(1, config.H + 1)
    ax.axhline(80, color=INK2, lw=1, ls="--")
    ax.text(0.75, 80.6, "목표 80%", fontsize=8.5, color=INK2)
    ends = []
    for kind in ["raw", "static", "online"]:
        v = [100 * R["intervals"][kind]["by_h"][str(h)]["coverage"] for h in hs]
        ax.plot(hs, v, color=IV_COLORS[kind], lw=2, marker="o", ms=5, label=IV_NAME[kind])
        ends.append((IV_NAME[kind].split(" (")[0] if kind == "raw" else kind, v[-1]))
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
        ax.plot(tx, d["mid"], color=COLORS["LGBM"], lw=2, label="LightGBM 1개월 앞")
        ax.set_title(f"{s} (톤/월)", loc="left", fontsize=10.5)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].legend(loc="upper left", fontsize=8.5)
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
    return f"{100 * x:+.{d}f}%p"


def _git(ref):
    try:
        return subprocess.run(["git", "rev-parse", "--short", ref], cwd=config.ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "n/a"


def results_md(R, V, F):
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
    w(f"- 동결: frozen.json 작성 {F['created']}, 태그 `freeze` = `{_git('freeze')}`. 테스트 채점은 그 뒤 한 번.\n")

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
    w("P는 '블록으로 다시 뽑은 테스트 기간 가운데 LightGBM의 WAPE가 더 낮았던 비율'이다. "
      "미래에 더 나을 확률이 아니다.\n")

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
    qtxt = ", ".join(f"h{h}: ({v[0]:+.3f}, {v[1]:+.3f})" for h, v in F["conformal"]["Q"].items())
    w(f"- CQR 보정폭 (Q_lo, Q_hi), 규모 대비: {qtxt}. 양수면 구간을 넓히고 음수면 좁힌다.")
    w("\n그림: `outputs/figures/` (fig_portfolio, fig_origin_diff, fig_selection_bias, fig_fan)")
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
             "(블록 부트스트랩, 블록 6, 10,000회)")
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
    figures(R)
    (config.ROOT / "RESULTS.md").write_text(results_md(R, V, F), encoding="utf-8")
    for doc in ("README.md", "SUMMARY.md"):
        inject(config.ROOT / doc, "summary_table", summary_table(R))
    print("RESULTS.md와 그림을 만들었습니다.")
