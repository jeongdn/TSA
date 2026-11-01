"""수집한 행을 18개 시리즈(제품별 상위 8개국 + 기타)의 월별 물량 표로 만든다."""
import numpy as np
import pandas as pd

from . import config


def build(df):
    """df(월, HS, 상대국, kg) -> wide(행=월, 열=시리즈, 단위 톤), info(시리즈 설명)."""
    months = pd.period_range(config.START, config.END, freq="M")
    # 상위국 선정 기간: 첫 검증 기준 시점(2022-02)까지 24개월. 그 뒤 값은 보지 않는다
    sel = pd.period_range(config.SELECT_END - (config.SELECT_MONTHS - 1), config.SELECT_END, freq="M")
    cols, info = {}, []
    for hs, prod in config.PRODUCTS.items():
        d = df[df.hs == hs]
        world = d[d.partner == 0].set_index("period")["kg"].reindex(months)
        assert world.notna().all(), f"{prod}: World 행이 빠진 달이 있음"
        # 상대국별 월 물량. 행이 없는 달은 그 나라로 수출이 없었다는 뜻이라 0으로 채운다
        part = (d[d.partner != 0]
                .pivot_table(index="period", columns="partner", values="kg", aggfunc="sum")
                .reindex(months).fillna(0.0))
        names = d.drop_duplicates("partner").set_index("partner")
        sel_total = part.loc[sel].sum().sort_values(ascending=False)
        top = list(sel_total.index[: config.TOP_K])
        world_sel = world.loc[sel].sum()
        for p in top:
            iso = names.loc[p, "iso"]
            cols[f"{prod}-{iso}"] = part[p] / 1000.0
            info.append({"series": f"{prod}-{iso}", "product": prod, "country": iso,
                         "name": names.loc[p, "name"], "share_select": sel_total[p] / world_sel})
        # 기타 = World - 상위 8개국. World가 상대국 합계와 정확히 같아서 나머지 나라 합과 같다
        other = (world - part[top].sum(axis=1)) / 1000.0
        assert (other > -1e-6).all(), f"{prod}: 기타가 음수인 달이 있음"
        cols[f"{prod}-OTH"] = other.clip(lower=0.0)
        info.append({"series": f"{prod}-OTH", "product": prod, "country": "OTH",
                     "name": "Others (rest of world)",
                     "share_select": 1.0 - sel_total[top].sum() / world_sel})
    wide = pd.DataFrame(cols, index=months)
    return wide, pd.DataFrame(info)


def intermittency(wide, end=config.SELECT_END):
    """Syntetos-Boylan 분류: ADI(수요 간 평균 간격)와 CV²(0 아닌 값의 변동계수 제곱).
    ADI > 1.32면 간헐적 -> Croston/TSB 후보. 선정 기간과 같은 시점까지의 데이터만 쓴다."""
    w = wide.loc[:end]
    out = []
    for c in w.columns:
        y = w[c].to_numpy()
        nz = y[y > 0]
        adi = len(y) / max(len(nz), 1)
        cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) > 1 else np.nan
        if adi <= 1.32:
            kind = "smooth" if cv2 <= 0.49 else "erratic"
        else:
            kind = "intermittent" if cv2 <= 0.49 else "lumpy"
        out.append({"series": c, "zero_months": int((y == 0).sum()), "ADI": adi, "CV2": cv2, "class": kind})
    return pd.DataFrame(out)
