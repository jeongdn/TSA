"""UN Comtrade 공개 API(키 필요 없음)에서 한국의 ABS·PC 월별 수출을 받는다.

2026-11-01에 직접 확인한 제약:
- 공개 preview 엔드포인트는 호출 한 번에 기간(YYYYMM)을 하나만 받는다.
- 결과가 500행을 넘으면 경고 없이 잘린다. 그래서 500행 이상이면 멈춘다.
- partnerCode를 빼면 모든 상대국 행과 World(코드 0) 행이 함께 온다. 한 달에 약 110~130행.
"""
import hashlib
import json
import time
import urllib.error
import urllib.request

import pandas as pd

from . import config

BASE = "https://comtradeapi.un.org/public/v1"
CAP = 500                                   # preview 행 상한 (넘으면 조용히 잘림)
RAW = config.RAW_DIR / "comtrade"


def _get_json(url, tries=3):
    """GET 요청. 서버 오류(5xx)나 시간 초과는 5초, 10초 쉬고 다시 시도한다."""
    req = urllib.request.Request(url, headers={"User-Agent": "tsa-practice/0.1 (urllib)"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise SystemExit("HTTP 429(요청 한도 초과): 잠시 뒤 다시 실행하세요.")
            if e.code < 500 or k == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if k == tries - 1:
                raise
        time.sleep(5 * 2 ** k)


def fetch_month(period, refresh=False):
    """한 달치(두 HS 코드 x 전체 상대국)를 받아 캐시에 저장하고 행 목록을 돌려준다."""
    ym = period.strftime("%Y%m")
    path = RAW / f"X_{ym}.json"
    if path.exists() and not refresh:       # 이미 받은 달은 다시 요청하지 않는다
        return json.loads(path.read_text(encoding="utf-8"))["data"]
    url = (f"{BASE}/preview/C/M/HS?reporterCode={config.REPORTER}&period={ym}"
           f"&cmdCode={','.join(config.PRODUCTS)}&flowCode=X&includeDesc=true")
    body = _get_json(url)
    rows = body.get("data") or []
    if body.get("error"):
        raise RuntimeError(f"{ym}: API 오류 {body['error']!r}")
    if len(rows) >= CAP:
        raise RuntimeError(f"{ym}: {len(rows)}행 - 500행 상한에 걸려 잘렸을 수 있음")
    if not rows:                            # 빈 응답 = 아직 공개 전. 캐시하지 않는다
        raise RuntimeError(f"{ym}: 빈 응답 (아직 공개되지 않은 달일 수 있음)")
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    time.sleep(2.0)                         # 서버 부담을 줄이려고 호출 사이에 2초 쉰다
    return rows


def fetch_all(start=config.START, end=config.END, refresh=False):
    """start~end 모든 달을 받는다 (캐시가 있으면 요청 없이 읽기만 함)."""
    periods = pd.period_range(start, end, freq="M")
    rows = []
    for i, p in enumerate(periods, 1):
        rows.extend(fetch_month(p, refresh))
        if i % 12 == 0 or i == len(periods):
            print(f"  Comtrade {p} 까지 {i}/{len(periods)}개월")
    return rows


def to_frame(rows):
    """API 행 목록 -> (월, HS, 상대국, kg, USD) 표. 집계 행만 남긴다."""
    df = pd.DataFrame(rows)
    # 2차 상대국·세관·운송수단으로 쪼갠 행이 섞이면 같은 물량을 두 번 세게 된다
    df = df[(df.flowCode == "X") & (df.partner2Code == 0)
            & (df.customsCode == "C00") & (df.motCode == 0)]
    # 목표는 순중량(kg). 없으면 단위가 kg(코드 8)인 qty로 대신한다
    kg = df["netWgt"].where(df["netWgt"].notna(), df["qty"].where(df["qtyUnitCode"] == 8))
    out = pd.DataFrame({
        "period": pd.to_datetime(df["period"].astype(str), format="%Y%m").dt.to_period("M"),
        "hs": df["cmdCode"].astype(str),
        "partner": df["partnerCode"].astype(int),
        "iso": df["partnerISO"].astype(str),
        "name": df["partnerDesc"].astype(str),
        "kg": kg.astype(float),
        "usd": df["primaryValue"].astype(float),
        "estimated": (df["isNetWgtEstimated"].fillna(False) | df["isQtyEstimated"].fillna(False)).astype(bool),
    })
    return out.reset_index(drop=True)


def check(df, start=config.START, end=config.END):
    """자료 무결성 검사. 하나라도 어긋나면 AssertionError로 멈춘다."""
    months = pd.period_range(start, end, freq="M")
    assert not df.duplicated(["period", "hs", "partner"]).any(), "(월, HS, 상대국) 중복 행"
    assert df["kg"].notna().all() and (df["kg"] >= 0).all(), "kg 결측 또는 음수"
    world = df[df.partner == 0].groupby(["hs", "period"])["kg"].sum()
    parts = df[df.partner != 0].groupby(["hs", "period"])["kg"].sum()
    for hs in config.PRODUCTS:
        missing = months.difference(world.loc[hs].index)
        assert len(missing) == 0, f"{hs}: World 행이 없는 달 {list(missing)}"
    rel = ((world - parts.reindex(world.index).fillna(0)).abs() / world).max()
    assert rel < 1e-6, f"World와 상대국 합계가 다름 (최대 상대 차이 {rel:.2e})"
    return {
        "rows": int(len(df)),
        "months": int(df["period"].nunique()),
        "max_rel_diff_world_vs_partners": float(rel),
        "estimated_rows": int(df["estimated"].sum()),
        "partners_per_month_min": int(df[df.partner != 0].groupby(["hs", "period"]).size().min()),
    }


def manifest():
    """받은 원자료 파일마다 행 수와 SHA-256을 기록한다 (다시 받았을 때 바뀌었는지 확인용)."""
    files = {}
    for path in sorted(RAW.glob("X_*.json")):
        b = path.read_bytes()
        files[path.name] = {"sha256": hashlib.sha256(b).hexdigest(),
                            "rows": len(json.loads(b)["data"])}
    return files
