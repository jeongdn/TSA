"""FRED(키 필요 없는 CSV)에서 외부 지표 두 개를 받는다.

- EXKOUS: 원/달러 환율 월평균 (일별 값의 평균). T월 값은 T+1월 1일쯤 공개.
- MCOILBRENTEU: 브렌트유 월평균 (EIA). T월 값은 T+1월 7일쯤 공개.
T월 무역 통계는 T+1월 중순(관세청 확정치)에 나온다고 보고, 기준 시점 T에서는 T월까지의 값만 쓴다.
"""
import hashlib

import pandas as pd

from . import config

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
SERIES = {"fx": "EXKOUS", "brent": "MCOILBRENTEU"}
RAW = config.RAW_DIR / "fred"


def fetch(refresh=False):
    """두 지표를 받아 월(Period) 인덱스의 표로 돌려준다."""
    RAW.mkdir(parents=True, exist_ok=True)
    cols = {}
    for name, sid in SERIES.items():
        path = RAW / f"{sid}.csv"
        if refresh or not path.exists():
            pd.read_csv(URL.format(sid)).to_csv(path, index=False)
        s = pd.read_csv(path, na_values=".")
        # FRED 월별 날짜는 그달 1일로 찍힌다 (2024-05-01 = 5월 평균) -> 월 Period로 바꾼다
        idx = pd.to_datetime(s["observation_date"]).dt.to_period("M")
        cols[name] = pd.Series(s[sid].astype(float).values, index=idx)
    ext = pd.DataFrame(cols)
    months = pd.period_range(config.START - 12, config.END, freq="M")   # 12개월 변화율용으로 1년 더
    ext = ext.reindex(months)
    assert ext.notna().all().all(), "외부 지표에 빠진 달이 있음"
    return ext


def manifest():
    return {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(RAW.glob("*.csv"))}
