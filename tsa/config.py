"""실험 설정을 한곳에 모은다.

기간·분할·시드는 테스트 구간 결과를 보기 전에 정했다 (WORKLOG.md 참고).
월은 모두 pandas Period('M')로 다룬다. 문자열로 정렬하다 생기는 실수를 막기 위해서다.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
CACHE_DIR = OUT_DIR / "cache"          # 진단용 대용량 예측 (커밋 안 함)
FROZEN_PATH = ROOT / "frozen.json"     # 검증 단계에서 고정한 선택 (테스트 전에 커밋)
FROZEN_MINT_PATH = ROOT / "frozen_mint.json"   # 7번 계층 조정 방법 선택 (테스트 전에 커밋)

# ---- 데이터 ----
REPORTER = 410                          # UN Comtrade 국가 코드: 대한민국
PRODUCTS = {"390330": "ABS", "390740": "PC"}   # HS 6단위 코드 -> 이름
START = pd.Period("2015-01", "M")
END = pd.Period("2025-12", "M")         # E: Comtrade에 있는 마지막 달 (2026-11-01 확인)

# ---- 예측 문제 ----
H = 6                                   # 1~6개월 앞을 예측
TOP_K = 8                               # 제품별 상위 상대국 수 (+ 기타)
SELECT_MONTHS = 24                      # 상위국을 고를 때 보는 기간 (개월)

# ---- 분할 (기준 시점 = 실제값을 아는 마지막 달) ----
N_VAL = 12
N_TEST = 24
# 테스트: 마지막 기준 시점은 END - H (그래야 1~6개월 앞 실제값이 모두 있음)
TEST_ORIGINS = pd.period_range(END - H - N_TEST + 1, END - H, freq="M")  # 2023-07 ~ 2025-06
CUTOFF = TEST_ORIGINS[0]                                                # C = 2023-07
# 검증: 모든 예측 대상 월이 C 이하가 되도록 C - H 에서 끝낸다
VAL_ORIGINS = pd.period_range(CUTOFF - H - N_VAL + 1, CUTOFF - H, freq="M")  # 2022-02 ~ 2023-01
SELECT_END = VAL_ORIGINS[0]             # 상위국 선정에 쓰는 마지막 달 (2022-02)

# ---- 재현성 ----
SEED = 2026
N_CONFIGS = 30                          # LightGBM 설정 랜덤 탐색 개수
LGB_THREADS = 4                         # 스레드 수를 고정해야 결과가 똑같이 나온다
N_JOBS = 4                              # 설정 탐색을 프로세스 4개로 나눠 돌림

# ---- 부트스트랩 (검증 단계에서 미리 정함) ----
BOOT_BLOCK = 6                          # 블록 길이 = 최대 예측 시점 (이웃 기준 시점끼리 목표 월이 겹치므로)
BOOT_B = 10_000

# ---- 확률 예측 ----
ALPHA_LO, ALPHA_HI = 0.1, 0.9           # P10, P90
