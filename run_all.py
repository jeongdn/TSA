"""한 줄 재현: python run_all.py

  --stage val    검증 단계만 (frozen.json이 없으면 만들고, 있으면 다시 고른 결과가 같은지만 확인)
  --stage final  테스트 단계만 (frozen.json 필요)
  --stage report 그림과 RESULTS.md만 다시 만든다
  --stage all    (기본) val -> final -> report
  --refresh      원자료를 다시 받는다 (기본은 data/raw 캐시 사용, 캐시가 없으면 자동으로 받음)
"""
import argparse
import json

from tsa import config, panel, stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "final", "report", "all"], default="all")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in ("val", "final", "all"):
        D = stages.load(refresh=args.refresh)
        print(f"데이터: {D.check}")
        panel.intermittency(D.wide).to_csv(config.OUT_DIR / "intermittency.csv", index=False)
        D.info.to_csv(config.OUT_DIR / "series_info.csv", index=False)
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        D.wide.to_csv(config.PROCESSED_DIR / "panel_tonnes.csv")

    if args.stage in ("val", "all"):
        frozen = stages.stage_val(D)
        if config.FROZEN_PATH.exists():
            old = json.loads(config.FROZEN_PATH.read_text(encoding="utf-8"))
            keys = ["series", "lgbm", "ensemble_weights", "conformal"]
            same = all(json.dumps(old[k], sort_keys=True) == json.dumps(frozen[k], sort_keys=True) for k in keys)
            print(f"[val] 동결본과 다시 고른 결과가 같은가: {same}")
            if not same:
                print("[val] 경고: 다시 고른 결과가 동결본과 다릅니다. final은 동결본(frozen.json)을 그대로 씁니다.")
        else:
            stages._json(frozen, config.FROZEN_PATH)
            print("[val] frozen.json을 썼습니다. 커밋하고 'freeze' 태그를 단 뒤 --stage final을 실행하세요.")
            if args.stage == "all":
                return

    if args.stage in ("final", "all"):
        stages.stage_final(D)

    if args.stage in ("report", "all"):
        from tsa import report
        report.build()


if __name__ == "__main__":      # Windows에서 joblib 병렬 실행에 필요
    main()
