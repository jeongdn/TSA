"""한 줄 재현: python run_all.py

  --stage val      검증 단계만 (frozen.json이 없으면 만들고, 있으면 다시 고른 결과가 같은지 확인하고
                   다르면 멈춘다. frozen.json과 data/ 지문은 덮어쓰지 않지만 outputs/val_* 파일은 다시 쓴다)
  --stage final    테스트 단계만 (frozen.json과, 같은 PC에서 먼저 돌린 val 단계의 outputs/val_* 파일 필요)
  --stage mint     7번 계층 조정 (val·final 단계의 예측 파일 필요). frozen_mint.json이 없으면 검증 구간에서 방법을
                   정해 쓰고 멈춘다. 있으면 다시 정한 결과가 같은지 확인한 뒤 테스트 구간을 채점한다
  --stage forecast 7번 마지막 자료 월 기준 앞으로 6개월 예측 (frozen.json, frozen_mint.json 필요)
  --stage report   그림과 RESULTS.md만 다시 만든다
  --stage all      (기본) val -> final -> mint -> forecast -> report
  --refresh        원자료를 다시 받는다 (기본은 data/raw 캐시 사용, 캐시가 없으면 자동으로 받음)
"""
import argparse
import json

from tsa import config, panel, stages


def same_choice(path, new, keys):
    old = json.loads(path.read_text(encoding="utf-8"))
    return all(json.dumps(old[k], sort_keys=True) == json.dumps(new[k], sort_keys=True) for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "final", "mint", "forecast", "report", "all"], default="all")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage != "report":
        D = stages.load(refresh=args.refresh)
        print(f"데이터: {D.check}")
        panel.intermittency(D.wide).to_csv(config.OUT_DIR / "intermittency.csv", index=False)
        D.info.to_csv(config.OUT_DIR / "series_info.csv", index=False)
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        D.wide.to_csv(config.PROCESSED_DIR / "panel_tonnes.csv")

    if args.stage in ("val", "all"):
        frozen = stages.stage_val(D)
        if config.FROZEN_PATH.exists():
            same = same_choice(config.FROZEN_PATH, frozen, ["series", "lgbm", "ensemble_weights", "conformal"])
            print(f"[val] 동결본과 다시 고른 결과가 같은가: {same}")
            if not same:
                # 다르면 outputs/val_*(온라인 보정에 쓰는 검증 구간 예측)가 동결본과 다른 설정에서 나온 것이 되므로 멈춘다
                raise SystemExit("[val] 다시 고른 결과가 frozen.json과 다릅니다. 원자료나 라이브러리 버전이 "
                                 "달라졌을 수 있습니다. 테스트 단계는 돌리지 않습니다 (커밋된 RESULTS.md가 기록입니다).")
        else:
            stages._json(frozen, config.FROZEN_PATH)
            print("[val] frozen.json을 썼습니다. 커밋하고 'freeze' 태그를 단 뒤 --stage final을 실행하세요.")
            if args.stage == "all":
                return

    if args.stage in ("final", "all"):
        stages.stage_final(D)

    if args.stage in ("mint", "all"):
        fm = stages.stage_mint_val(D)
        if config.FROZEN_MINT_PATH.exists():
            same = same_choice(config.FROZEN_MINT_PATH, fm, ["candidate", "default", "adopted", "gate"])
            print(f"[mint] 동결본과 다시 정한 결과가 같은가: {same}")
            if not same:
                raise SystemExit("[mint] 다시 정한 결과가 frozen_mint.json과 다릅니다. 테스트 채점은 하지 않습니다.")
        else:
            stages._json(fm, config.FROZEN_MINT_PATH)
            print("[mint] frozen_mint.json을 썼습니다. 커밋하고 'freeze-mint' 태그를 단 뒤 --stage mint를 다시 "
                  "실행하세요.")
            return
        stages.stage_mint_final(D)

    if args.stage in ("forecast", "all"):
        stages.stage_forecast(D)

    if args.stage in ("report", "all"):
        from tsa import report
        report.build()


if __name__ == "__main__":      # Windows에서 joblib 병렬 실행에 필요
    main()
