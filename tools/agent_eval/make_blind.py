"""가린 재검토용 묶음 파일을 만든다 (세션 도우미, WORKLOG 16절).

사용: python tools/agent_eval/make_blind.py <출력 폴더>
- 1차 테스트(처음·2차)와 테스트2(처음·2차·3차)의 답을 질문 묶음별로 섞고, 버전을 알려 주는 항목을 뺀다.
  남기는 것: 질문, 보여 준 답, 도구 호출과 결과. 빼는 것: 파일 이름, 숫자 검사 기록, 3차의 final·guard 등.
- 섞은 순서대로 두 개씩 한 검토자에게 준다. 같은 질문의 두 답이 한 짝이 되지 않는 첫 seed를 쓴다.
- <출력 폴더>/blind/set_t1.json, set_t2.json (검토자에게 주는 파일)과 <출력 폴더>/blind_key.json
  (묶음 id -> 실행 이름·질문 id, 검토자에게 주지 않음)을 쓴다.
답변 기록(outputs/cache/)이 있는 이 PC에서만 돈다. 같은 기록이면 몇 번을 돌려도 같은 파일이 나온다.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = {"t1": [("test_v1", "agent_questions_test.json"), ("test", "agent_questions_test.json")],
        "t2": [("t2_v1", "agent_questions_test2.json"), ("t2_v2", "agent_questions_test2.json"),
               ("t2_v3", "agent_questions_test2.json")]}
PREFIX = {"t1": "A", "t2": "B"}
SEED = 20261102

out_dir = Path(sys.argv[1])
(out_dir / "blind").mkdir(parents=True, exist_ok=True)
key = {}
for set_name, runs in SETS.items():
    base = []
    for run, qfile in runs:
        qs = json.loads((ROOT / "tests" / qfile).read_text(encoding="utf-8"))
        full = json.loads((ROOT / "outputs" / "cache" / f"agent_answers_{run}.json").read_text(encoding="utf-8"))
        assert [o["question"] for o in full] == [q["question"] for q in qs]
        for q, o in zip(qs, full):
            rec = {"question": o["question"], "answer": o["answer"],
                   "tool_calls": [{"name": t["name"], "args": t["args"], "result": t["result"]} for t in o["tool_calls"]]}
            base.append((rec, run, q["id"]))
    seed = SEED
    while True:
        recs = list(base)
        random.Random(seed).shuffle(recs)
        if all(recs[i][2] != recs[i + 1][2] for i in range(0, len(recs), 2)):
            break
        seed += 1
    out = []
    for i, (rec, run, qid) in enumerate(recs, 1):
        rid = f"{PREFIX[set_name]}{i:02d}"
        out.append({"rid": rid, **rec})
        key[rid] = {"run": run, "qid": qid}
    (out_dir / "blind" / f"set_{set_name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{set_name}: 답 {len(out)}개, seed {seed}")
(out_dir / "blind_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
