"""가린 재검토 결과를 실행별 평가 파일에 semantic_review_blind로 합친다 (세션 도우미, WORKLOG 16절).

사용: python tools/agent_eval/merge_blind.py <검토 결과.json> <blind_key.json>
- 검토 결과: review_blind_workflow.js가 돌려준 JSON (byId: 묶음 id -> 두 관점의 판정).
- blind_key.json: make_blind.py가 쓴 표 (묶음 id -> 실행 이름·질문 id). 이것으로 판정을 원래 실행에 되돌린다.
- 판정 규칙은 merge_review.py와 같다. 원래 뜻 검토와 통과 판정이 같은 질문 수도 찍는다.
- 검토 요약(summary_ko)도 평가 파일에 들어간다. 도구 결과의 숫자가 들어갈 수 있으니 커밋 전에 확인한다.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_review import show, summarize, verdict  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REVIEWER = ("가린 재검토: 독립 검토 에이전트(Claude) 2개 관점. 버전을 알려 주는 항목을 빼고 질문 묶음별로 섞어 새 id를 붙였고, "
            "읽을 파일을 묶음 파일과 tsa/qa_tools.py·metrics.py·config.py로 정함")

review = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["byId"]
key = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
runs = {}
for rid, k in key.items():
    runs.setdefault(k["run"], {})[k["qid"]] = {**verdict(review[rid]["faithful"], review[rid]["answering"]), "rid": rid}

for run, got in runs.items():
    path = ROOT / "outputs" / f"agent_eval_{run}.json"
    A = json.loads(path.read_text(encoding="utf-8"))
    by_id = {q["id"]: got[q["id"]] for q in A["questions"]}          # 질문 순서대로
    A["semantic_review_blind"] = summarize(A, by_id, REVIEWER)
    path.write_text(json.dumps(A, ensure_ascii=False, indent=2), encoding="utf-8")
    orig = A["semantic_review"]["by_id"]
    same = sum(orig[q]["faithful"] == v["faithful"] for q, v in by_id.items())
    print(f"== {run}: 가린 재검토 통과 {A['semantic_review_blind']['faithful']}/{len(by_id)}, "
          f"원래 검토 {A['semantic_review']['faithful']}/{len(by_id)}, 통과 판정이 같은 질문 {same}/{len(by_id)}")
    show(A["semantic_review_blind"])

# 판정 전문에는 답의 수치가 들어 있어 커밋하지 않는 cache/에 둔다
(ROOT / "outputs" / "cache" / "agent_review_blind.json").write_text(json.dumps({"byId": review, "key": key},
                                                                               ensure_ascii=False, indent=2),
                                                                    encoding="utf-8")
