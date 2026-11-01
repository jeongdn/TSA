"""뜻 검토 결과(워크플로 반환값 JSON)를 outputs/agent_eval.json에 합친다 (세션 도우미).

사용: python merge_review.py <review.json> <프로젝트 폴더> [평가 파일 이름 (기본 agent_eval.json)]
판정: 통과 = 두 검토자 모두 '모든 주장이 도구 결과와 맞음' + '질문에 답함' 검토자가 답했다고 봄.
"""
import json
import sys
from pathlib import Path

review = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["byId"]
root = Path(sys.argv[2])
name = sys.argv[3] if len(sys.argv) > 3 else "agent_eval.json"
a_path = root / "outputs" / name
A = json.loads(a_path.read_text(encoding="utf-8"))

by_id, n_ok, n_passed_bad = {}, 0, 0
for q in A["questions"]:
    f, a = review[q["id"]]["faithful"], review[q["id"]]["answering"]
    claims_ok = f["faithful"] and a["faithful"]
    ok = claims_ok and a["answers_question"]
    if ok:
        summary = "문제 없음"
    elif not claims_ok:
        summary = f["summary_ko"] if not f["faithful"] else a["summary_ko"]
    else:
        summary = a["summary_ko"]
    by_id[q["id"]] = {"faithful": ok, "claims_faithful": claims_ok, "answers_question": a["answers_question"],
                      "right_tool_and_args": a["right_tool_and_args"], "korean_only": f["korean_only"] and a["korean_only"],
                      "lenses_agree_on_claims": f["faithful"] == a["faithful"], "summary_ko": summary}
    n_ok += ok
    n_passed_bad += q.get("delivered_ok", q["final_ok"]) and not ok    # 보여 준 답이 숫자 검사는 통과했는데 뜻 검토에서 걸림

A["semantic_review"] = {
    "reviewer": "독립 검토 에이전트(Claude) 2개 관점 x 답변 10개: 주장별 사실 확인, 질문에 답했는지·도구 사용",
    "definition": "통과 = 두 검토자 모두 모든 주장이 그 답변의 도구 결과와 맞다고 보고, 질문에도 답했다고 본 경우",
    "faithful": n_ok, "passed_numbers_but_unfaithful": n_passed_bad,
    "claims_faithful": sum(v["claims_faithful"] for v in by_id.values()),
    "answers_question": sum(v["answers_question"] for v in by_id.values()),
    "right_tool_and_args": sum(v["right_tool_and_args"] for v in by_id.values()),
    "korean_only": sum(v["korean_only"] for v in by_id.values()),
    "by_id": by_id,
}
a_path.write_text(json.dumps(A, ensure_ascii=False, indent=2), encoding="utf-8")
(root / "outputs" / "cache" / name.replace("agent_eval", "agent_review")).write_text(json.dumps(review, ensure_ascii=False, indent=2),
                                                             encoding="utf-8")
for k, v in by_id.items():
    print(f"{k:12s} ok={v['faithful']!s:5s} claims={v['claims_faithful']!s:5s} answers={v['answers_question']!s:5s} "
          f"tool={v['right_tool_and_args']!s:5s} ko={v['korean_only']!s:5s} agree={v['lenses_agree_on_claims']!s:5s} "
          f"| {v['summary_ko']}")
s = A["semantic_review"]
print(f"통과 {s['faithful']}/10, 주장 맞음 {s['claims_faithful']}, 질문에 답함 {s['answers_question']}, "
      f"도구·인자 맞음 {s['right_tool_and_args']}, 한국어만 {s['korean_only']}, 숫자 통과인데 뜻 문제 {s['passed_numbers_but_unfaithful']}")
