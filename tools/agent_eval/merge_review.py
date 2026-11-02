"""뜻 검토 결과(워크플로 반환값 JSON)를 outputs/agent_eval*.json에 합친다 (세션 도우미).

사용: python merge_review.py <review.json> <프로젝트 폴더> [평가 파일 이름 (기본 agent_eval.json)]
판정: 통과 = 두 검토자 모두 '모든 주장이 도구 결과와 맞음' + '질문에 답했는지' 관점 검토자가 답했다고 봄.
가린 재검토(merge_blind.py)도 같은 판정 함수(verdict, summarize)를 쓴다.
검토 요약(summary_ko)은 커밋하는 평가 파일에 들어간다. 요약에는 도구 결과의 숫자(작년 같은 달 대비 증감률 등)가
들어갈 수 있으니 커밋 전에 확인한다.
"""
import json
import sys
from pathlib import Path

DEFINITION = ("통과 = 두 검토자 모두 모든 주장이 그 답변의 도구 결과와 맞다고 보고, '질문에 답했는지' 관점 검토자가 "
              "질문에 답했다고 본 경우")


def verdict(f, a):
    """한 답변에 대한 두 관점(f: 주장별 사실 확인, a: 질문에 답했는지)의 판정을 합친다."""
    claims_ok = f["faithful"] and a["faithful"]
    ok = claims_ok and a["answers_question"]
    if ok:
        summary = "문제 없음"
    elif not claims_ok:
        summary = f["summary_ko"] if not f["faithful"] else a["summary_ko"]
    else:
        summary = a["summary_ko"]
    return {"faithful": ok, "claims_faithful": claims_ok, "answers_question": a["answers_question"],
            "right_tool_and_args": a["right_tool_and_args"], "korean_only": f["korean_only"] and a["korean_only"],
            "lenses_agree_on_claims": f["faithful"] == a["faithful"], "summary_ko": summary}


def summarize(A, by_id, reviewer):
    """평가 파일 A의 질문들에 대한 판정(by_id: 질문 id -> verdict)을 센다."""
    passed_bad = sum(q.get("delivered_ok", q["final_ok"]) and not by_id[q["id"]]["faithful"] for q in A["questions"])
    return {"reviewer": reviewer, "definition": DEFINITION,
            "faithful": sum(v["faithful"] for v in by_id.values()),
            "passed_numbers_but_unfaithful": passed_bad,     # 보여 준 답이 숫자 검사는 통과했는데 뜻 검토에서 걸림
            "claims_faithful": sum(v["claims_faithful"] for v in by_id.values()),
            "answers_question": sum(v["answers_question"] for v in by_id.values()),
            "right_tool_and_args": sum(v["right_tool_and_args"] for v in by_id.values()),
            "korean_only": sum(v["korean_only"] for v in by_id.values()),
            "by_id": by_id}


def show(s):
    for k, v in s["by_id"].items():
        print(f"{k:12s} ok={v['faithful']!s:5s} claims={v['claims_faithful']!s:5s} answers={v['answers_question']!s:5s} "
              f"tool={v['right_tool_and_args']!s:5s} ko={v['korean_only']!s:5s} agree={v['lenses_agree_on_claims']!s:5s} "
              f"| {v['summary_ko']}")
    n = len(s["by_id"])
    print(f"통과 {s['faithful']}/{n}, 주장 맞음 {s['claims_faithful']}, 질문에 답함 {s['answers_question']}, "
          f"도구·인자 맞음 {s['right_tool_and_args']}, 한국어만 {s['korean_only']}, "
          f"숫자 통과인데 뜻 문제 {s['passed_numbers_but_unfaithful']}")


if __name__ == "__main__":
    review = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["byId"]
    root = Path(sys.argv[2])
    name = sys.argv[3] if len(sys.argv) > 3 else "agent_eval.json"
    a_path = root / "outputs" / name
    A = json.loads(a_path.read_text(encoding="utf-8"))
    by_id = {q["id"]: verdict(review[q["id"]]["faithful"], review[q["id"]]["answering"]) for q in A["questions"]}
    A["semantic_review"] = summarize(A, by_id, "독립 검토 에이전트(Claude) 2개 관점 x 답변 10개: 주장별 사실 확인, "
                                               "질문에 답했는지·도구 사용")
    a_path.write_text(json.dumps(A, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "outputs" / "cache" / name.replace("agent_eval", "agent_review")).write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    show(A["semantic_review"])
