"""질의응답 에이전트 실행 (7번).

  python ask.py "중국으로 가는 ABS 수출의 2026년 1~3월 예측은?"
  python ask.py --eval     tests/agent_questions.json의 질문을 모두 돌려 outputs/agent_eval.json에 쓴다

설정: .env.example을 .env로 복사하고 OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL을 채운다 (.env는 커밋되지 않는다).
먼저 python run_all.py로 outputs/의 결과 파일을 만들어 두어야 한다.
"""
import argparse
import json
import sys
from datetime import datetime
from urllib.parse import urlparse

from tsa import config, qa_agent


def show(out):
    print(out["answer"])
    c = out["check"]
    print("\n---- 숫자 검사:", "통과" if c["ok"] else "실패", f"(숫자 {c['n_numbers']}개, 시도 {len(out['attempts'])}번)")
    for it in c["items"]:
        print(f"  {it['text']:>12}  <- {it['source'] or '도구 결과에 없음'}")
    if not c["ok"]:
        print("경고: 도구 결과에서 찾을 수 없는 숫자가 남아 있습니다:", ", ".join(c["unsupported"]))


def run_eval(llm):
    qs = json.loads((config.ROOT / "tests" / "agent_questions.json").read_text(encoding="utf-8"))
    rows, full = [], []
    for i, q in enumerate(qs, 1):
        out = qa_agent.ask(q["question"], llm)
        first = out["attempts"][0]["check"]
        called = sorted({t["name"] for t in out["tool_calls"]})
        rows.append({"id": q["id"], "question": q["question"], "expect_tools": q["expect_tools"],
                     "called_tools": called, "tool_choice_ok": set(q["expect_tools"]) <= set(called),
                     "first_try_ok": first["ok"], "final_ok": out["ok"], "attempts": len(out["attempts"]),
                     "n_numbers_first": first["n_numbers"], "n_numbers_final": out["check"]["n_numbers"],
                     "unsupported_first": first["unsupported"], "unsupported_final": out["check"]["unsupported"]})
        full.append(out)
        print(f"[{i}/{len(qs)}] {q['id']}: 첫 답변 {'통과' if first['ok'] else '실패'}, "
              f"최종 {'통과' if out['ok'] else '실패'}, 도구 {called}")
    A = {"created": datetime.now().isoformat(timespec="seconds"), "model": llm.model,
         "endpoint_host": urlparse(llm.base_url).netloc, "n_questions": len(rows),
         "first_try_ok": sum(r["first_try_ok"] for r in rows), "final_ok": sum(r["final_ok"] for r in rows),
         "tool_choice_ok": sum(r["tool_choice_ok"] for r in rows),
         "numbers_first_try": sum(r["n_numbers_first"] for r in rows),
         "numbers_final": sum(r["n_numbers_final"] for r in rows),
         "unsupported_first_try": sum(len(r["unsupported_first"]) for r in rows),
         "unsupported_final": sum(len(r["unsupported_final"]) for r in rows), "questions": rows}
    (config.OUT_DIR / "agent_eval.json").write_text(json.dumps(A, ensure_ascii=False, indent=2), encoding="utf-8")
    # 답변 전문에는 실제 물량이 들어갈 수 있어 커밋하지 않는 cache/에 둔다
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (config.CACHE_DIR / "agent_answers.json").write_text(json.dumps(full, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    print(f"숫자 검사 통과: 첫 답변 {A['first_try_ok']}/{A['n_questions']}, 최종 {A['final_ok']}/{A['n_questions']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    llm = qa_agent.OpenAICompatible()
    if args.eval:
        run_eval(llm)
        return
    if not args.question:
        ap.error("질문을 따옴표로 감싸 주거나 --eval을 쓰세요")
    out = qa_agent.ask(args.question, llm)
    show(out)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
