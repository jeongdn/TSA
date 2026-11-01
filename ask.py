"""질의응답 에이전트 실행 (7번).

  python ask.py "중국으로 가는 ABS 수출의 2026년 1~3월 예측은?"
  python ask.py --eval                       tests/agent_questions.json(개발용 질문)을 모두 돌려 outputs/agent_eval.json에 쓴다
  python ask.py --eval --questions tests/agent_questions_test.json --name test
                                             다른 질문 파일로 돌려 outputs/agent_eval_test.json에 쓴다

설정: .env.example을 .env로 복사하고 OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL을 채운다 (.env는 커밋되지 않는다).
먼저 python run_all.py로 outputs/의 결과 파일을 만들어 두어야 한다.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from tsa import config, numcheck, qa_agent


def show(out):
    print(out["answer"])
    c = out["check"]
    print("\n---- 숫자 검사:", "통과" if c["ok"] else "실패", f"(숫자 {c['n_numbers']}개, 시도 {len(out['attempts'])}번)")
    for it in c["items"]:
        print(f"  {it['text']:>12}  <- {it['source'] or '도구 결과에 없음'}")
    for x in c.get("issues", []):
        print("  문제:", x)
    if not c["ok"]:
        print("경고: 모델의 마지막 답이 검사를 통과하지 못했습니다:", ", ".join(c["unsupported"] + c.get("issues", [])))
        print(f"      그래서 걸린 문장 {out['redacted']}개를 빼고 보여 줬습니다.")


def _git_head():
    """평가에 쓴 에이전트 코드의 커밋. 에이전트 파일에 커밋 안 된 변경이 있으면 '+dirty'."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=config.ROOT, capture_output=True, text=True,  # noqa: E731
                                        check=True).stdout.strip()
        dirty = run("status", "--porcelain", "--", "tsa/qa_agent.py", "tsa/qa_tools.py", "tsa/numcheck.py", "ask.py")
        return run("rev-parse", "--short", "HEAD") + ("+dirty" if dirty else "")
    except Exception:
        return "n/a"


def run_eval(llm, questions, name):
    qs = json.loads(Path(questions).read_text(encoding="utf-8"))
    suffix = f"_{name}" if name else ""
    commit = _git_head()                     # 실행을 시작할 때의 에이전트 코드
    rows, full = [], []
    for i, q in enumerate(qs, 1):
        n0 = len(llm.prompt_tokens)
        out = qa_agent.ask(q["question"], llm)
        first = out["attempts"][0]["check"]
        called = sorted({t["name"] for t in out["tool_calls"]})
        used = [t for t in llm.prompt_tokens[n0:] if t]
        rows.append({"id": q["id"], "question": q["question"], "expect_tools": q["expect_tools"],
                     "called_tools": called, "tool_choice_ok": set(q["expect_tools"]) <= set(called),
                     "first_try_ok": first["ok"], "final_ok": out["ok"], "attempts": len(out["attempts"]),
                     "n_numbers_first": first["n_numbers"], "n_numbers_final": out["check"]["n_numbers"],
                     "unsupported_first": first["unsupported"], "unsupported_final": out["check"]["unsupported"],
                     "issues_first": first.get("issues", []), "issues_final": out["check"].get("issues", []),
                     "redacted": out.get("redacted", 0),
                     "delivered_ok": numcheck.check(out["answer"], out["tool_calls"], q["question"])["ok"],
                     "max_prompt_tokens": max(used, default=None)})
        full.append(out)
        print(f"[{i}/{len(qs)}] {q['id']}: 첫 답변 {'통과' if first['ok'] else '실패'}, "
              f"최종 {'통과' if out['ok'] else '실패'}, 도구 {called}")     # 질문 문장은 찍지 않는다
    A = {"created": datetime.now().isoformat(timespec="seconds"), "model": llm.model,
         "endpoint_host": urlparse(llm.base_url).netloc, "questions_file": Path(questions).as_posix(),
         "agent_commit": commit, "n_questions": len(rows),
         "first_try_ok": sum(r["first_try_ok"] for r in rows), "final_ok": sum(r["final_ok"] for r in rows),
         "tool_choice_ok": sum(r["tool_choice_ok"] for r in rows),
         "numbers_first_try": sum(r["n_numbers_first"] for r in rows),
         "numbers_final": sum(r["n_numbers_final"] for r in rows),
         "unsupported_first_try": sum(len(r["unsupported_first"]) for r in rows),
         "unsupported_final": sum(len(r["unsupported_final"]) for r in rows),
         "issues_first_try": sum(len(r["issues_first"]) for r in rows),
         "issues_final": sum(len(r["issues_final"]) for r in rows),
         "redacted_answers": sum(r["redacted"] > 0 for r in rows), "delivered_ok": sum(r["delivered_ok"] for r in rows),
         "max_prompt_tokens": max((r["max_prompt_tokens"] for r in rows if r["max_prompt_tokens"]), default=None),
         "questions": rows}
    (config.OUT_DIR / f"agent_eval{suffix}.json").write_text(json.dumps(A, ensure_ascii=False, indent=2),
                                                             encoding="utf-8")
    # 답변 전문에는 실제 물량이 들어갈 수 있어 커밋하지 않는 cache/에 둔다
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (config.CACHE_DIR / f"agent_answers{suffix}.json").write_text(json.dumps(full, ensure_ascii=False, indent=2),
                                                                  encoding="utf-8")
    print(f"숫자 검사 통과: 첫 답변 {A['first_try_ok']}/{A['n_questions']}, 최종 {A['final_ok']}/{A['n_questions']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--questions", default=str(config.ROOT / "tests" / "agent_questions.json"))
    ap.add_argument("--name", default="", help="결과 파일 이름 뒤에 붙일 이름 (예: test -> agent_eval_test.json)")
    args = ap.parse_args()
    llm = qa_agent.OpenAICompatible()
    if args.eval:
        run_eval(llm, args.questions, args.name)
        return
    if not args.question:
        ap.error("질문을 따옴표로 감싸 주거나 --eval을 쓰세요")
    out = qa_agent.ask(args.question, llm)
    show(out)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
