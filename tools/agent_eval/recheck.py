"""기록에 남은 답을 지금의 숫자 검사로 다시 본다 (세션 도우미, WORKLOG 16절의 숫자).

사용: python tools/agent_eval/recheck.py
- 3차: 보호 장치가 답을 바꾸면 기록에는 숫자 검사 통과로 남는다. 그래서 모델이 고른 문장(final_answer의 id)을
  도구 기록에서 다시 만들어, 보호 장치 전의 답으로 검사한다.
- 처음 에이전트: 그때의 숫자 검사에는 기간 검사가 없었다. 보여 준 답을 지금의 검사로 다시 본다.
답변 기록(outputs/cache/agent_answers_*.json)은 커밋하지 않으므로 작성자 컴퓨터에서만 돈다.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tsa import numcheck, qa_agent3, qa_facts  # noqa: E402

QUESTIONS = {"dev": "agent_questions.json", "t1": "agent_questions_test.json", "t2": "agent_questions_test2.json"}
V3_RUNS = [("v3dev4", "dev"), ("v3t1d", "t1"), ("t2_v3", "t2")]       # 3차 동결본으로 돌린 실행
FIRST_RUNS = [("test_v1", "t1"), ("t2_v1", "t2")]                    # 처음 에이전트로 돌린 실행


def load(name):
    return json.loads((ROOT / "outputs" / "cache" / f"agent_answers_{name}.json").read_text(encoding="utf-8"))


def questions(key):
    return json.loads((ROOT / "tests" / QUESTIONS[key]).read_text(encoding="utf-8"))


def facts_from_log(log):
    """qa_agent3.view와 같은 id('k:키')로 도구 기록에서 사실 문장을 다시 만든다."""
    facts = {}
    for k, t in enumerate(log, 1):
        if "error" not in t["result"]:
            for key, text in qa_facts.build(t["name"], t["result"]):
                facts[f"{k}:{key}"] = text
    return facts


def model_pick(out):
    """보호 장치 전의 답: 모델이 고른 문장을 그대로 이은 것."""
    if out["final"] is None:
        return qa_agent3.EMPTY, None
    f = out["final"]
    return qa_agent3.render(f["answerable"], f["fact_ids"], facts_from_log(out["tool_calls"])), f["answerable"]


for name, key in V3_RUNS:
    ok = 0
    for q, out in zip(questions(key), load(name)):
        answer, answerable = model_pick(out)
        c = qa_agent3._check(answer, out["tool_calls"], q["question"], answerable)
        ok += c["ok"]
        if not c["ok"]:
            print(f"  {name} {q['id']}: 모델이 고른 답이 걸림 ({', '.join(c['unsupported'] + c['issues'])})")
    print(f"3차 {name}: 모델이 고른 답의 숫자 검사 {ok}/{len(load(name))}")

for name, key in FIRST_RUNS:
    ok = 0
    for q, out in zip(questions(key), load(name)):
        c = numcheck.check(out["answer"], out["tool_calls"], q["question"])
        ok += c["ok"]
        if not c["ok"]:
            print(f"  {name} {q['id']}: 지금 검사에 걸림 ({', '.join(c['unsupported'] + c['issues'])})")
    print(f"처음 {name}: 보여 준 답을 지금의 숫자 검사로 보면 {ok}/{len(load(name))}")
