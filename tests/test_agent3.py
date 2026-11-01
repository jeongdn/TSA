"""3차 에이전트 테스트 (LLM·키 없이): 사실 문장, final_answer 흐름, 범위 밖 질문 처리.

실행: python -m tests.test_agent3   (먼저 python run_all.py로 outputs/를 만들어 두어야 한다)
1. 사실 문장: 도구마다 문장이 나오고, 대상 이름·달·값이 들어 있고, 비교(가장 낮은 모델·가장 많은/적은 달)가 맞다.
2. 흐름: 도구 -> final_answer(id) -> 답 = 고른 문장 그대로. 없는 id는 고치게 하고, 글로 답하면 돌려보낸다.
3. 답할 수 없는 질문: answerable=false면 '답할 수 없음' + 고른 문장. 범위 밖 기간을 답하려 하면 '답할 수 없음'으로 바꾼다.
"""
import json

from tsa import numcheck, qa_agent3, qa_facts, qa_tools


class Scripted:
    def __init__(self, steps):
        self.steps, self.seen = list(steps), []

    def __call__(self, messages, tools):
        self.seen.append(list(messages))
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step


def call(name, args, i=0):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def final(ids, answerable=True):
    return call("final_answer", {"answerable": answerable, "fact_ids": ids}, 9)


def last_facts(messages):
    """가장 최근 도구 결과의 사실 문장 목록 (final_answer 오류 메시지는 건너뛴다)."""
    results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    return [r for r in results if "facts" in r][-1]["facts"]


def test_facts():
    r = qa_tools.get_backtest()
    f = qa_facts.build("get_backtest", r)
    best = min(r["models"], key=lambda k: r["models"][k]["wape_pct"])
    assert any(f"가장 낮은 모델은 {best}" in t for t in f)
    t = qa_tools.get_forecast("TOTAL")
    ft = qa_facts.build("get_forecast", t)
    lo = min(t["rows"], key=lambda x: x["forecast_t"])["target"]
    hi = max(t["rows"], key=lambda x: x["forecast_t"])["target"]
    assert any(f"가장 많은 달은 {hi}" in x and f"가장 적은 달은 {lo}" in x for x in ft)
    assert all("ABS+PC 전체 합계(TOTAL)" in x for x in ft[:12])          # 대상 이름이 문장마다 들어 있다
    for name, args in [("get_backtest", {"h": [3]}), ("get_backtest", {"series": "ABS-MEX"}),
                       ("get_backtest", {"series": "ABS"}), ("get_backtest", {"series": "PC-USA", "h": [1, 2]}),
                       ("get_forecast", {"series": "PC-IND", "h": [4, 5, 6]}),
                       ("get_forecast", {"series": "ABS-TUR", "origin": "2024-12", "h": [2]})]:
        r = qa_tools.call(name, args)
        texts = qa_facts.build(name, r)
        assert texts and numcheck.check(" ".join(texts), [{"name": name, "result": r}])["ok"], (name, args)
    assert qa_facts.ida("앙상블(15.1%)") == "앙상블(15.1%)이다" and qa_facts.ida("ETS") == "ETS다"
    assert qa_facts.eun("LightGBM") == "LightGBM은" and qa_facts.eun("계절 나이브") == "계절 나이브는"


def test_answer_is_the_chosen_facts():
    picked = {}

    def choose(messages):
        facts = last_facts(messages)
        picked["text"] = next(x["text"] for x in facts if "가장 낮은 모델" in x["text"])
        return final([next(x["id"] for x in facts if x["text"] == picked["text"])])
    llm = Scripted([call("get_backtest", {}), choose])
    out = qa_agent3.ask("오차가 가장 작은 모델은?", llm)
    assert out["answer"] == picked["text"] and out["ok"] and out["final"]["answerable"]


def test_bad_ids_and_text_answers_are_sent_back():
    llm = Scripted([{"role": "assistant", "content": "LightGBM이 제일 좋습니다."},      # 글로 답함 -> 도구부터
                    call("get_backtest", {}),
                    {"role": "assistant", "content": "앙상블이 가장 낮습니다."},          # 글로 답함 -> final_answer로
                    final(["9-9"]),                                                   # 없는 id -> 고치게 함
                    lambda m: final([last_facts(m)[1]["id"]])])
    out = qa_agent3.ask("오차가 가장 작은 모델은?", llm)
    assert out["nudges"] == 2 and out["fixes"] == 1 and out["ok"] and "가장 낮은 모델" in out["answer"]


def test_text_tool_call_for_final_answer():
    raw = '<tool_call>{"name": "final_answer", "arguments": {"answerable": true, "fact_ids": ["1-2"]}}</tool_call>'
    llm = Scripted([call("get_backtest", {}), {"role": "assistant", "content": raw}])
    out = qa_agent3.ask("오차가 가장 작은 모델은?", llm)
    assert out["final"] == {"answerable": True, "fact_ids": ["1-2"]} and "가장 낮은 모델" in out["answer"]


def test_unanswerable_and_out_of_range():
    def refuse(messages):
        rid = next(x["id"] for x in last_facts(messages) if x["text"] == qa_facts.SCOPE)
        return final([rid], answerable=False)
    llm = Scripted([call("get_forecast", {"series": "TOTAL"}), refuse])
    out = qa_agent3.ask("폴리프로필렌(PP) 수출은 2026년 상반기에 얼마로 예측되나요?", llm)
    assert out["answer"].startswith(qa_agent3.REFUSE) and qa_facts.SCOPE in out["answer"] and out["ok"]
    # 범위 밖 기간(2027년)에 다른 달 값을 답하려 하면 '답할 수 없음'과 예측 범위 안내로 바뀐다
    pick_june = lambda m: final([next(x["id"] for x in last_facts(m) if "2026-06 수출 예측" in x["text"])])  # noqa: E731
    llm = Scripted([call("get_forecast", {"series": "ABS-CHN"}), pick_june])
    out = qa_agent3.ask("2027년 중국 ABS 수출은 얼마나 될까요?", llm)
    assert out["guard"] and out["answer"] == f"{qa_agent3.REFUSE} {qa_facts.RANGE}" and out["ok"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: 통과")
