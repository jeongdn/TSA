"""3차 에이전트: LLM은 도구를 고르고, 도구 결과의 사실 문장(facts) 중 답에 쓸 것의 id만 고른다 (final_answer).

답의 글은 모두 코드(qa_facts)가 만든 문장이다. 그래서 숫자와 대상(제품·국가·달·모델)의 짝이 틀릴 수 없다.
2차에서 숫자 검사로 못 막은 오류(맞는 숫자를 틀린 대상에 붙임)를 검사를 더하는 대신 구조로 막으려는 설계다.
남는 위험: 엉뚱한 문장 고르기, 도구·인자 잘못 고르기, 답할 수 있는데 없다고 하기.
HTTP 클라이언트와 .env 읽기, 글로 쓴 도구 호출 읽기, 걸린 문장 빼기는 2차(qa_agent)의 것을 그대로 쓴다.
"""
import json

from . import numcheck, qa_facts, qa_tools
from .qa_agent import OpenAICompatible, load_env, redact, text_tool_calls  # noqa: F401 (ask.py가 같은 클라이언트를 쓴다)

SYSTEM = f"""너는 한국의 ABS·PC 수출 수요예측 프로젝트 결과를 알려 주는 도우미다. 자료는 UN Comtrade의 한국 전체 수출 통계(톤)이고 롯데케미칼 판매 자료가 아니다.

답하는 순서
1. 알맞은 도구를 부른다.
   - 모델 성능·정확도·비교, 편향, P10~P90 적중률, 부트스트랩, 합계 맞추기(BU·MinT) -> get_backtest. 전체 요약은 인자 없이 부른다.
   - n개월 앞 성능 -> get_backtest(h=[n]) (series 없이). 시리즈 하나의 성능 -> get_backtest(series=시리즈).
   - 앞으로의 예측 물량 -> get_forecast(series=시리즈). 과거 기준 시점에서 낸 예측과 실제값 -> get_forecast(series=시리즈, origin='YYYY-MM').
   - 시리즈는 질문한 제품·국가와 같아야 한다. 'ABS 합계·ABS 전체'는 ABS, 'PC 합계·PC 전체'는 PC, 'ABS와 PC를 합친 전체'는 TOTAL, 나라 하나는 '제품-국가'(예: ABS-CHN)다.
   - 이 자료에는 ABS와 PC만 있다. 질문한 제품이 ABS나 PC가 아니면(예: PP, PE) 다른 시리즈의 값을 대신 쓰지 않는다.
   - 질문한 달·시점만 필요하면 h로 좁혀 부른다 (예: 2026년 4~6월이면 h=[4, 5, 6]).
2. 도구 결과의 facts는 코드가 만든 사실 문장이다. id는 '호출번호:내용'이다 (예: '1:2026-04-예측', '1:최저모델', '2:ABS-MEX-모델별WAPE'). 질문에 답하는 문장의 id만 질문 순서대로 골라 final_answer(answerable=true, fact_ids=[...])를 부른다. 질문한 대상(제품·국가·달·모델)이 그대로 나오는 문장만 고른다. 문장의 대상이 질문한 대상과 다르면 고르지 않는다. 질문이 여러 가지를 물으면 각각에 맞는 문장을 모두 고른다.
3. 도구 결과로 답할 수 없으면(자료에 없는 제품, {qa_tools.FIRST} ~ {qa_tools.LAST} 밖의 미래 기간 등) 도구로 확인한 뒤 final_answer(answerable=false, fact_ids=[그 이유를 보여 주는 문장 id])를 부른다.
4. 글로 직접 답하지 않는다. 답은 final_answer로만 한다."""

FINAL = {"type": "function", "function": {
    "name": "final_answer",
    "description": "최종 답을 낸다. 도구 결과의 facts에서 질문에 답하는 문장의 id를 고른다. 문장은 코드가 그대로 붙인다.",
    "parameters": {"type": "object", "properties": {
        "answerable": {"type": "boolean", "description": "도구 결과로 답할 수 있으면 true, 없으면 false"},
        "fact_ids": {"type": "array", "items": {"type": "string"},
                     "description": "고른 사실 문장 id (예: '1-2'), 답할 순서대로"}},
        "required": ["answerable", "fact_ids"]}}}
NUDGE_TOOL = "답하기 전에 알맞은 도구(get_backtest 또는 get_forecast)를 먼저 부르세요."
NUDGE_FINAL = ("글로 답하지 말고, 도구 결과 facts의 id를 골라 final_answer를 부르세요. 도구 호출이 어려우면 답에 쓸 id만 "
               "'1:2026-04-예측, 1:2026-05-예측'처럼 적어도 됩니다.")
REFUSE = "질문하신 내용은 이 프로젝트의 도구 결과로 답할 수 없습니다."
EMPTY = "도구 결과로 확인할 수 있는 답을 만들지 못했습니다."


def view(name, result, k, facts):
    """LLM에 보내는 도구 결과: 무엇을 조회했는지와 사실 문장(id 붙임)만. 전체 결과는 기록(log)에 남는다.
    id는 'k:키' (k번째 도구 호출, 키는 문장의 뜻: '2026-04-예측' 등)라 여러 번 불러도 겹치지 않는다."""
    if "error" in result:
        return result
    items = []
    for key, text in qa_facts.build(name, result):
        facts[f"{k}:{key}"] = text
        items.append({"id": f"{k}:{key}", "text": text})
    head = {key: result[key] for key in ("series", "origin", "h", "kind", "origin_note") if key in result}
    return {**head, "facts": items}


def run_tools(calls, log, facts):
    out = []
    for i, call in enumerate(calls):
        fn = call.get("function") or {}
        name, raw = fn.get("name"), fn.get("arguments") or "{}"
        args = {}
        try:
            args = raw if isinstance(raw, dict) else json.loads(raw)
            result = qa_tools.call(name, args)
        except (qa_tools.ToolError, json.JSONDecodeError, TypeError) as e:
            result = {"error": str(e), "hint": "인자를 고쳐 같은 도구를 다시 부르세요."}
        log.append({"name": name, "args": args, "result": result})
        out.append({"role": "tool", "tool_call_id": call.get("id") or f"call_{i}",
                    "content": json.dumps(view(name, result, len(log), facts), ensure_ascii=False)})
    return out


def _final_args(call):
    raw = (call.get("function") or {}).get("arguments") or "{}"
    try:
        a = raw if isinstance(raw, dict) else json.loads(raw)
    except json.JSONDecodeError:
        a = {}
    ids = a.get("fact_ids") or []
    if isinstance(ids, str):
        ids = ids.replace("[", "").replace("]", "").split(",")
    flag = a.get("answerable", True)
    return (flag if isinstance(flag, bool) else str(flag).lower() != "false"), [str(x).strip(" '\"") for x in ids]


def ids_in_text(text, facts):
    """도구 호출 대신 글에 id를 적은 경우: 글에 나온 순서대로, 실제로 있는 id만 고른다."""
    found = [(text.find(i), i) for i in facts if i in (text or "")]
    return [i for _, i in sorted(found)]


def render(answerable, ids, facts):
    texts = [facts[i] for i in dict.fromkeys(ids) if i in facts]
    if not answerable:
        return " ".join([REFUSE] + texts)
    return " ".join(texts) if texts else EMPTY


def _check(answer, log, question, answerable):
    """숫자 검사(뒷받침용). 답할 수 없다고 한 답은 그 자체가 '없다'는 말이라 범위 밖 질문 규칙은 빼고 본다."""
    c = numcheck.check(answer, log, question)
    if answerable is False:
        c["issues"] = [x for x in c["issues"] if "질문의 기간" not in x]
        c["ok"] = not c["unsupported"] and not c["issues"]
    return c


def ask(question, llm, max_steps=10, max_fixes=2):
    """질문 하나에 답한다. 반환 형식은 2차 ask()와 같고, final(LLM이 고른 것)·nudges·fixes·guard를 더한다."""
    tools = qa_tools.specs() + [FINAL]
    names = set(qa_tools.FUNCS) | {"final_answer"}
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    log, facts, nudges, fixes, final, texts = [], {}, 0, 0, None, []
    for _ in range(max_steps):
        msg = llm(messages, tools)
        calls = msg.get("tool_calls") or text_tool_calls(msg.get("content"), names)
        messages.append({"role": "assistant", "content": msg.get("content"), **({"tool_calls": calls} if calls else {})})
        if not calls:                              # 글로 답함: 글에 id가 있으면 그것으로, 없으면 돌려보낸다
            texts.append(msg.get("content") or "")
            picked = ids_in_text(msg.get("content"), facts)
            if picked:
                final = {"answerable": True, "fact_ids": picked, "from_text": True}
                break
            if nudges >= 2:
                break
            nudges += 1
            messages.append({"role": "user", "content": NUDGE_FINAL if log else NUDGE_TOOL})
            continue
        data = [c for c in calls if (c.get("function") or {}).get("name") != "final_answer"]
        fin = [c for c in calls if (c.get("function") or {}).get("name") == "final_answer"]
        if data:
            messages += run_tools(data, log, facts)
        if not fin:
            continue
        answerable, ids = _final_args(fin[0])
        problem = None
        if not log:
            problem = "먼저 get_backtest나 get_forecast를 불러 확인한 뒤 final_answer를 부르세요."
        elif answerable and not any(i in facts for i in ids):
            problem = "fact_ids가 도구 결과의 facts에 없습니다. 위 도구 결과의 id(예: '1-2')를 골라 final_answer를 다시 부르세요."
        if problem and fixes < max_fixes:
            fixes += 1
            messages.append({"role": "tool", "tool_call_id": fin[0].get("id") or "final",
                             "content": json.dumps({"error": problem}, ensure_ascii=False)})
            continue
        final = {"answerable": answerable, "fact_ids": ids}
        break

    guard = None
    if final is None:
        answer, answerable = EMPTY, None
    else:
        answerable = final["answerable"]
        answer = render(answerable, final["fact_ids"], facts)
    check = _check(answer, log, question, answerable)
    if answerable and any("질문의 기간" in x for x in check["issues"]):
        # 질문한 기간이 도구 결과에 없는데 답하려 했다: '답할 수 없음'과 예측 범위 안내로 바꾼다.
        # 이 두 문장은 코드가 config에서 만든 것이라 숫자 검사로 지우지 않는다 (지우면 '답을 못 만듦'만 남는다)
        guard = "질문한 기간이 도구 결과에 없어 '답할 수 없음'으로 바꿈"
        answerable, answer = False, " ".join([REFUSE, qa_facts.RANGE])
        check = {"ok": True, "n_numbers": 0, "unsupported": [], "issues": [], "items": [], "trusted": "guard"}
    shown, removed = (answer, 0) if check["ok"] else redact(answer, log, question)
    shown_ok = guard is not None or _check(shown, log, question, answerable)["ok"]   # 보여 주는 답을 같은 규칙으로 다시 봄
    return {"question": question, "answer": shown, "shown_ok": shown_ok,
            "model_answer": json.dumps(final, ensure_ascii=False),
            "ok": check["ok"], "check": check, "redacted": removed, "attempts": [{"answer": answer, "check": check}],
            "tool_calls": log, "final": final, "answerable": answerable, "nudges": nudges, "fixes": fixes,
            "guard": guard, "n_facts": len(facts), "model_texts": texts}
