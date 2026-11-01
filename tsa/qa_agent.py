"""작은 질의응답 에이전트: OpenAI 호환 /chat/completions API + 도구 호출(function calling).

흐름
1. 질문과 도구 설명(get_forecast, get_backtest)을 모델에 보낸다.
2. 모델이 도구를 부르면 실행해서 결과(JSON)를 대화에 붙이고 다시 보낸다. 도구를 더 부르지 않으면 그 글이 답변이다.
   도구를 한 번도 부르지 않고 답하면, 한 번은 도구부터 부르라고 돌려보낸다.
3. 답변 속 숫자를 numcheck로 검사한다. 근거 없는 숫자나 문제(기간 불일치, 한자)가 있으면 알려 주고 한 번 다시
   쓰게 한다. 그래도 남으면 ok=False로 돌려주고, 실행 화면(ask.py)에 경고와 함께 보여 준다.
2차(1차 평가를 본 뒤 고친 것): 도구 고르는 기준을 지시문에 넣었고, 도구 오류 때 다시 부르라는 힌트, 도구를 안 부른
답 돌려보내기, 기간 불일치·한자 검사를 더했다.
OpenAI 호환이면 어디든 된다 (OpenAI, Ollama http://localhost:11434/v1, LM Studio, vLLM 등).
설정: 환경 변수나 .env 파일의 OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL. 키는 코드와 깃에 넣지 않는다.
"""
import json
import os
import re
import urllib.error
import urllib.request

from . import config, numcheck, qa_tools

SYSTEM = f"""너는 한국의 ABS·PC 수출 수요예측 프로젝트 결과를 설명하는 도우미다. 자료는 UN Comtrade의 한국 전체 수출 통계(톤)이고 롯데케미칼 판매 자료가 아니다.

도구 고르기
- 모델 성능·정확도·비교, n개월 앞 성능, P10~P90 적중률, 부트스트랩 P, 합계 맞추기(BU·MinT) 결정 -> get_backtest. 전체 요약은 인자 없이 부른다.
- 시리즈 하나의 성능 -> get_backtest(series=시리즈).
- 앞으로의 예측 물량 -> get_forecast(series=시리즈). 과거 기준 시점에서 낸 예측과 실제값 -> get_forecast(series=시리즈, origin='YYYY-MM').
- 답하기 전에 반드시 도구를 한 번 이상 부른다. 도구가 error를 돌려주면 인자를 고쳐 다시 부른다.

답 쓰기
- 한국어로만 쓴다.
- 숫자는 도구 결과에 있는 값만 쓴다. 반올림은 괜찮지만 직접 계산한 새 숫자는 쓰지 않는다. 도구 결과의 summary_ko 문장을 바탕으로 쓰면 안전하다.
- 숫자마다 무엇의 값인지(시리즈, 달, 모델, 무엇 대비)를 함께 쓴다.
- 예측 물량은 달마다 따로다. 여러 달을 물으면 달마다 쓰거나 합계(rows_total_t)를 쓴다.
- 미래 예측은 {qa_tools.FIRST} ~ {qa_tools.LAST}만 있다. 그 밖의 기간은 예측이 없다고 답하고, 다른 달의 숫자를 대신 쓰지 않는다.
- 미래 실적을 장담하지 않는다. 짧게 답한다."""
NUDGE = ("답하기 전에 알맞은 도구(get_backtest 또는 get_forecast)를 먼저 불러 확인하세요. 도구 결과로 답할 수 없는 "
         "질문이면 도구 결과를 근거로 없다고 답하세요.")


def load_env(path=config.ROOT / ".env"):
    """.env의 KEY=VALUE 줄을 환경 변수로 읽는다. 같은 이름이 여러 번 있으면 아래 줄의 값을 쓴다.
    이미 설정된 환경 변수(터미널에서 직접 지정한 값)가 .env보다 우선한다. 메모장이 붙이는 BOM은 무시한다."""
    if not path.exists():
        return
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip("\"'")
    for k, v in values.items():
        os.environ.setdefault(k, v)


def _mask(text):
    """오류 메시지에 키 비슷한 문자열이 섞여 나오면 가린다."""
    return re.sub(r"(sk|key|gsk|up)[-_][A-Za-z0-9_\-*]{6,}", r"\1-***", text)


class OpenAICompatible:
    """POST {base_url}/chat/completions 한 번 = 모델 응답 한 번. 표준 라이브러리(urllib)만 쓴다."""

    def __init__(self, base_url=None, api_key=None, model=None, timeout=None):
        load_env()
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL")
        self.timeout = float(timeout or os.environ.get("OPENAI_TIMEOUT", 300))
        self.prompt_tokens = []          # 요청마다 서버가 센 입력 토큰 수 (문맥 길이를 넘어 잘렸는지 볼 때 쓴다)
        if not self.model:
            raise SystemExit("OPENAI_MODEL이 없습니다. .env.example을 .env로 복사해 채우세요 (README '질의응답 에이전트').")

    def __call__(self, messages, tools):
        body = {"model": self.model, "messages": messages, "tools": tools, "temperature": 0}
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"API 오류 {e.code}: {_mask(e.read().decode('utf-8', 'replace'))[:400]}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"{self.base_url}에 연결하지 못했습니다: {e.reason}") from None
        except TimeoutError:
            raise RuntimeError(f"{self.timeout:.0f}초 안에 응답이 없습니다 (OPENAI_TIMEOUT으로 늘릴 수 있음)") from None
        self.prompt_tokens.append((data.get("usage") or {}).get("prompt_tokens"))
        return data["choices"][0]["message"]


def run_tools(msg, log):
    """모델이 부른 도구를 실행해 tool 메시지를 만든다. 잘못된 인자는 오류 내용을 돌려줘 모델이 고치게 한다."""
    out = []
    for i, call in enumerate(msg.get("tool_calls") or []):
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
                    "content": json.dumps(result, ensure_ascii=False)})
    return out


def feedback(check):
    parts = []
    if check["unsupported"]:
        parts.append("답변의 다음 숫자는 도구 결과에서 찾을 수 없습니다: " + ", ".join(check["unsupported"]) + ".")
    parts += [f"{x}." for x in check.get("issues", [])]
    return (" ".join(parts) + " 도구 결과에 있는 숫자만 (반올림만 허용) 한국어로 쓰도록 답변 전체를 다시 써 주세요. "
            "직접 계산한 차이·비율·합은 빼고, 도구 결과에 없는 기간은 예측이 없다고 따로 쓰세요. 필요하면 도구를 다시 "
            "부르세요.")


def ask(question, llm, max_steps=6, max_retries=1):
    """질문 하나에 답한다. llm(messages, tools) -> assistant 메시지(dict).
    반환: 최종 답변, 숫자 검사 결과(ok), 시도별 답변·검사, 도구 호출 기록."""
    tools = qa_tools.specs()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    log, attempts, nudged = [], [], False
    for attempt in range(max_retries + 1):
        answer = None
        for _ in range(max_steps):
            msg = llm(messages, tools)
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content"), **({"tool_calls": calls} if calls else {})})
            if calls:
                messages += run_tools(msg, log)
                continue
            if not log and not nudged:       # 도구를 한 번도 안 부르고 답했으면 한 번은 도구부터 부르게 한다
                nudged = True
                messages.append({"role": "user", "content": NUDGE})
                continue
            answer = (msg.get("content") or "").strip()
            break
        if answer is None:
            answer = "(도구를 너무 많이 불러 답변을 끝내지 못했습니다)"
        check = numcheck.check(answer, log, question)
        attempts.append({"answer": answer, "check": check})
        if check["ok"] or attempt == max_retries:
            break
        messages.append({"role": "user", "content": feedback(check)})
    return {"question": question, "answer": attempts[-1]["answer"], "ok": attempts[-1]["check"]["ok"],
            "check": attempts[-1]["check"], "attempts": attempts, "tool_calls": log}
