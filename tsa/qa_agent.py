"""작은 질의응답 에이전트: OpenAI 호환 /chat/completions API + 도구 호출(function calling).

흐름
1. 질문과 도구 설명(get_forecast, get_backtest)을 모델에 보낸다.
2. 모델이 도구를 부르면 실행해서 결과(JSON)를 대화에 붙이고 다시 보낸다. 도구를 더 부르지 않으면 그 글이 답변이다.
3. 답변 속 숫자를 numcheck로 검사한다. 근거 없는 숫자가 있으면 그 숫자를 알려 주고 한 번 다시 쓰게 한다.
   그래도 남으면 ok=False로 돌려주고, 실행 화면(ask.py)에 경고와 함께 보여 준다.
OpenAI 호환이면 어디든 된다 (OpenAI, Ollama http://localhost:11434/v1, LM Studio, vLLM 등).
설정: 환경 변수나 .env 파일의 OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL. 키는 코드와 깃에 넣지 않는다.
"""
import json
import os
import re
import urllib.error
import urllib.request

from . import config, numcheck, qa_tools

SYSTEM = (
    "너는 한국의 ABS·PC 수출 수요예측 프로젝트 결과를 설명하는 도우미다. 자료는 UN Comtrade의 한국 전체 수출 통계(톤)이고 "
    "롯데케미칼 판매 자료가 아니다.\n"
    "규칙:\n"
    "- 숫자가 필요한 질문이면 먼저 get_forecast나 get_backtest를 불러 확인한다.\n"
    "- 답에는 도구 결과에 있는 숫자만 쓴다. 반올림은 괜찮지만 더하기·빼기·비율 계산으로 새 숫자를 만들지 않는다.\n"
    "- 물량은 톤, 비율은 %로 쓰고 달은 YYYY-MM 형식으로 쓴다.\n"
    "- 도구 결과에 없는 내용은 모른다고 말한다. 미래 실적을 장담하지 않는다.\n"
    "- 한국어로 짧게 답한다."
)


def load_env(path=config.ROOT / ".env"):
    """.env의 KEY=VALUE 줄을 환경 변수로 읽는다 (이미 설정된 환경 변수가 우선)."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


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
        if not self.model:
            raise SystemExit("OPENAI_MODEL이 없습니다. .env.example을 .env로 복사해 채우세요 (README '질의응답 에이전트').")

    def __call__(self, messages, tools):
        body = {"model": self.model, "messages": messages, "tools": tools, "temperature": 0}
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"API 오류 {e.code}: {_mask(e.read().decode('utf-8', 'replace'))[:400]}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"{self.base_url}에 연결하지 못했습니다: {e.reason}") from None


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
            result = {"error": str(e)}
        log.append({"name": name, "args": args, "result": result})
        out.append({"role": "tool", "tool_call_id": call.get("id") or f"call_{i}",
                    "content": json.dumps(result, ensure_ascii=False)})
    return out


def feedback(check):
    return ("답변의 다음 숫자는 도구 결과에서 찾을 수 없습니다: " + ", ".join(check["unsupported"]) + ". "
            "도구 결과에 있는 숫자만 (반올림만 허용) 쓰도록 답변 전체를 다시 써 주세요. 직접 계산한 차이·비율·합은 빼고, "
            "필요하면 도구를 다시 부르세요.")


def ask(question, llm, max_steps=6, max_retries=1):
    """질문 하나에 답한다. llm(messages, tools) -> assistant 메시지(dict).
    반환: 최종 답변, 숫자 검사 결과(ok), 시도별 답변·검사, 도구 호출 기록."""
    tools = qa_tools.specs()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    log, attempts = [], []
    for attempt in range(max_retries + 1):
        answer = None
        for _ in range(max_steps):
            msg = llm(messages, tools)
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content"), **({"tool_calls": calls} if calls else {})})
            if not calls:
                answer = (msg.get("content") or "").strip()
                break
            messages += run_tools(msg, log)
        if answer is None:
            answer = "(도구를 너무 많이 불러 답변을 끝내지 못했습니다)"
        check = numcheck.check(answer, log, question)
        attempts.append({"answer": answer, "check": check})
        if check["ok"] or attempt == max_retries:
            break
        messages.append({"role": "user", "content": feedback(check)})
    return {"question": question, "answer": attempts[-1]["answer"], "ok": attempts[-1]["check"]["ok"],
            "check": attempts[-1]["check"], "attempts": attempts, "tool_calls": log}
