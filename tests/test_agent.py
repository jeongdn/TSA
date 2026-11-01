"""질의응답 에이전트 테스트 (LLM·키 없이).

실행: python -m tests.test_agent   (먼저 python run_all.py로 outputs/를 만들어 두어야 한다)
1. 숫자 검사 규칙: 반올림·부호·퍼센트·날짜는 통과, 지어낸 숫자·없는 달은 걸린다.
2. 대본대로 답하는 가짜 모델: 도구 결과의 숫자로 답하면 한 번에 통과, 지어낸 숫자를 쓰면 다시 쓰게 해서 통과.
3. 도구: 잘못된 시리즈 이름은 고칠 수 있는 오류로 돌려주고, 미래 예측은 합계가 맞는다(전체 = ABS + PC = 하위 합).
4. OpenAI 호환 HTTP 경로: 로컬 가짜 서버가 /chat/completions 형식으로 도구 호출 -> 답변을 돌려준다.
"""
import http.server
import json
import threading

from tsa import numcheck, qa_agent, qa_tools


class Scripted:
    """정해 둔 순서대로 응답하는 가짜 모델. 응답이 함수면 지금까지의 대화를 받아 내용을 만든다."""

    def __init__(self, steps):
        self.steps, self.seen = list(steps), []

    def __call__(self, messages, tools):
        self.seen.append(list(messages))
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step


def tool_call(name, args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_0", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def last_tool_result(messages):
    return json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])


def answer_from_backtest(messages):
    r = last_tool_result(messages)["models"]
    return {"role": "assistant", "content": f"테스트 WAPE는 LightGBM {r['LightGBM']['wape_pct']}%, "
                                            f"ETS {r['ETS']['wape_pct']}%입니다."}


def test_numcheck_rules():
    log = [{"name": "t", "args": {}, "result": {"wape_pct": 15.24, "p": 0.542, "bias_pct": -0.3, "target": "2026-01",
                                                "v": 12345.6, "note": "P10~P90 80% 구간"}}]
    good = numcheck.check("WAPE는 15.2%(약 15%), P는 54.2%이고 0.3% 과소 예측입니다. "
                          "2026년 1월은 12,345.6톤, P10~P90 80% 구간입니다.", log)
    assert good["ok"], good
    bad = numcheck.check("WAPE는 15.3%이고 2026-02 예측은 12,000톤입니다.", log)
    assert sorted(bad["unsupported"]) == sorted(["15.3%", "2026-02", "12,000"]), bad
    asked = numcheck.check("2024년 6월 기준 3개월 앞 예측입니다.", [], question="2024-06 기준 3개월 앞 예측은?")
    assert asked["ok"], asked


def test_agent_passes_with_tool_numbers():
    llm = Scripted([tool_call("get_backtest", {}), answer_from_backtest])
    out = qa_agent.ask("LightGBM과 ETS의 WAPE는?", llm)
    assert out["ok"] and len(out["attempts"]) == 1, out["check"]
    assert out["tool_calls"][0]["name"] == "get_backtest"


def test_agent_retries_on_made_up_number():
    wrong = {"role": "assistant", "content": "LightGBM의 WAPE는 3.1415%입니다."}
    llm = Scripted([tool_call("get_backtest", {}), wrong, answer_from_backtest])
    out = qa_agent.ask("LightGBM의 WAPE는?", llm)
    assert out["ok"] and len(out["attempts"]) == 2, out
    assert out["attempts"][0]["check"]["unsupported"] == ["3.1415%"]
    assert "3.1415%" in llm.seen[-1][-1]["content"]      # 다시 쓰라는 메시지가 근거 없는 숫자를 알려 준다


def test_tools():
    names = qa_tools.series_names()
    assert names[:3] == ["TOTAL", "ABS", "PC"] and len(names) == 21
    try:
        qa_tools.call("get_forecast", {"series": "중국"})
        raise AssertionError("잘못된 시리즈 이름이 통과함")
    except qa_tools.ToolError as e:
        assert "ABS-CHN" in str(e)
    llm = Scripted([tool_call("get_forecast", {"series": "china"}), tool_call("get_forecast", {"series": "ABS-CHN"}),
                    {"role": "assistant", "content": "확인했습니다."}])
    out = qa_agent.ask("중국 ABS 예측?", llm)
    assert "error" in out["tool_calls"][0]["result"] and "rows" in out["tool_calls"][1]["result"]
    # 미래 예측은 합계가 맞아야 한다 (소수 1자리 반올림 오차만 허용)
    f = {s: qa_tools.get_forecast(s)["rows"] for s in names}
    for i in range(6):
        total, abs_, pc = (f[s][i]["forecast_t"] for s in ["TOTAL", "ABS", "PC"])
        assert abs(total - (abs_ + pc)) <= 0.1 + 1e-9
        assert abs(abs_ - sum(f[s][i]["forecast_t"] for s in names[3:12])) <= 0.5
        assert abs(pc - sum(f[s][i]["forecast_t"] for s in names[12:])) <= 0.5
        assert all(f[s][i]["p10_t"] <= f[s][i]["p90_t"] for s in names[3:])
    past = qa_tools.get_forecast("PC-USA", "2024-06", 1)["rows"][0]
    assert set(past) >= {"forecast_t", "actual_t", "p10_t", "p90_t"}


def test_http_client_with_local_fake_server():
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers.get("Authorization"), body))
            tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
            msg = answer_from_backtest(body["messages"]) if tool_msgs else tool_call("get_backtest", {})
            data = json.dumps({"choices": [{"message": msg}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        llm = qa_agent.OpenAICompatible(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="local-test",
                                        model="fake-model")
        out = qa_agent.ask("LightGBM과 ETS의 WAPE는?", llm)
    finally:
        srv.shutdown()
    assert out["ok"], out["check"]
    path, auth, body = seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer local-test" and body["model"] == "fake-model"
    assert {t["function"]["name"] for t in body["tools"]} == {"get_forecast", "get_backtest"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: 통과")
