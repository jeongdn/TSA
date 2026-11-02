"""질의응답 에이전트 테스트 (LLM·키 없이).

실행: python -m tests.test_agent   (먼저 python run_all.py로 outputs/를 만들어 두어야 한다)
1. 숫자 검사 규칙: 반올림·부호·퍼센트·날짜는 통과, 지어낸 숫자·없는 달은 걸린다.
2. 대본대로 답하는 가짜 모델: 도구 결과의 숫자로 답하면 한 번에 통과, 지어낸 숫자를 쓰면 다시 쓰게 해서 통과.
3. 도구: 잘못된 시리즈 이름은 고칠 수 있는 오류로 돌려주고, 미래 예측은 합계가 맞는다(전체 = ABS + PC = 하위 합).
4. OpenAI 호환 HTTP 경로: 로컬 가짜 서버가 /chat/completions 형식으로 도구 호출 -> 답변을 돌려준다.
5. .env 읽기: 같은 이름이 두 번 있으면 아래 줄, 이미 있는 환경 변수가 우선, BOM·따옴표·주석 처리.
"""
import http.server
import json
import os
import pathlib
import tempfile
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


def test_numcheck_period_and_language():
    """2차에서 더한 검사: 도구 결과에 없는 기간에 도구 숫자를 붙인 문장, 한자가 섞인 답."""
    log = [{"name": "get_forecast", "args": {}, "result": {"available_targets": "2026-01 ~ 2026-06", "rows": [
        {"h": 6, "target": "2026-06", "forecast_t": 11826.3}]}}]
    q = "2027년 중국 ABS 수출은?"
    wrong = numcheck.check("2027년 중국 ABS 수출은 약 11,826톤으로 예측됩니다.", log, q)
    assert not wrong["ok"] and wrong["unsupported"] == [] and wrong["issues"], wrong   # 숫자 하나하나는 근거가 있다
    for fine, question in [("2027년 예측은 없습니다. 미래 예측은 2026-06까지이고, 2026-06 예측은 11,826.3톤입니다.", q),
                           ("2027년 예측은 없고 예측은 6개월치뿐입니다.", q),
                           ("2026년 6월 예측은 11,826.3톤입니다.", "2026년 6월 중국 ABS 예측은?"),
                           ("2026-09 예측은 없습니다.", "2026-09 중국 ABS 예측은?")]:
        c = numcheck.check(fine, log, question)
        assert c["ok"], (fine, c)
    han = numcheck.check("2026-06 예측은 11,826.3톤입니다. 这比去年增长了.", log, "")
    assert not han["ok"] and any("한자" in x for x in han["issues"])
    # 범위 밖 질문에 다른 달 값만 늘어놓고 '없다'고 말하지 않으면 걸린다 (해와 달 모두)
    for answer, question in [("2026-06 예측은 11,826.3톤입니다.", q),
                             ("2026-06 예측은 11,826.3톤입니다.", "2026-09 중국 ABS 예측은?")]:
        c = numcheck.check(answer, log, question)
        assert not c["ok"] and any("질문의 기간" in x for x in c["issues"]), (answer, c)


def test_h_argument_forms():
    assert qa_tools._h(None) == [1, 2, 3, 4, 5, 6] and qa_tools._h([]) == [1, 2, 3, 4, 5, 6]
    assert qa_tools._h(2) == [2] and qa_tools._h("2") == [2] and qa_tools._h([3, 1, 2]) == [1, 2, 3]
    assert qa_tools._h("1~3") == [1, 2, 3] and qa_tools._h("1, 2") == [1, 2]
    for bad in ([7], "0", "x"):
        try:
            qa_tools._h(bad)
            raise AssertionError(f"잘못된 h가 통과함: {bad}")
        except qa_tools.ToolError:
            pass
    f = qa_tools.get_forecast("ABS-CHN", h=[1, 2, 3])
    assert [r["h"] for r in f["rows"]] == [1, 2, 3]
    assert abs(f["rows_total_t"] - sum(r["forecast_t"] for r in f["rows"])) <= 0.15 and "합계" in f["summary_ko"]
    # 예측할 달을 origin에 넣어도 그 달의 미래 예측으로 읽는다
    g = qa_tools.get_forecast("ABS-CHN", origin="2026-02")
    assert [r["target"] for r in g["rows"]] == ["2026-02"] and "origin_note" in g


def test_text_tool_call_and_redaction():
    """2차: 글로 쓴 도구 호출을 읽고, 다시 써도 근거 없는 숫자가 남으면 그 문장만 빼고 보여 준다."""
    raw = 'leton\n{"name": "get_backtest", "arguments": {}}\n</tool_call>'
    calls = qa_agent.text_tool_calls(raw)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_backtest"
    assert qa_agent.text_tool_calls('예: {"a": 1} 와 {"name": "other"}') == []
    llm = Scripted([{"role": "assistant", "content": raw}, answer_from_backtest])
    out = qa_agent.ask("LightGBM과 ETS의 WAPE는?", llm)
    assert out["ok"] and out["tool_calls"][0]["name"] == "get_backtest"

    def stubborn(messages):
        r = last_tool_result(messages)["models"]
        return {"role": "assistant", "content": f"LightGBM WAPE는 {r['LightGBM']['wape_pct']}%입니다. "
                                                "그리고 3개월 합계는 777.7톤입니다."}
    llm = Scripted([tool_call("get_backtest", {}), stubborn, stubborn, stubborn])
    out = qa_agent.ask("LightGBM WAPE는?", llm)
    assert not out["ok"] and out["redacted"] == 1 and len(out["attempts"]) == 3
    assert "777.7" not in out["answer"] and "LightGBM WAPE" in out["answer"] and "777.7" in out["model_answer"]
    assert numcheck.check(out["answer"], out["tool_calls"])["ok"]


def test_agent_nudges_when_no_tool_called():
    no_tool = {"role": "assistant", "content": "도구 결과에 없는 내용이라 모릅니다."}
    llm = Scripted([no_tool, tool_call("get_backtest", {}), answer_from_backtest])
    out = qa_agent.ask("LightGBM과 ETS의 WAPE는?", llm)
    assert out["ok"] and out["tool_calls"][0]["name"] == "get_backtest", out
    assert any(m["role"] == "user" and m["content"] == qa_agent.NUDGE for m in llm.seen[1])


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
    assert "error" in out["tool_calls"][0]["result"] and "hint" in out["tool_calls"][0]["result"]
    assert "rows" in out["tool_calls"][1]["result"]
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
    assert llm.prompt_tokens == [None, None]          # 가짜 서버는 usage를 주지 않는다


def test_load_env():
    keys = ["TSA_T_A", "TSA_T_B", "TSA_T_C", "TSA_T_D"]
    for k in keys:
        os.environ.pop(k, None)
    os.environ["TSA_T_D"] = "from-shell"
    try:
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / ".env"
            # 맨 앞의 \ufeff는 메모장이 파일 앞에 붙이는 BOM이다
            p.write_text("\ufeffTSA_T_A=first\nTSA_T_A=second\n# TSA_T_B=commented\nTSA_T_C=\"quoted\"\n"
                         "TSA_T_D=from-file\n", encoding="utf-8")
            qa_agent.load_env(p)
        assert os.environ["TSA_T_A"] == "second" and "TSA_T_B" not in os.environ
        assert os.environ["TSA_T_C"] == "quoted" and os.environ["TSA_T_D"] == "from-shell"
    finally:
        for k in keys:
            os.environ.pop(k, None)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: 통과")
