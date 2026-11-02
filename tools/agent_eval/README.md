# 에이전트 평가 도구 (7번)

RESULTS.md 11절의 '뜻 검토'와 '테스트 질문'을 만든 스크립트입니다. `.js` 네 개는 Claude Code의 Workflow 도구에서 돌리는 스크립트라, Claude Code 밖에서는 그대로 실행되지 않습니다. 파일 경로는 작성자 컴퓨터 기준이라 다른 곳에서 쓰려면 바꿔야 합니다.

| 파일 | 하는 일 |
|---|---|
| [`review_workflow.js`](review_workflow.js) | 뜻 검토. 답변 2개씩 묶어 두 관점(주장별 사실 확인 / 질문에 답했는지·도구 사용)의 검토 에이전트가 따로 판정합니다 (답변 10개면 에이전트 10개). 검토자에게는 기록 원본(`outputs/cache/agent_answers_*.json`)만 주고, 고친 세션의 판단은 주지 않습니다. 인자: `{"file": 답변 파일, "ids": [질문 id ...]}` |
| [`test_questions_workflow.js`](test_questions_workflow.js) | 테스트 질문. 작성 에이전트 1개가 쓰고, 검증 에이전트 1개가 형식·주제 비율·도구로 답할 수 있는지를 확인합니다. 둘 다 지난 답변과 검토 결과는 읽지 말라고 지시했고(막는 장치는 없음), 마지막 답에 질문 문장을 넣지 않게 해서 에이전트를 고치는 세션이 질문을 보지 않게 했습니다 (기록일 뿐 확인할 장치는 없음). |
| [`test2_questions_workflow.js`](test2_questions_workflow.js) | 테스트2 질문 (3차 비교용). 방식은 위와 같습니다. 질문 칸은 1차 테스트를 따라 정해서 대부분 같습니다 (두 칸은 종류가 다름). 에이전트의 답 틀(`tsa/qa_facts.py`, `tsa/qa_agent*.py`), 문서(README 7번 절부터, RESULTS·WORKLOG·LEARNING·SUMMARY), `tools/agent_eval/`, `tests/test_agent*.py`도 읽지 말라고 지시했습니다 (막는 장치는 없고, git 기록도 막지 않음). 앞의 두 질문 파일은 겹치지 않게 하려고 읽을 수 있습니다. |
| [`merge_review.py`](merge_review.py) | 검토 결과(Workflow가 돌려준 JSON)를 `outputs/agent_eval_<이름>.json`의 `semantic_review`에 합칩니다. 통과 = 두 검토자 모두 '모든 주장이 도구 결과와 맞다'고 보고, '질문에 답했는지' 관점 검토자가 '질문에 답했다'고 본 경우입니다. '질문에 답함'과 '알맞은 도구·인자'는 그 검토자 한 명의 판정입니다. 검토 요약(summary_ko)도 커밋하는 평가 파일에 들어갑니다. 요약에는 도구 결과의 숫자(작년 같은 달 대비 증감률 등)가 들어갈 수 있으니 커밋 전에 확인합니다. |
| [`make_blind.py`](make_blind.py) | 가린 재검토용 묶음 파일. 1차 테스트와 테스트2의 답을 질문 묶음별로 섞고, 버전을 알려 주는 항목(파일 이름, 숫자 검사 기록, 3차의 final·guard 등)을 빼고 새 id를 붙입니다. 같은 질문의 두 답이 한 짝이 되지 않게 합니다. 짝 표(`blind_key.json`)는 검토자에게 주지 않습니다. 같은 기록이면 같은 파일이 나옵니다. 묶음 파일에는 도구 결과(실제 물량 포함)가 들어 있습니다. 출력 폴더는 저장소 밖이나 `outputs/cache/` 아래로 정합니다. `.gitignore`가 `blind/`와 `blind_key.json`을 무시하지만, 코드가 막지는 않습니다. |
| [`review_blind_workflow.js`](review_blind_workflow.js) | 가린 재검토. 두 관점은 원래 검토와 같고, 읽을 파일을 묶음 파일과 `tsa/qa_tools.py`·`metrics.py`·`config.py`로 정했습니다 (막는 장치는 없어서, 실행 뒤 검토 기록으로 확인). 인자: `{"sets": [{"file": 묶음 파일, "rids": [id ...]}]}` |
| [`merge_blind.py`](merge_blind.py) | 가린 재검토 결과를 짝 표로 원래 실행에 되돌려 `semantic_review_blind`에 합칩니다. 판정 규칙은 `merge_review.py`와 같은 함수를 씁니다. 요약(summary_ko)에 대한 주의도 같습니다. |
| [`recheck.py`](recheck.py) | 기록에 남은 답을 지금의 숫자 검사로 다시 봅니다. 3차는 보호 장치가 바꾸기 전, 모델이 고른 문장으로 다시 검사합니다 (기록에는 보호 장치 답이 통과로 남음). 처음 에이전트는 기간 검사가 없던 그때의 검사 대신 지금 검사로 봅니다. 답변 기록(`outputs/cache/`)은 커밋하지 않아서 작성자 컴퓨터에서만 돕니다. |

합치기 예: `python tools/agent_eval/merge_review.py <검토 결과.json> . agent_eval_test.json`

한계: 검토자도 AI(Claude)이고, 에이전트를 고친 세션과 같은 모델이라 완전히 독립적이지는 않습니다. 검토자는 답변 파일 이름(예: `agent_answers_t2_v3.json`)으로 버전을 알 수 있고, 실행마다 새로 띄우므로 검토자에 따른 흔들림이 결과에 섞입니다. 읽는 파일도 막지 않아서, 일부 검토자는 이전 실행의 판정 파일(`outputs/cache/agent_review*.json`)과 git 기록을 봤습니다. 가린 재검토는 버전 이름과 이전 판정을 보는 것을 줄였지만, 버전을 완전히 가리지는 못합니다. 처음 에이전트의 도구 결과에는 summary_ko가 없고 3차 답은 모양이 달라 버전 무리는 알아볼 수 있으며, 검토자마다 자기 묶음의 답 전부가 든 파일을 받습니다. 판정 전문은 `outputs/cache/agent_review_*.json`에 남기지만 커밋하지 않습니다.
