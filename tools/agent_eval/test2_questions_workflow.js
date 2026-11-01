export const meta = {
  name: 'agent-test2-questions',
  description: 'Write and validate a second held-out set of 10 test questions for the QA agent; the developer stays blind to them',
  phases: [{ title: 'Write', detail: 'one writer' }, { title: 'Validate', detail: 'one independent validator' }],
}

const ROOT = 'C:/Users/ame/Desktop/TSA'
const OUT = `${ROOT}/tests/agent_questions_test2.json`

const COMMON = `Project: ${ROOT}. It is a practice demand-forecasting project on Korean ABS and PC export volumes (UN Comtrade, tonnes). To understand it, read README.md only up to (not including) the heading that starts with '## 7번', and read tsa/qa_tools.py. A small QA agent answers user questions with two data tools defined in tsa/qa_tools.py: get_forecast(series, origin='latest' or 'YYYY-MM', h) and get_backtest(series=None, h=None). You can call these tool functions directly to see what they return, from the project root, for example:
  cd ${ROOT} && PYTHONIOENCODING=utf-8 .venv/Scripts/python -c "from tsa import qa_tools; import json; print(json.dumps(qa_tools.get_backtest(), ensure_ascii=False))"
Hard rules:
- Do NOT run ask.py or any LLM.
- Do NOT read: outputs/agent_eval*.json, anything under outputs/cache/, RESULTS.md, LEARNING.md, SUMMARY.md, WORKLOG.md, README.md from the '## 7번' heading on, tsa/qa_facts.py, tsa/qa_agent.py, tsa/qa_agent3.py, tools/agent_eval/, tests/test_agent*.py. They contain past agent answers, reviews and the agent's answer templates, and the new questions must be realistic user questions, not tailored to past failures or to the agent's templates.
- You MAY read tests/agent_questions.json and tests/agent_questions_test.json (earlier question sets) only to avoid repeating them.
- Do not create, modify or delete any file except ${OUT}. Run any checks inline (python -c) without writing scripts to disk.
- Your final reply must NOT contain any question text: the developer of the agent must stay blind to the test questions.`

const MIX = `Topic mix (the same as tests/agent_questions_test.json, exactly):
- 4 questions answered by get_backtest: one overall model comparison or ranking; one about a specific horizon (1 to 6 months ahead); one about the P10~P90 interval coverage or the forecast bias; one about a specific bottom series (one product to one country) or about a product/total aggregate and the reconciliation (BU vs MinT).
- 4 questions answered by get_forecast with origin 'latest' (forecasts for the months after the last data month): one bottom series over several months; one product total (ABS or PC); the grand total of ABS and PC; one single month for a bottom series.
- 1 question about a past test origin (2023-07 ~ 2025-06): what was forecast and what actually happened.
- 1 question the tools cannot answer (for example a period outside the available forecasts, or a product or country not in the data). The correct behaviour for the agent is to say it is not available.
Question rules: natural Korean, as a business user at a chemical company would ask; no tool names, field names, series codes in brackets or JSON; each answerable question must be answerable from one or two tool calls; do not repeat or closely paraphrase any question in the two earlier sets (use different series, horizons, months, origins or angles).
File format: a JSON array (UTF-8, indent 2) of exactly 10 objects {"id": "u01" ... "u10", "question": "...", "expect_tools": ["get_backtest"] or ["get_forecast"], or [] for the unanswerable one}.`

const WRITE_SCHEMA = {
  type: 'object',
  properties: {
    written: { type: 'boolean' },
    n_questions: { type: 'number' },
    per_category: { type: 'object' },
    all_answerable_checked_with_tools: { type: 'boolean' },
  },
  required: ['written', 'n_questions', 'per_category', 'all_answerable_checked_with_tools'],
}

const VALIDATE_SCHEMA = {
  type: 'object',
  properties: {
    valid: { type: 'boolean' },
    n_questions: { type: 'number' },
    category_counts: { type: 'object' },
    answerability_checked: { type: 'boolean' },
    fixes: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['valid', 'n_questions', 'category_counts', 'answerability_checked', 'fixes', 'notes'],
}

phase('Write')
const written = await agent(`${COMMON}

Task: write a new held-out TEST set of exactly 10 questions for this agent and save it to ${OUT}.

${MIX}

Check every answerable question by calling the tool functions yourself (inline python only). Reply only with counts per category and whether you checked them (no question text).`,
  { label: 'writer', phase: 'Write', schema: WRITE_SCHEMA })

phase('Validate')
const validated = await agent(`${COMMON}

Task: validate the held-out test question file ${OUT}, which another agent just wrote. Check that:
- it is a valid JSON array of exactly 10 objects with keys id, question, expect_tools, and the ids u01..u10 are unique;
- it follows the topic mix below exactly;
- every answerable question really can be answered from one or two calls of the tools (call them inline to check), and its expect_tools is right;
- the unanswerable question really cannot be answered from the tools;
- no question repeats or closely paraphrases tests/agent_questions.json or tests/agent_questions_test.json;
- every question is natural Korean with no tool names, field names or JSON.

${MIX}

If something is wrong, fix ${OUT} minimally (rewrite only what is broken). In 'fixes', describe each fix without quoting any question text (for example "u05: expect_tools corrected"). Also report in 'notes' whether you found any leftover files that might reveal the questions.`,
  { label: 'validator', phase: 'Validate', schema: VALIDATE_SCHEMA })

return { written, validated }
