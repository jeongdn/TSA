export const meta = {
  name: 'agent-answer-review-blind',
  description: 'Blind re-review of 50 QA-agent answers (versions mixed and relabelled) with the same two lenses as the original review',
  phases: [{ title: 'Review', detail: '25 answer pairs x 2 lenses, read-only, file allowlist' }],
}

const SETS = args.sets
const ALLOWED = 'C:/Users/ame/Desktop/TSA/tsa/qa_tools.py, C:/Users/ame/Desktop/TSA/tsa/metrics.py and C:/Users/ame/Desktop/TSA/tsa/config.py'

const LENSES = {
  faithful: 'LENS: FAITHFULNESS. Go claim by claim through the FINAL answer: numbers AND qualitative statements (for example "...보다 나았다", "...로 예측됩니다", "정보를 찾을 수 없습니다"). For each claim decide whether the tool results in the same record support it. A number that does appear in the tool output but is attached to the wrong quantity, month, model, horizon or period is NOT supported. A statement that some information is unavailable is unsupported if the tool results in that record actually contain it. A claim about a period the tools do not cover is unsupported.',
  answering: 'LENS: DID IT ANSWER THE QUESTION. Decide whether the final answer actually answers what the user asked, whether the agent called the right tool with suitable arguments (read C:/Users/ame/Desktop/TSA/tsa/qa_tools.py to see exactly what get_forecast and get_backtest can return and which arguments they accept), how it handled tool errors, and whether the answer is written only in Korean. Also check every claim against the tool results, but your focus is usefulness and tool use.',
}

const SCHEMA = {
  type: 'object',
  properties: {
    reviews: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          claims: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                claim: { type: 'string' },
                verdict: { type: 'string', enum: ['supported', 'unsupported', 'misleading'] },
                evidence: { type: 'string' },
              },
              required: ['claim', 'verdict', 'evidence'],
            },
          },
          faithful: { type: 'boolean' },
          answers_question: { type: 'boolean' },
          right_tool_and_args: { type: 'boolean' },
          korean_only: { type: 'boolean' },
          summary_ko: { type: 'string' },
        },
        required: ['id', 'claims', 'faithful', 'answers_question', 'right_tool_and_args', 'korean_only', 'summary_ko'],
      },
    },
  },
  required: ['reviews'],
}

function prompt(file, pair, lensKey) {
  const list = pair.map(rid => `- rid "${rid}"`).join('\n')
  return `You are an independent reviewer of answers produced by a small question-answering agent (a local 7B LLM) for a demand-forecasting practice project (Korean ABS/PC export volumes from UN Comtrade, in tonnes). The agent may only quote numbers from two tools, get_forecast and get_backtest.

File: ${file}
It is a JSON array of anonymous answer records. Each record has: rid, question, answer (the final answer shown to the user), tool_calls (each with name, args, result: what the agent actually saw). The records come from different versions of the agent, mixed and relabelled. Do not try to work out which version wrote a record; judge each record on its own.

Review exactly these records:
${list}

${LENSES[lensKey]}

Rules:
- Judge each record ONLY against its own question and its own tool_calls[].result (what the agent actually saw). Do not use outside knowledge about the data.
- Field meanings: wape_pct = weighted absolute percentage error in %, lower is better; bias_pct = forecast minus actual in %; p_* fields = share of block-bootstrap resamples in which one method had the lower error (not a probability about the future); wape_reduction_* = relative reduction in WAPE in %, not percentage points; forecast_t = point forecast in tonnes for that row's 'target' month only; h = months ahead; origin 'latest' = forecasts made at 2025-12 for the months 2026-01 to 2026-06 only; p10_t/p90_t = 80% interval.
- faithful = true only if every claim is 'supported'. If the answer has no factual claim at all, faithful = true but say so in summary_ko.
- answers_question = true only if a user would get a correct, direct answer to what they asked.
- summary_ko: one short Korean sentence (80 characters or fewer) naming the main problem, or '문제 없음'. Use only numbers that appear in that record.
- Files: you may read ONLY the file above and, to see what the tools can return and which arguments they accept, ${ALLOWED}. Do not open, list or search any other file or folder: nothing else in the project folder or in the folder that holds the file above, no earlier reviews, no evaluation results, no documents, no tests. Do not run git. You may run python to compute with values read from the allowed files.
- Read-only task: do not create or modify any file.

Return one review per listed record, with id set to the given rid.`
}

const jobs = []
for (const s of SETS) {
  for (let i = 0; i < s.rids.length; i += 2) {
    const pair = s.rids.slice(i, i + 2)
    for (const lens of Object.keys(LENSES)) jobs.push({ file: s.file, pair, lens })
  }
}
log(`${jobs.length} review agents`)

phase('Review')
const results = await parallel(jobs.map(j => () =>
  agent(prompt(j.file, j.pair, j.lens), { label: `${j.lens}:${j.pair.join('+')}`, phase: 'Review', schema: SCHEMA })
    .then(r => (r ? r.reviews.map(v => ({ ...v, lens: j.lens })) : null))))

const byId = {}
for (const v of results.filter(Boolean).flat()) {
  byId[v.id] = byId[v.id] || {}
  byId[v.id][v.lens] = v
}
const all = SETS.flatMap(s => s.rids)
const missing = all.filter(id => !byId[id] || !byId[id].faithful || !byId[id].answering)
if (missing.length) log(`Missing a lens for: ${missing.join(', ')}`)
return { byId, missing }
