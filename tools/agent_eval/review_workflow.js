export const meta = {
  name: 'agent-answer-review',
  description: 'Independently check each QA-agent answer against its own tool outputs (two lenses per answer)',
  phases: [{ title: 'Review', detail: '5 answer pairs x 2 lenses, read-only' }],
}

const FILE = args.file || 'C:/Users/ame/Desktop/TSA/outputs/cache/agent_answers.json'
const IDS = args.ids
const pairs = []
for (let i = 0; i < IDS.length; i += 2) {
  pairs.push([[i, IDS[i]], [i + 1, IDS[i + 1]]].filter(p => p[1] !== undefined))
}

const LENSES = {
  faithful: 'LENS: FAITHFULNESS. Go claim by claim through the FINAL answer: numbers AND qualitative statements (for example "...보다 나았다", "...로 예측됩니다", "정보를 찾을 수 없습니다"). For each claim decide whether the tool results in the same element support it. A number that does appear in the tool output but is attached to the wrong quantity, month, model, horizon or period is NOT supported. A statement that some information is unavailable is unsupported if the tool results in that element actually contain it. A claim about a period the tools do not cover is unsupported.',
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

function prompt(pair, lensKey) {
  const list = pair.map(([i, id]) => `- array index ${i} (0-based), id "${id}"`).join('\n')
  return `You are an independent reviewer of answers produced by a small question-answering agent (a local 7B LLM) for a demand-forecasting practice project (Korean ABS/PC export volumes from UN Comtrade, in tonnes). The agent may only quote numbers from two tools, get_forecast and get_backtest. An automatic check compared every number in each answer against the tool results and the question (its verdict is in each element's 'check'); your job is to find what that check cannot see.

File: ${FILE}
It is a JSON array. Each element has: question, answer (the final answer), attempts, check (the automatic number check, with which tool field each number matched), tool_calls (each with name, args, result).

Review exactly these elements:
${list}

${LENSES[lensKey]}

Rules:
- Judge each element ONLY against its own question and its own tool_calls[].result (what the agent actually saw). Do not use outside knowledge about the data.
- Field meanings: wape_pct = weighted absolute percentage error in %, lower is better; bias_pct = forecast minus actual in %; p_* fields = share of block-bootstrap resamples in which one method had the lower error (not a probability about the future); wape_reduction_* = relative reduction in WAPE in %, not percentage points; forecast_t = point forecast in tonnes for that row's 'target' month only; h = months ahead; origin 'latest' = forecasts made at 2025-12 for the months 2026-01 to 2026-06 only; p10_t/p90_t = 80% interval.
- faithful = true only if every claim is 'supported'. If the answer has no factual claim at all, faithful = true but say so in summary_ko.
- answers_question = true only if a user would get a correct, direct answer to what they asked.
- summary_ko: one short Korean sentence (80 characters or fewer) naming the main problem, or '문제 없음'. Use only numbers that appear in that element.
- Read-only task: do not create or modify any file.

Return one review per listed element, with id set to the given id.`
}

phase('Review')
const results = await parallel(pairs.flatMap(pair => Object.keys(LENSES).map(lens => () =>
  agent(prompt(pair, lens), {
    label: `${lens}:${pair.map(p => p[1]).join('+')}`,
    phase: 'Review',
    schema: SCHEMA,
  }).then(r => (r ? r.reviews.map(v => ({ ...v, lens })) : null)))))

const byId = {}
for (const v of results.filter(Boolean).flat()) {
  byId[v.id] = byId[v.id] || {}
  byId[v.id][v.lens] = v
}
const missing = IDS.filter(id => !byId[id] || !byId[id].faithful || !byId[id].answering)
if (missing.length) log(`Missing a lens for: ${missing.join(', ')}`)
return { byId, missing }
