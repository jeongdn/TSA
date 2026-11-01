"""답변 속 숫자 검사: 답변에 나온 숫자가 모두 도구 결과(또는 질문)에 있는지 확인한다.

허용하는 것 (이 밖의 숫자는 '근거 없음'):
1. 반올림: 답에 쓴 자릿수로 반올림해서 같으면 된다 (도구 15.24 -> 답 '15.2', '15').
2. 부호: '-0.3%'를 '0.3% 과소'처럼 부호 없이 써도 된다.
3. 퍼센트: '%'가 붙은 숫자는 도구 값의 100배와 같아도 된다 (확률 0.542 -> '54.2%').
4. 날짜: 'YYYY-MM'과 'YYYY년 M월'은 같은 달로 본다. 도구 결과나 질문에 나온 달이어야 한다.
P10·P90, ABS-CHN처럼 영문자에 붙은 숫자는 이름으로 보고 검사하지 않는다.
더하기·빼기·비율 계산으로 새로 만든 숫자는 도구 결과에 없으므로 걸린다 (의도한 동작).

그 밖에 '문제'로 잡는 것 (2차, 1차 평가에서 숫자 검사를 통과한 오류를 보고 추가):
5. 기간 불일치: 도구 결과에 없는 달·해(질문에만 있는 기간 포함)와 도구 결과의 물량·비율을 한 문장에 쓴 경우.
   예: 2026-06 예측값을 '2027년' 값이라고 쓴 문장. 작은 정수(개월 수 등)는 물량으로 보지 않는다.
6. 한자: 한국어 답에 한자(중국어)가 섞인 경우.
7. 범위 밖 질문: 질문에 도구 결과에 없는 기간이 있는데, 답이 그 기간은 없다고 말하지 않은 경우 (3차에서 추가).
한계: 값이 같은 숫자가 도구 결과 어딘가에 있으면 통과한다. 어느 항목과 맞았는지는 'source'에 남겨 사람이 확인한다.
"""
import re

KDATE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월")
DATE = re.compile(r"(?<![\d.])(\d{4})-(\d{1,2})(?![\d.])")
NUM = re.compile(r"(?<![A-Za-z0-9_.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(\s*%)?")
YEAR = re.compile(r"(?<![\d.])(\d{4})\s*년")                     # '2027년'처럼 해만 쓴 경우
RANGE = re.compile(r"(\d{4})-(\d{2})\s*~\s*(\d{4})-(\d{2})")       # '2023-07 ~ 2025-06'
SENTENCE = re.compile(r"(?<=[.!?。])\s+|\n+")
HAN = re.compile(r"[一-鿿]")                              # 한자(중국어) 글자
NEG = re.compile(r"없|않|아니|못|불가|제공되지|모릅|모른")                # '없다'는 뜻의 말


def _dates(text):
    """텍스트 속 달을 'YYYY-MM'으로 모으고, 그 자리를 지운 텍스트도 돌려준다 (날짜 속 숫자를 따로 세지 않게)."""
    found = []

    def take(m):
        y, mo = int(m.group(1)), int(m.group(2))
        if not 1 <= mo <= 12:
            return m.group(0)
        found.append(f"{y:04d}-{mo:02d}")
        return " "
    text = DATE.sub(take, KDATE.sub(take, text))
    return found, text


def _numbers(text):
    """[(원문, 값, 소수 자릿수, % 여부)]."""
    out = []
    for m in NUM.finditer(text):
        raw = m.group(0).strip()
        s = raw.rstrip("%").strip().replace(",", "")
        out.append((raw, float(s), len(s.split(".")[1]) if "." in s else 0, bool(m.group(1))))
    return out


def _span(text):
    """'2023-07 ~ 2025-06'처럼 범위로 쓴 달을 모두 펼친다 (기간 불일치 검사용)."""
    out = set()
    for y0, m0, y1, m1 in RANGE.findall(text):
        y, mo = int(y0), int(m0)
        while (y, mo) <= (int(y1), int(m1)):
            out.add(f"{y:04d}-{mo:02d}")
            y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
    return out


def references(tool_log, question=""):
    """도구 결과와 질문에서 비교할 숫자와 달을 모은다.
    반환: [(값, 출처)], {달: 출처} (질문 포함), 도구 결과에 나온 달 집합 (범위는 펼침, 질문 제외)."""
    nums, months, tool_months = [], {}, set()

    def add_text(s, src):
        ds, rest = _dates(s)
        for d in ds:
            months.setdefault(d, src)
            nums.extend([(float(d[:4]), src), (float(d[5:]), src)])     # '2026년', '1월'처럼 따로 쓴 경우
        if src != "질문":
            tool_months.update(ds)
            tool_months.update(_span(s))
        nums.extend((v, src) for _, v, _, _ in _numbers(rest))

    def walk(x, path):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            nums.append((float(x), path))
        elif isinstance(x, str):
            add_text(x, path)
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    for t in tool_log:
        walk(t["result"], t["name"])
    add_text(question or "", "질문")
    return nums, months, tool_months


def _source(v, dec, pct, refs):
    tol = 0.5 * 10 ** -dec + 1e-9                 # 답의 자릿수로 반올림한 값이면 같은 것으로 본다
    for r, src in refs:
        if abs(abs(v) - abs(r)) <= tol or (pct and abs(abs(v) - abs(100 * r)) <= tol):
            return src
    return None


def period_issues(answer, nums, tool_months):
    """도구 결과에 없는 기간(달·해)에 도구 결과의 물량·비율을 붙인 문장을 찾는다."""
    tool_nums = [(v, s) for v, s in nums if s != "질문"]
    years = {m[:4] for m in tool_months}
    out = []
    for sent in SENTENCE.split(answer or ""):
        ds, rest = _dates(sent)
        ys = YEAR.findall(rest)
        off = [d for d in ds if d not in tool_months] + [f"{y}년" for y in ys if y not in years]
        qty = [raw for raw, v, dec, pct in _numbers(YEAR.sub(" ", rest))
               if (pct or dec > 0 or abs(v) >= 100) and _source(v, dec, pct, tool_nums)]
        if off and qty:
            out.append(f"도구 결과에 없는 기간({', '.join(off)})에 도구 결과의 숫자({', '.join(qty)})를 붙인 문장이 있습니다")
    return out


def question_period_issues(answer, question, tool_months):
    """질문에 도구 결과에 없는 기간(달·해)이 있으면, 답에 그 기간과 '없다'는 말이 한 문장에 있어야 한다."""
    if not tool_months:                       # 도구를 안 불렀으면 판단할 수 없다
        return []
    ds, rest = _dates(question or "")
    years = {m[:4] for m in tool_months}
    off = [d for d in ds if d not in tool_months] + [y for y in YEAR.findall(rest) if y not in years]
    out = []
    for p in dict.fromkeys(off):
        said = False
        for sent in SENTENCE.split(answer or ""):
            mentions = p in _dates(sent)[0] if len(p) == 7 else p in sent     # 달은 달로, 해는 글자로 찾는다
            said = said or bool(mentions and NEG.search(sent))
        if not said:
            out.append(f"질문의 기간({p})은 도구 결과에 없는데, 그 기간의 예측(자료)이 없다는 말이 답에 없습니다")
    return out


def check(answer, tool_log, question=""):
    """반환: {'ok', 'n_numbers', 'unsupported': [원문], 'issues': [문장], 'items': [{text, kind, source}]}.
    ok = 근거 없는 숫자도 없고 문제(기간 불일치, 한자)도 없음."""
    refs, months, tool_months = references(tool_log, question)
    ds, rest = _dates(answer or "")
    items = [{"text": d, "kind": "month", "source": months.get(d)} for d in ds]
    items += [{"text": raw, "kind": "number", "source": _source(v, dec, pct, refs)}
              for raw, v, dec, pct in _numbers(rest)]
    bad = [it["text"] for it in items if it["source"] is None]
    issues = period_issues(answer, refs, tool_months) + question_period_issues(answer, question, tool_months)
    if HAN.search(answer or ""):
        issues.append("한국어 답에 한자(중국어)가 섞여 있습니다")
    return {"ok": not bad and not issues, "n_numbers": len(items), "unsupported": bad, "issues": issues,
            "items": items}
