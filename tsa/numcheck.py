"""답변 속 숫자 검사: 답변에 나온 숫자가 모두 도구 결과(또는 질문)에 있는지 확인한다.

허용하는 것 (이 밖의 숫자는 '근거 없음'):
1. 반올림: 답에 쓴 자릿수로 반올림해서 같으면 된다 (도구 15.24 -> 답 '15.2', '15').
2. 부호: '-0.3%'를 '0.3% 과소'처럼 부호 없이 써도 된다.
3. 퍼센트: '%'가 붙은 숫자는 도구 값의 100배와 같아도 된다 (확률 0.542 -> '54.2%').
4. 날짜: 'YYYY-MM'과 'YYYY년 M월'은 같은 달로 본다. 도구 결과나 질문에 나온 달이어야 한다.
P10·P90, ABS-CHN처럼 영문자에 붙은 숫자는 이름으로 보고 검사하지 않는다.
더하기·빼기·비율 계산으로 새로 만든 숫자는 도구 결과에 없으므로 걸린다 (의도한 동작).
한계: 값이 같은 숫자가 도구 결과 어딘가에 있으면 통과한다. 어느 항목과 맞았는지는 'source'에 남겨 사람이 확인한다.
"""
import re

KDATE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월")
DATE = re.compile(r"(?<![\d.])(\d{4})-(\d{1,2})(?![\d.])")
NUM = re.compile(r"(?<![A-Za-z0-9_.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(\s*%)?")


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


def references(tool_log, question=""):
    """도구 결과와 질문에서 비교할 숫자와 달을 모은다. 반환: [(값, 출처)], {달: 출처}."""
    nums, months = [], {}

    def add_text(s, src):
        ds, rest = _dates(s)
        for d in ds:
            months.setdefault(d, src)
            nums.extend([(float(d[:4]), src), (float(d[5:]), src)])     # '2026년', '1월'처럼 따로 쓴 경우
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
    return nums, months


def _source(v, dec, pct, refs):
    tol = 0.5 * 10 ** -dec + 1e-9                 # 답의 자릿수로 반올림한 값이면 같은 것으로 본다
    for r, src in refs:
        if abs(abs(v) - abs(r)) <= tol or (pct and abs(abs(v) - abs(100 * r)) <= tol):
            return src
    return None


def check(answer, tool_log, question=""):
    """반환: {'ok': 근거 없는 숫자가 없는가, 'n_numbers', 'unsupported': [원문], 'items': [{text, kind, source}]}."""
    refs, months = references(tool_log, question)
    ds, rest = _dates(answer or "")
    items = [{"text": d, "kind": "month", "source": months.get(d)} for d in ds]
    items += [{"text": raw, "kind": "number", "source": _source(v, dec, pct, refs)}
              for raw, v, dec, pct in _numbers(rest)]
    bad = [it["text"] for it in items if it["source"] is None]
    return {"ok": not bad, "n_numbers": len(items), "unsupported": bad, "items": items}
