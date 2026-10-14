#!/usr/bin/env python3
"""Read / annotate the team's HRM issue workbook ("Yash Gallery HRM.xlsx", Sheet1).

  dump <in.xlsx>                          → JSON rows (row number + column header → text)
  writeback <in.xlsx> <updates.json> <out.xlsx>

updates.json: {"<row>": {"English Translation": "...", "AI Status": "...", ...}, ...}
AI columns (R–U) are added after the team's last column; cells are patched straight in the
sheet XML so data validations, styles and the team's columns stay untouched (openpyxl drops
the Data Validation extension the sheet uses).
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from xml.sax.saxutils import escape

AI_COLUMNS = ["English Translation", "AI Status", "AI Comment", "Version Deployed"]
AI_FIRST_COL = "R"
AI_WIDTHS = [70, 18, 80, 22]
HEADER_STYLE = "6"
BODY_STYLE = "2"
SHEET = "xl/worksheets/sheet1.xml"


def _col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _col_index(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def _shared_strings(z: zipfile.ZipFile) -> list[str]:
    try:
        xml = z.read("xl/sharedStrings.xml").decode("utf-8")
    except KeyError:
        return []
    out = []
    for si in re.findall(r"<si>(.*?)</si>", xml, re.S):
        out.append("".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S)))
    return [_unescape(s) for s in out]


def _unescape(s: str) -> str:
    return s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")


def dump(path: str) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        ss = _shared_strings(z)
        xml = z.read(SHEET).decode("utf-8")
    rows: dict[int, dict[str, str]] = {}
    for rnum, body in re.findall(r'<row r="(\d+)"[^>]*?(?:/>|>(.*?)</row>)', xml, re.S):
        for ref, attrs, inner in re.findall(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', body or "", re.S):
            if not inner:
                continue
            v = re.search(r"<v>(.*?)</v>", inner, re.S)
            t = re.search(r'\bt="(\w+)"', attrs)
            if 't="s"' in attrs and v:
                val = ss[int(v.group(1))]
            elif t and t.group(1) == "inlineStr":
                val = _unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", inner, re.S)))
            else:
                val = _unescape(v.group(1)) if v else ""
            if val.strip().strip("\u200b\u200c"):
                rows.setdefault(int(rnum), {})[ref] = val
    header = rows.pop(1, {})
    out = []
    for r in sorted(rows):
        cells = rows[r]
        if not any(v.strip() and not v.strip().isdigit() for c, v in cells.items() if c not in ("A", "B", "C")):
            continue
        out.append({"row": r, **{header.get(c, c): v for c, v in sorted(cells.items(), key=lambda kv: _col_index(kv[0]))}})
    return out


def _cell(ref: str, text: str, style: str) -> str:
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'


def _set_cells(row_xml: str, rnum: int, values: dict[str, str], style: str) -> str:
    """Replace/insert cells in one <row>, keeping column order."""
    m = re.match(r'(<row [^>]*?)(/?)>', row_xml)
    head, selfclose = m.group(1), m.group(2)
    body = "" if selfclose else row_xml[m.end():-len("</row>")]
    cells = re.findall(r'<c r="[A-Z]+\d+"[^>]*?(?:/>|>.*?</c>)', body, re.S)
    by_col = {re.match(r'<c r="([A-Z]+)', c).group(1): c for c in cells}
    for col, text in values.items():
        by_col[col] = _cell(f"{col}{rnum}", text, style)
    ordered = "".join(by_col[c] for c in sorted(by_col, key=_col_index))
    head = re.sub(r'\sspans="[^"]*"', "", head)
    return f"{head}>{ordered}</row>"


def writeback(src: str, updates_path: str, dst: str) -> None:
    updates = json.load(open(updates_path, encoding="utf-8"))
    first = _col_index(AI_FIRST_COL)
    col_of = {name: _col_letter(first + i) for i, name in enumerate(AI_COLUMNS)}
    with zipfile.ZipFile(src) as z:
        xml = z.read(SHEET).decode("utf-8")
        members = {i.filename: z.read(i.filename) for i in z.infolist()}
        infos = {i.filename: i for i in z.infolist()}

    def patch_row(rnum: int, values: dict[str, str], style: str) -> None:
        nonlocal xml
        pat = re.compile(rf'<row r="{rnum}"[^>]*?(?:/>|>.*?</row>)', re.S)
        m = pat.search(xml)
        if m:
            xml = xml[: m.start()] + _set_cells(m.group(0), rnum, values, style) + xml[m.end():]
            return
        new_row = _set_cells(f'<row r="{rnum}"/>', rnum, values, style)
        later = [mm for mm in re.finditer(r'<row r="(\d+)"', xml) if int(mm.group(1)) > rnum]
        pos = later[0].start() if later else xml.index("</sheetData>")
        xml = xml[:pos] + new_row + xml[pos:]

    patch_row(1, {col_of[n]: n for n in AI_COLUMNS}, HEADER_STYLE)
    for rnum, vals in updates.items():
        patch_row(int(rnum), {col_of[k]: str(v) for k, v in vals.items() if k in col_of}, BODY_STYLE)

    last = _col_letter(first + len(AI_COLUMNS) - 1)
    xml = re.sub(r'<dimension ref="([A-Z]+)(\d+):[A-Z]+(\d+)"/>', rf'<dimension ref="\g<1>\g<2>:{last}\g<3>"/>', xml, count=1)
    xml = re.sub(rf'<col min="({first}|{first + 1}|{first + 2}|{first + 3})" [^>]*/>', "", xml)
    extra = "".join(
        f'<col min="{first + i}" max="{first + i}" width="{w}" customWidth="1"/>' for i, w in enumerate(AI_WIDTHS)
    )
    xml = xml.replace("</cols>", extra + "</cols>", 1)
    members[SHEET] = xml.encode("utf-8")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as out:
        for name, data in members.items():
            out.writestr(infos[name], data)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "dump" and len(sys.argv) == 3:
        print(json.dumps(dump(sys.argv[2]), ensure_ascii=False, indent=1))
    elif cmd == "writeback" and len(sys.argv) == 5:
        writeback(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(__doc__)
        sys.exit(2)
