"""Render a captured school evaluation as a safe, standalone Korean report.

  python scripts/render_school_report.py report.json --output report.html
  python scripts/render_school_report.py report.json --baseline before.json

This command reads only the supplied JSON files. It does not import application
settings, access a database/environment/session, or make HTTP/model requests.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
from urllib.parse import parse_qsl, urlsplit


class ReportError(ValueError):
    pass


_PRIVATE_KEY = re.compile(r"(?:cookie|password|passwd|secret|token|authorization|api.?key|session|conversation.?id|turn.?id|owner)", re.I)


def _public_value(value):
    if isinstance(value, dict):
        return {str(key): _public_value(item) for key, item in value.items() if not _PRIVATE_KEY.search(str(key))}
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ReportError("검사 조건과 결과는 JSON 값이어야 합니다.")


def _text(value, field):
    if not isinstance(value, str):
        raise ReportError(f"{field} 항목은 문자열이어야 합니다.")
    return value


def _official_url(value):
    if not isinstance(value, str) or re.search(r"[\x00-\x20\x7f]", value):
        return None
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or parsed.hostname != "life.hanyang.ac.kr"
            or parsed.username or parsed.password or parsed.port not in (None, 443)):
            return None
        if any(_PRIVATE_KEY.search(key) for key, _ in parse_qsl(parsed.query)) or _PRIVATE_KEY.search(parsed.fragment):
            return None
        return value
    except (ValueError, TypeError):
        return None


def _source(value):
    if not isinstance(value, dict):
        raise ReportError("출처 항목은 객체여야 합니다.")
    return {"id": _text(value.get("id", ""), "source.id"), "title": _text(value.get("title", "공식 출처"), "source.title"),
        "url": _official_url(value.get("url")), "text": _text(value.get("text", ""), "source.text"),
        "fetched_at": _text(value.get("fetched_at", ""), "source.fetched_at"), "date": _text(value.get("date", ""), "source.date"),
        "doc_type": _text(value.get("doc_type", ""), "source.doc_type")}


def _boolean(value, field):
    if not isinstance(value, bool):
        raise ReportError(f"{field} 항목은 true 또는 false여야 합니다.")
    return value


def _timestamp(value):
    if not value:
        return ""
    try:
        return datetime.fromisoformat(_text(value, "generated_at").replace("Z", "+00:00")).isoformat()
    except ValueError as error:
        raise ReportError("generated_at 시각 형식이 올바르지 않습니다.") from error


def build_report(report):
    """Allowlist public QA fields and derive honest counts from case results."""
    if not isinstance(report, dict) or not isinstance(report.get("cases"), list) or not isinstance(report.get("results"), list):
        raise ReportError("evaluate_school의 cases/results 배열이 필요합니다.")
    cases, known = [], set()
    for case in report["cases"]:
        if not isinstance(case, dict):
            raise ReportError("사례는 객체여야 합니다.")
        identity = _text(case.get("id"), "case.id")
        if not identity or identity in known:
            raise ReportError("사례 ID가 비어 있거나 중복됐습니다.")
        known.add(identity)
        steps = case.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ReportError("사례별 질문 steps 배열이 필요합니다.")
        normalized = []
        for step in steps:
            if not isinstance(step, dict) or not isinstance(step.get("expect", {}), dict):
                raise ReportError("질문 단계와 기대 조건은 객체여야 합니다.")
            normalized.append({"question": _text(step.get("question"), "step.question"),
                "expect": _public_value(step.get("expect", {}))})
        cases.append({"id": identity, "category": _text(case.get("category", "uncategorized"), "category"),
            "question": _text(case.get("question", normalized[-1]["question"]), "question"),
            "steps": normalized, "status": "PENDING"})
    case_map = {case["id"]: case for case in cases}
    evaluated = set()
    for result in report["results"]:
        if not isinstance(result, dict) or result.get("case_id") not in case_map:
            raise ReportError("결과에 알 수 없는 사례 ID가 있습니다.")
        identity = result["case_id"]
        if identity in evaluated:
            raise ReportError("같은 사례의 실행 결과가 중복됐습니다.")
        evaluated.add(identity)
        case = case_map[identity]
        result_steps = result.get("steps")
        if not isinstance(result_steps, list) or len(result_steps) != len(case["steps"]):
            raise ReportError("실행 결과의 단계 수가 질문 단계 수와 다릅니다.")
        all_passed = True
        for expected, actual_step in zip(case["steps"], result_steps):
            if not isinstance(actual_step, dict) or not isinstance(actual_step.get("checks"), list) or not actual_step["checks"]:
                raise ReportError("실행 단계에 비어 있지 않은 checks 배열이 필요합니다.")
            checks = []
            for check in actual_step["checks"]:
                if not isinstance(check, dict):
                    raise ReportError("검사 결과는 객체여야 합니다.")
                public = {"name": _text(check.get("name"), "check.name"),
                    "passed": _boolean(check.get("passed"), "check.passed")}
                for key in ("expected", "actual"):
                    if key in check:
                        public[key] = _public_value(check[key])
                checks.append(public)
            step_passed = _boolean(actual_step.get("passed"), "step.passed")
            if step_passed != all(check["passed"] for check in checks):
                raise ReportError("단계 통과 상태와 세부 검사 결과가 서로 다릅니다.")
            all_passed = all_passed and step_passed
            actual = actual_step.get("actual")
            if isinstance(actual, dict):
                sources = actual.get("sources", [])
                if not isinstance(sources, list):
                    raise ReportError("응답 sources 항목은 배열이어야 합니다.")
                public_actual = {"answer": _text(actual.get("answer", actual.get("detail", "")), "actual.answer"),
                    "mode": _text(actual.get("mode", ""), "actual.mode"), "model": _text(actual.get("model") or "", "actual.model"),
                    "sources": [_source(source) for source in sources],
                    "image_count": len(actual["images"]) if isinstance(actual.get("images"), list) else 0}
            else:
                public_actual = {"answer": "기록된 응답 객체가 없습니다.", "mode": "", "model": "", "sources": [], "image_count": 0}
            expected.update({"passed": step_passed, "checks": checks, "actual": public_actual,
                "http_status": actual_step.get("http_status") if isinstance(actual_step.get("http_status"), int) else None})
        if _boolean(result.get("passed"), "result.passed") != all_passed:
            raise ReportError("사례 통과 상태와 단계 결과가 서로 다릅니다.")
        case["status"] = "PASS" if all_passed else "FAIL"
    counts = Counter(case["status"] for case in cases)
    categories = []
    for name in sorted({case["category"] for case in cases}):
        group = [case for case in cases if case["category"] == name]
        categories.append({"name": name, "planned": len(group),
            **{status.lower(): sum(case["status"] == status for case in group) for status in ("PASS", "FAIL", "PENDING")}})
    return {"generated_at": _timestamp(report.get("generated_at")),
        "mode": report.get("mode") if report.get("mode") in {"prepared", "live", "offline_replay"} else "unknown",
        "summary": {"planned": len(cases), "evaluated": len(evaluated), "passed": counts["PASS"],
            "failed": counts["FAIL"], "pending": counts["PENDING"]}, "categories": categories, "cases": cases}


def _baseline(value, index):
    if not isinstance(value, dict):
        raise ReportError("이전 기록은 JSON 객체여야 합니다.")
    if isinstance(value.get("cases"), list) and isinstance(value.get("results"), list):
        summary = build_report(value)["summary"]
    else:
        raw = value.get("summary")
        if not isinstance(raw, dict):
            raise ReportError("이전 기록에 summary가 필요합니다.")
        summary = {}
        for key in ("planned", "evaluated", "passed", "failed"):
            number = raw.get(key)
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ReportError("이전 기록의 집계 값은 0 이상의 정수여야 합니다.")
            summary[key] = number
        if summary["passed"] + summary["failed"] != summary["evaluated"] or summary["evaluated"] > summary["planned"]:
            raise ReportError("이전 기록의 집계 합계가 맞지 않습니다.")
        summary["pending"] = summary["planned"] - summary["evaluated"]
    return {"label": _text(value.get("label", f"이전 점검 {index}"), "baseline.label"), "summary": summary}


def render_report(report, baselines=()):
    data = build_report(report)
    data["baselines"] = [_baseline(value, index) for index, value in enumerate(baselines, 1)]
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        serialized = serialized.replace(character, escaped)
    return _HTML.replace("__REPORT_JSON__", serialized)


_HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'none'; connect-src 'none'; font-src 'none'; base-uri 'none'; form-action 'none'">
<title>학교 정보 종합 점검 보고서</title>
<style>
:root{color-scheme:light;font-family:system-ui,-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Malgun Gothic',sans-serif;color:#172333;background:#f4f6f8;font-size:16px}*{box-sizing:border-box}body{margin:0;line-height:1.65}main{max-width:1120px;margin:auto;padding:44px 24px 64px}h1{font-size:2rem;line-height:1.3;margin:4px 0 12px}h2{font-size:1.2rem;margin:28px 0 12px}h3{font-size:1.05rem}p{margin:8px 0}.muted{color:#526170}.eyebrow{font-size:.8rem;font-weight:700;letter-spacing:.08em;color:#496278}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:24px 0}.card{padding:16px;background:white;border:1px solid #d9e1e8;border-radius:10px}.card span{display:block;color:#526170;font-size:.9rem}.card strong{font-size:1.8rem}.panel,.case{background:white;border:1px solid #d9e1e8;border-radius:10px}.panel{padding:16px}.filters{display:grid;grid-template-columns:160px 220px 1fr;gap:12px}.filters label{font-size:.9rem;font-weight:600}select,input{display:block;width:100%;font:inherit;padding:9px;margin-top:4px;border:1px solid #9aaaba;border-radius:6px;background:white;color:#172333}select:focus-visible,input:focus-visible,summary:focus-visible,a:focus-visible{outline:3px solid #2173bd;outline-offset:3px}table{width:100%;border-collapse:collapse;font-size:.92rem}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #dfe6ed;vertical-align:top}th{color:#425365;background:#f7f9fb}.table-scroll{overflow-x:auto}.case{margin:12px 0;overflow:hidden}.case>summary{padding:16px;cursor:pointer;font-weight:650}.case[open]>summary{border-bottom:1px solid #e0e6ed}.case-body{padding:4px 20px 20px}.step+.step{border-top:1px solid #dfe6ed;padding-top:10px;margin-top:24px}.badge{display:inline-block;min-width:58px;text-align:center;font-size:.78rem;line-height:1.8;border-radius:4px;margin-right:9px;padding:0 5px;vertical-align:middle}.pass{color:#0c593e;background:#e4f3eb}.fail{color:#9b2027;background:#fde9e9}.pending{color:#66520b;background:#fff4cd}.category{font-size:.8rem;font-weight:400;color:#526170;margin-right:8px}.case-id{font-size:.78rem;color:#526170;margin:4px 0}pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere;word-break:normal;padding:13px 15px;margin:8px 0;background:#f7f9fb;border:1px solid #e0e6ed;border-radius:6px;line-height:1.7}.json{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.83rem}.answer{background:#fafcfe}.checks summary,.source summary{cursor:pointer;padding:7px 0}.source{margin:12px 0;padding-left:12px;border-left:3px solid #d5e1ec}a{color:#135da0;overflow-wrap:anywhere}.failed-check{border-left:3px solid #b12730;padding-left:12px;margin:12px 0}.failure-name{font-weight:650;color:#95202a}.empty{text-align:center;padding:32px}footer{margin-top:28px;font-size:.88rem;color:#526170}button{font:inherit;background:white;border:1px solid #9aaaba;border-radius:6px;padding:7px 14px;cursor:pointer}.section-line{display:flex;align-items:center;justify-content:space-between;gap:12px}.hidden,[hidden]{display:none!important}
@media(max-width:680px){main{padding:24px 14px}.cards{grid-template-columns:repeat(2,1fr)}.filters{grid-template-columns:1fr}h1{font-size:1.6rem}.case-body{padding:4px 14px 16px}th,td{padding:7px}.category{display:block;margin:8px 0}.cards .card:first-child{grid-column:span 2}}
@media print{@page{size:A4;margin:16mm}body{background:white;font-size:10pt}main{max-width:none;padding:0}.filters,#reset{display:none!important}.panel,.case,.card{border-radius:0}.cards{gap:5px;margin:12px 0}.card{padding:8px}.card strong{font-size:16pt}details> :not(summary){display:block!important}details::details-content{content-visibility:visible;display:block}.case>summary{font-size:11pt}.step{break-inside:avoid}pre{background:white}a{color:inherit}.table-scroll{overflow:visible}}
</style></head><body><main>
<header><div class="eyebrow">QA · 공개 학교생활 정보</div><h1>학교 정보 종합 점검 보고서</h1>
<p class="muted">기록된 질문·기대 조건과 실제 응답을 대조한 결과입니다. 원문 답변은 Markdown을 실행하지 않고 일반 텍스트로 표시합니다.</p><p id="metadata" class="muted"></p></header>
<section aria-label="전체 점검 집계" id="summary" class="cards"></section>
<section id="baseline-section" hidden><h2>이전 점검 기록</h2><div class="panel table-scroll" id="baselines"></div><p class="muted">질문 수와 검사 조건은 기록마다 다를 수 있습니다. 각 기록의 집계로 확인하세요.</p></section>
<section><h2>질문 유형별 결과</h2><div class="panel table-scroll" id="categories"></div></section>
<section aria-labelledby="results-heading"><div class="section-line"><h2 id="results-heading">사례별 상세 결과</h2><button type="button" id="reset">필터 초기화</button></div>
<div class="panel filters"><label for="status">결과<select id="status"><option value="all">전체</option><option value="PASS">PASS · 통과</option><option value="FAIL">FAIL · 실패</option><option value="PENDING">미실행</option></select></label>
<label for="category">질문 유형<select id="category"><option value="all">전체 유형</option></select></label><label for="search">검색<input type="search" id="search" placeholder="질문·답변·검사 이름 검색" autocomplete="off"></label></div>
<p id="visible-count" class="muted" role="status" aria-live="polite"></p><div id="cases"></div><p id="empty" class="panel empty" hidden>선택한 조건에 맞는 사례가 없습니다.</p></section>
<footer>집계는 이 보고서에 포함된 QA 사례의 결과이며 서비스 전체의 정확도를 나타내는 점수가 아닙니다. 외부 스크립트·스타일·이미지는 요청하지 않습니다.</footer>
<noscript>상세 결과와 필터를 보려면 이 파일에서 JavaScript 실행을 허용해 주세요.</noscript></main>
<script id="report-data" type="application/json">__REPORT_JSON__</script>
<script>
(()=>{'use strict';
const data=JSON.parse(document.getElementById('report-data').textContent);
const names={count:'개수',list:'목록',categories:'종류별 집계',brand:'브랜드',business:'업종',location_filter:'위치 조건',location:'위치',details:'상세 정보',facility_core:'전체 시설 기본 정보',followup:'후속 대화',context_boundary:'대화 대상 구분',entity_boundary:'유사한 이름 구분',two_facilities:'복수 시설',multiintent:'복수 요청 항목',missing_information:'미확인 정보',current_status:'실시간 상태 구분',hours_boundary:'요일·24시간 조건',language_variant:'다양한 입력 표현',photos:'사진',hours:'운영시간',contact:'연락처',refusal:'근거 부족',scope_refusal:'수집 범위 밖 질문',notice:'공지',restaurant_notice:'식당 관련 공지',meal:'학식'};
names.mixed_notice='기본 정보와 공지';
const label=value=>names[value]||value;
function el(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=String(text);if(className)node.className=className;return node;}
function pre(value,json=false){return el('pre',json?JSON.stringify(value,null,2):value,json?'json':'answer');}
function table(headers,rows){const node=el('table');const head=el('thead'),tr=el('tr');for(const title of headers){const th=el('th',title);th.scope='col';tr.append(th);}head.append(tr);node.append(head);const body=el('tbody');for(const cells of rows){const row=el('tr');for(const value of cells)row.append(el('td',value));body.append(row);}node.append(body);return node;}
function badge(status){return el('span',status==='PENDING'?'미실행':status,'badge '+status.toLowerCase());}
function safeOfficial(value){try{const url=new URL(value);return url.protocol==='https:'&&url.hostname==='life.hanyang.ac.kr'&&!url.username&&!url.password&&(!url.port||url.port==='443');}catch{return false;}}
const modes={live:'실제 API 실행',offline_replay:'저장 응답 재검사',prepared:'질문 준비',unknown:'실행 방식 미상'};
document.getElementById('metadata').textContent='기록 시각: '+(data.generated_at||'기록 없음')+' · '+modes[data.mode];
for(const [title,key]of [['전체 사례','planned'],['실행','evaluated'],['PASS · 통과','passed'],['FAIL · 실패','failed'],['미실행','pending']]){const card=el('div',undefined,'card');card.append(el('span',title),el('strong',data.summary[key]));document.getElementById('summary').append(card);}
document.getElementById('categories').append(table(['질문 유형','전체','PASS','FAIL','미실행'],data.categories.map(row=>[label(row.name),row.planned,row.pass,row.fail,row.pending])));
if(data.baselines.length){document.getElementById('baseline-section').hidden=false;document.getElementById('baselines').append(table(['기록','전체','실행','PASS','FAIL','미실행'],data.baselines.map(row=>[row.label,row.summary.planned,row.summary.evaluated,row.summary.passed,row.summary.failed,row.summary.pending])));}
const status=document.getElementById('status'),category=document.getElementById('category'),search=document.getElementById('search');
for(const row of data.categories){const option=el('option',label(row.name));option.value=row.name;category.append(option);}
const cards=[];
for(const item of data.cases){const card=el('details',undefined,'case'),summary=el('summary');summary.append(badge(item.status),el('span',label(item.category),'category'),el('span',item.question));card.append(summary);card.open=item.status==='FAIL';const body=el('div',undefined,'case-body');body.append(el('p',item.id,'case-id'));
for(const [index,step]of item.steps.entries()){const section=el('section',undefined,'step');section.append(el('h3','단계 '+(index+1)+' · '+(step.passed===undefined?'미실행':step.passed?'PASS':'FAIL')),el('p','질문'),pre(step.question));const expected=el('details');expected.append(el('summary','기대 조건'),pre(step.expect,true));section.append(expected);
if(step.actual){section.append(el('p','실제 답변'),pre(step.actual.answer));section.append(el('p','HTTP '+(step.http_status??'기록 없음')+' · 응답 방식 '+(step.actual.mode||'기록 없음')+(step.actual.model?' · 모델 '+step.actual.model:'')+' · 반환 사진 '+step.actual.image_count+'개','muted'));const failures=step.checks.filter(check=>!check.passed);section.append(el('h3','실패한 검사 '+failures.length+'개'));if(!failures.length)section.append(el('p','실패한 검사가 없습니다.','muted'));for(const check of failures){const failure=el('div',undefined,'failed-check');failure.append(el('p',check.name,'failure-name'));if('expected'in check)failure.append(el('p','기대 값'),pre(check.expected,true));if('actual'in check)failure.append(el('p','실제 값'),pre(check.actual,true));section.append(failure);}const checks=el('details',undefined,'checks');checks.append(el('summary','전체 검사 '+step.checks.length+'개'),table(['검사','결과'],step.checks.map(check=>[check.name,check.passed?'PASS':'FAIL'])));section.append(checks);section.append(el('h3','공식 출처 '+step.actual.sources.length+'개'));for(const source of step.actual.sources){const detail=el('details',undefined,'source');detail.append(el('summary','['+source.id+'] '+source.title));if(source.url&&safeOfficial(source.url)){const link=el('a','공식 원문 열기');link.href=source.url;link.target='_blank';link.rel='noopener noreferrer';detail.append(link);}else detail.append(el('p','허용된 공식 HTTPS 링크가 없습니다.','muted'));detail.append(el('p','수집 시각: '+(source.fetched_at||'기록 없음')+(source.date?' · 자료 날짜: '+source.date:''),'muted'),pre(source.text));section.append(detail);}}
else section.append(el('p','아직 실행된 응답이 없습니다.','muted'));body.append(section);}card.append(body);document.getElementById('cases').append(card);cards.push({node:card,item,search:JSON.stringify(item).toLocaleLowerCase()});}
function filter(){const query=search.value.trim().toLocaleLowerCase();let visible=0;for(const card of cards){const show=(status.value==='all'||card.item.status===status.value)&&(category.value==='all'||card.item.category===category.value)&&(!query||card.search.includes(query));card.node.hidden=!show;if(show)visible++;}document.getElementById('visible-count').textContent=visible+' / '+cards.length+'개 사례 표시';document.getElementById('empty').hidden=visible!==0;}
status.addEventListener('change',filter);category.addEventListener('change',filter);search.addEventListener('input',filter);document.getElementById('reset').addEventListener('click',()=>{status.value='all';category.value='all';search.value='';filter();});filter();
})();
</script></body></html>'''


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ReportError(f"입력 JSON을 읽을 수 없습니다: {Path(path).name}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="evaluate_school 결과 JSON")
    parser.add_argument("--output", "-o", type=Path, default=Path("outputs/학교정보_종합점검_보고서.html"))
    parser.add_argument("--baseline", type=Path, action="append", default=[], help="이전 보고서 또는 label/summary JSON (반복 가능)")
    args = parser.parse_args(argv)
    try:
        output = render_report(_read(args.input), [_read(path) for path in args.baseline])
        if args.output.resolve() in {path.resolve() for path in (args.input, *args.baseline)}:
            raise ReportError("출력 파일은 입력 JSON과 다른 경로여야 합니다.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    except (ReportError, OSError, ValueError) as error:
        parser.error(str(error))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
