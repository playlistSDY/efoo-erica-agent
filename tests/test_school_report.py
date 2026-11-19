"""Offline standalone report/privacy/security regressions over synthetic JSON."""

from copy import deepcopy
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_school_report.py"
spec = importlib.util.spec_from_file_location("school_report_renderer", SCRIPT)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
URL = "https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id=49"


def sample():
    cases, results = [], []
    for identity, category, passed, question in [("contact.ok", "contact", True, "우체국 전화번호는?"),
        ("hours.bad", "hours", False, "토요일 몇 시까지 해?"),
        ("notice.pending", "notice", None, "식권 결제 공지를 알려줘")]:
        cases.append({"id": identity, "category": category, "question": question,
            "steps": [{"question": question, "expect": {"facts": [{"kind": "phone", "value": "031-400-0049"}]}}]})
        if passed is not None:
            results.append({"case_id": identity, "category": category, "passed": passed, "steps": [
                {"question": question, "http_status": 200, "passed": passed,
                 "checks": [{"name": "nonempty_answer", "passed": True},
                    {"name": "fact_phone", "passed": passed, "expected": "031-400-0049", "actual": "031-400-0049" if passed else "누락"}],
                 "actual": {"answer": "**우체국** 전화번호는 031-400-0049예요. [S1]", "mode": "retrieval", "model": None,
                    "sources": [{"id": "S1", "title": "우체국 공식 안내", "url": URL, "text": "연락처 031-400-0049",
                        "fetched_at": "2026-11-14T00:00:00+00:00", "date": "2026-11-14", "doc_type": "profile"}], "images": []}}]})
    return {"schema_version": 1, "generated_at": "2026-11-14T00:00:00Z", "mode": "live",
        "cases": cases, "results": results, "summary": {"passed": 999}, "ground_truth": {"unused": "not exported"}}


class Document(HTMLParser):
    def __init__(self, value):
        super().__init__(convert_charrefs=False)
        self.tags, self.scripts, self.current_script = [], [], None
        self.feed(value)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag == "script":
            self.current_script = {"attrs": attrs, "text": ""}
            self.scripts.append(self.current_script)

    def handle_data(self, value):
        if self.current_script is not None:
            self.current_script["text"] += value

    def handle_endtag(self, tag):
        if tag == "script":
            self.current_script = None

    def data(self):
        return json.loads(next(item["text"] for item in self.scripts if item["attrs"].get("id") == "report-data"))


def test_recomputes_real_case_category_counts_and_preserves_pending():
    report = renderer.build_report(sample())
    assert report["summary"] == {"planned": 3, "evaluated": 2, "passed": 1, "failed": 1, "pending": 1}
    categories = {row["name"]: row for row in report["categories"]}
    assert categories["contact"]["pass"] == categories["hours"]["fail"] == categories["notice"]["pending"] == 1
    assert [case["status"] for case in report["cases"]] == ["PASS", "FAIL", "PENDING"]
    step = report["cases"][1]["steps"][0]
    assert step["checks"][1]["actual"] == "누락"
    assert step["expect"]["facts"][0]["value"] == "031-400-0049"
    assert step["actual"]["answer"].startswith("**우체국**")
    assert step["actual"]["sources"][0]["url"] == URL


def test_prepared_report_does_not_present_unrun_cases_as_passes():
    value = sample()
    value["results"] = []
    value["mode"] = "prepared"
    report = renderer.build_report(value)
    assert report["summary"]["evaluated"] == report["summary"]["passed"] == report["summary"]["failed"] == 0
    assert report["summary"]["pending"] == 3


def test_only_public_allowlisted_qa_data_is_exported_and_input_is_unchanged():
    value = sample()
    actual = value["results"][0]["steps"][0]["actual"]
    actual.update({"cookies": {"efoo_session": "PRIVATE_COOKIE"}, "api_key": "PRIVATE_KEY",
        "conversation_id": "PRIVATE_CONVERSATION", "turn_id": "PRIVATE_TURN", "owner_hash": "PRIVATE_OWNER",
        "private_unknown_field": "PRIVATE_UNUSED"})
    actual["sources"][0]["metadata"] = {"token": "PRIVATE_SOURCE_TOKEN"}
    expected = value["cases"][0]["steps"][0]["expect"]
    expected["nested"] = {"safe": "공개 조건", "session_token": "PRIVATE_NESTED_SESSION"}
    value["results"][0]["steps"][0]["checks"][0]["actual"] = {"value": 1, "conversationId": "PRIVATE_CHECK_ID"}
    value["ground_truth"]["database_path"] = "PRIVATE_DATABASE_PATH"
    original = deepcopy(value)
    html = renderer.render_report(value)
    assert "PRIVATE_" not in html
    assert "공개 조건" in html and "031-400-0049" in html
    assert value == original


@pytest.mark.parametrize("url", ["http://life.hanyang.ac.kr/info", "https://evil.example/info",
    "https://life.hanyang.ac.kr.evil.example/info", "javascript:alert(1)", "//life.hanyang.ac.kr/info",
    "https://user:secret@life.hanyang.ac.kr/info", "https://life.hanyang.ac.kr:8443/info",
    "https://life.hanyang.ac.kr/info?api_key=PRIVATE_QUERY", "https://life.hanyang.ac.kr/info#session=PRIVATE_FRAGMENT"])
def test_sources_link_only_to_official_https_without_embedded_sensitive_values(url):
    value = sample()
    value["results"][0]["steps"][0]["actual"]["sources"][0]["url"] = url
    normalized = Document(renderer.render_report(value)).data()
    assert normalized["cases"][0]["steps"][0]["actual"]["sources"][0]["url"] is None
    assert "PRIVATE_QUERY" not in renderer.render_report(value) and "PRIVATE_FRAGMENT" not in renderer.render_report(value)


def test_untrusted_model_html_cannot_close_json_script_or_create_active_tags():
    value = sample()
    hostile = '</script><script>window.pwned=1</script><img src="https://evil.example/leak" onerror="alert(1)">\u2028\u2029&'
    value["results"][0]["steps"][0]["actual"]["answer"] = hostile
    value["cases"][0]["question"] = '<svg onload="alert(1)">'
    html = renderer.render_report(value)
    document = Document(html)
    assert len(document.scripts) == 2
    assert not any(tag in {"img", "svg", "iframe", "object"} for tag, _ in document.tags)
    assert document.data()["cases"][0]["steps"][0]["actual"]["answer"] == hostile
    embedded = next(item["text"] for item in document.scripts if item["attrs"].get("id") == "report-data")
    assert "</script" not in embedded and "\\u003c" in embedded and "\\u2028" in embedded and "\\u2029" in embedded


def test_standalone_document_has_no_external_assets_or_html_insertion_sink():
    document = Document(renderer.render_report(sample()))
    for tag, attributes in document.tags:
        assert not (tag == "script" and "src" in attributes)
        assert not (tag == "link" and attributes.get("rel") == "stylesheet")
        assert not any(key.lower().startswith("on") for key in attributes)
    client = document.scripts[-1]["text"]
    assert ".textContent" in client
    assert ".innerHTML" not in client and "insertAdjacentHTML" not in client
    assert "fetch(" not in client and "XMLHttpRequest" not in client
    identifiers = {attrs.get("id") for _, attrs in document.tags}
    assert {"status", "category", "search", "reset", "visible-count"} <= identifiers


def test_optional_multiple_baselines_accept_full_and_summary_inputs():
    previous = {"label": "첫 점검", "summary": {"planned": 16, "evaluated": 16, "passed": 2, "failed": 14},
        "cookie": "PRIVATE_BASELINE_COOKIE"}
    captured = sample()
    captured["label"] = "후속 점검"
    data = Document(renderer.render_report(sample(), [previous, captured])).data()
    assert data["baselines"][0] == {"label": "첫 점검", "summary": {"planned": 16, "evaluated": 16, "passed": 2, "failed": 14, "pending": 0}}
    assert data["baselines"][1]["summary"]["passed"] == 1
    assert "PRIVATE_BASELINE_COOKIE" not in json.dumps(data)


@pytest.mark.parametrize("mutation", [
    lambda report: report["cases"].append(deepcopy(report["cases"][0])),
    lambda report: report["results"].append(deepcopy(report["results"][0])),
    lambda report: report["results"][0].update(case_id="unknown"),
    lambda report: report["results"][0].update(passed="true"),
    lambda report: report["results"][0]["steps"][0].update(passed=False),
    lambda report: report["results"][0]["steps"][0].update(checks="invalid"),
    lambda report: report["results"][0]["steps"][0].update(checks=[]),
    lambda report: report.update(generated_at="invalid"),
    lambda report: report["results"][0]["steps"][0]["actual"].update(model={"api_key": "PRIVATE_KEY"}),
])
def test_invalid_or_contradictory_reports_fail_without_fabricated_passes(mutation):
    report = sample()
    mutation(report)
    with pytest.raises(renderer.ReportError):
        renderer.render_report(report)


def test_cli_reads_only_supplied_json_writes_utf8_html_and_preserves_inputs(tmp_path, capsys):
    source = tmp_path / "report.json"
    baseline = tmp_path / "before.json"
    output = tmp_path / "nested" / "학교정보.html"
    source.write_text(json.dumps(sample(), ensure_ascii=False), encoding="utf-8")
    baseline.write_text(json.dumps({"label": "이전 결과", "summary": {"planned": 79, "evaluated": 79, "passed": 73, "failed": 6}}), encoding="utf-8")
    original = source.read_bytes()
    assert renderer.main([str(source), "--output", str(output), "--baseline", str(baseline)]) == 0
    assert source.read_bytes() == original
    assert Document(output.read_text(encoding="utf-8")).data()["baselines"][0]["summary"]["passed"] == 73
    assert str(output) in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        renderer.main([str(source), "--output", str(source)])
    assert error.value.code == 2 and source.read_bytes() == original
