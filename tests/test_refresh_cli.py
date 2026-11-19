from datetime import date
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

with patch("dotenv.load_dotenv"):
    from app.cli import main


def report(*, errors=None, index_status="indexed"):
    return {"start_date": "2026-11-13", "end_date": "2026-11-14", "dates": ["2026-11-13", "2026-11-14"],
        "checked": 8, "updated": 2, "fresh": 6, "skipped": 0, "indexed": 6,
        "status": "partial" if errors else "ready", "index_status": index_status,
        "errors": errors or [], "targets": []}


@pytest.fixture
def command():
    settings = SimpleNamespace(max_date_range_days=31, api_key="isolated-cli-test")
    store, service = Mock(), Mock()
    service.ensure.return_value = report()
    with patch("app.cli.Settings", return_value=settings) as settings_factory, \
         patch("app.cli.Store", return_value=store) as store_factory, \
         patch("app.cli.MealRefreshService", return_value=service) as service_factory, \
         patch("app.cli.MealCrawler") as crawler, patch("app.cli.index_pending") as global_index:
        yield SimpleNamespace(settings=settings, store=store, service=service, settings_factory=settings_factory,
            store_factory=store_factory, service_factory=service_factory, crawler=crawler, global_index=global_index)


def run(*arguments):
    with patch("sys.argv", ["cli", *arguments]):
        return main()


def test_custom_inclusive_range_and_restaurants_use_refresh_service_and_emit_only_json(command, capsys):
    run("refresh", "--start", "2026-11-13", "--end", "2026-11-14", "--restaurants", "re12", "re13", "--force")
    command.service_factory.assert_called_once_with(command.store, command.settings)
    command.service.ensure.assert_called_once_with(date(2026, 11, 13), date(2026, 11, 14), ["re12", "re13"], True)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == command.service.ensure.return_value
    assert captured.err == ""
    command.crawler.assert_not_called()
    command.global_index.assert_not_called()


def test_default_scope_is_all_four_cafeterias_without_force(command, capsys):
    run("refresh", "--start", "2026-11-13", "--end", "2026-11-13")
    command.service.ensure.assert_called_once_with(date(2026, 11, 13), date(2026, 11, 13), ["re11", "re12", "re13", "re15"], False)
    assert json.loads(capsys.readouterr().out)["status"] == "ready"


def test_partial_safe_errors_are_reported_as_json_and_exit_one(command, capsys):
    expected = report(errors=[{"date": "2026-11-13", "code": "re12", "stage": "crawl", "error": "Timeout"}])
    command.service.ensure.return_value = expected
    with pytest.raises(SystemExit) as raised:
        run("refresh", "--start", "2026-11-13", "--end", "2026-11-14")
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected
    assert captured.err == ""


def test_no_api_key_keeps_collection_preview_report_without_attempting_global_index(command, capsys):
    command.settings.api_key = ""
    command.service.ensure.return_value = report(index_status="not_configured")
    run("refresh", "--start", "2026-11-13", "--end", "2026-11-14")
    actual = json.loads(capsys.readouterr().out)
    assert actual["index_status"] == "not_configured"
    assert actual["updated"] == 2
    command.service.ensure.assert_called_once()
    command.global_index.assert_not_called()


@pytest.mark.parametrize("arguments", [
    ["refresh"], ["refresh", "--start", "2026-11-13"], ["refresh", "--end", "2026-11-14"],
    ["refresh", "--start", "2026-11-31", "--end", "2026-12-01"],
    ["refresh", "--start", "20261113", "--end", "2026-11-14"],
    ["refresh", "--start", "2026-11-13", "--end", "2026-11-14", "--restaurants", "re99"],
])
def test_required_dates_and_invalid_arguments_fail_before_configuration_or_resources(command, capsys, arguments):
    with pytest.raises(SystemExit) as raised:
        run(*arguments)
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    command.settings_factory.assert_not_called()
    command.store_factory.assert_not_called()
    command.service_factory.assert_not_called()


@pytest.mark.parametrize("start,end,expected", [
    ("2026-11-14", "2026-11-13", "--end는 --start와 같거나 이후"),
    ("2026-11-13", "2026-12-14", "최대 31일"),
])
def test_range_order_and_inclusive_maximum_are_checked_before_database_creation(command, capsys, start, end, expected):
    with pytest.raises(SystemExit) as raised:
        run("refresh", "--start", start, "--end", end)
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert expected in captured.err
    command.store_factory.assert_not_called()
    command.service_factory.assert_not_called()


def test_exact_configured_maximum_is_allowed(command, capsys):
    command.settings.max_date_range_days = 3
    run("refresh", "--start", "2026-11-13", "--end", "2026-11-15")
    command.service.ensure.assert_called_once()
    assert json.loads(capsys.readouterr().out)["status"] == "ready"


def test_invalid_settings_have_an_argparse_error_before_database_creation(command, capsys):
    command.settings_factory.side_effect = ValueError("private configuration detail")
    with pytest.raises(SystemExit) as raised:
        run("refresh", "--start", "2026-11-13", "--end", "2026-11-14")
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert "수집 환경설정이 올바르지 않습니다" in captured.err
    assert "private configuration detail" not in captured.err
    assert captured.out == ""
    command.store_factory.assert_not_called()
    command.service_factory.assert_not_called()


def test_existing_status_command_still_uses_its_original_branch(command, capsys):
    expected = {"document_count": 76, "chunk_count": 450, "indexed_count": 450}
    command.store.stats.return_value = expected
    run("status")
    assert json.loads(capsys.readouterr().out) == expected
    command.service_factory.assert_not_called()

