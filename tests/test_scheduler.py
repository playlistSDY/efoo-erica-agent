"""Public refresh scheduling regressions; no crawler, API or database calls."""

from datetime import date, datetime, timedelta
from threading import Event, Thread
from unittest.mock import Mock

import pytest

from app.scheduler import MealScheduler
from app.settings import Settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    defaults = {
        "MAX_DATE_RANGE_DAYS": "31", "MEAL_REFRESH_HOURS": "6",
        "MEAL_EMPTY_REFRESH_MINUTES": "60", "REFRESH_RETRY_MINUTES": "5",
        "SCHEDULE_ENABLED": "true", "SCHEDULE_INTERVAL_MINUTES": "60",
        "SCHEDULE_START_OFFSET_DAYS": "0", "SCHEDULE_DAYS": "14",
    }
    for name, value in defaults.items():
        monkeypatch.setenv(name, value)
    return Settings(data_dir=tmp_path, database_url="", api_key="")


def test_default_schedule_covers_today_through_day_thirteen(settings):
    refresh = Mock()
    refresh.ensure.return_value = {"updated": 2}
    scheduler = MealScheduler(refresh, settings, today_fn=lambda: date(2026, 11, 13))

    assert scheduler.run_once() == {"updated": 2}
    refresh.ensure.assert_called_once_with(date(2026, 11, 13), date(2026, 11, 26), force=False)
    state = scheduler.status()
    assert not state["running"] and not state["in_progress"]
    assert state["last_error"] is None
    started = datetime.fromisoformat(state["last_started_at"])
    completed = datetime.fromisoformat(state["last_completed_at"])
    assert started <= completed and started.utcoffset() == timedelta(hours=9)


def test_schedule_resolves_today_each_time_and_honors_offset(settings):
    settings.schedule_start_offset_days = 2
    settings.schedule_days = 3
    days = iter([date(2026, 12, 31), date(2027, 1, 1)])
    refresh = Mock()
    scheduler = MealScheduler(refresh, settings, today_fn=lambda: next(days))
    scheduler.run_once()
    scheduler.run_once()
    assert refresh.ensure.call_args_list[0].args == (date(2027, 1, 2), date(2027, 1, 4))
    assert refresh.ensure.call_args_list[1].args == (date(2027, 1, 3), date(2027, 1, 5))


def test_overlapping_runs_never_start_a_second_refresh(settings):
    entered, release = Event(), Event()

    def ensure(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return "done"

    refresh = Mock(ensure=Mock(side_effect=ensure))
    scheduler = MealScheduler(refresh, settings, today_fn=lambda: date(2026, 11, 13))
    worker = Thread(target=scheduler.run_once)
    worker.start()
    try:
        assert entered.wait(1)
        assert scheduler.status()["in_progress"]
        assert scheduler.run_once() is None
        assert refresh.ensure.call_count == 1
    finally:
        release.set()
        worker.join(timeout=2)
    assert not worker.is_alive() and not scheduler.status()["in_progress"]


def test_error_status_does_not_expose_exception_message_and_next_run_recovers(settings):
    refresh = Mock()
    refresh.ensure.side_effect = [RuntimeError("secret-key and private request"), "recovered"]
    scheduler = MealScheduler(refresh, settings, today_fn=lambda: date(2026, 11, 13))
    assert scheduler.run_once() is None
    state = scheduler.status()
    assert state["last_error"] == "RuntimeError"
    assert "secret-key" not in str(state) and "private request" not in str(state)
    assert state["last_completed_at"] is not None
    assert scheduler.run_once() == "recovered"
    assert scheduler.status()["last_error"] is None


def test_loop_continues_after_failure_and_waits_configured_interval(settings):
    class TwoIntervals:
        def __init__(self):
            self.waits = []

        def is_set(self):
            return False

        def wait(self, seconds):
            self.waits.append(seconds)
            return len(self.waits) == 2

    refresh = Mock()
    refresh.ensure.side_effect = [OSError("temporary failure"), "success"]
    scheduler = MealScheduler(refresh, settings, today_fn=lambda: date(2026, 11, 13))
    event = TwoIntervals()
    scheduler._stop_event = event
    scheduler._loop()
    assert refresh.ensure.call_count == 2
    assert event.waits == [3600, 3600]
    assert scheduler.status()["last_error"] is None
    assert scheduler.status()["next_run_at"] is None


def test_disabled_start_does_not_run_refresh(settings):
    settings.schedule_enabled = False
    refresh = Mock()
    scheduler = MealScheduler(refresh, settings)
    scheduler.start()
    scheduler.stop()
    refresh.ensure.assert_not_called()
    assert scheduler.status()["enabled"] is False
    assert scheduler.status()["running"] is False


def test_start_is_idempotent_and_stop_wakes_interval_wait(settings):
    called = Event()
    refresh = Mock(ensure=Mock(side_effect=lambda *args, **kwargs: called.set()))
    scheduler = MealScheduler(refresh, settings)
    scheduler.start()
    try:
        assert called.wait(1)
        first_thread = scheduler._thread
        scheduler.start()
        assert scheduler._thread is first_thread
        assert scheduler.status()["running"]
    finally:
        scheduler.stop()
    assert not scheduler.status()["running"]
    assert scheduler.status()["next_run_at"] is None
    assert refresh.ensure.call_count == 1


def test_stopping_during_refresh_prevents_another_run(settings):
    entered, release = Event(), Event()

    def ensure(*args, **kwargs):
        entered.set()
        assert release.wait(3)

    refresh = Mock(ensure=Mock(side_effect=ensure))
    scheduler = MealScheduler(refresh, settings)
    scheduler.start()
    try:
        assert entered.wait(1)
        scheduler.stop()
        assert scheduler.status()["stopping"]
        assert scheduler.status()["next_run_at"] is None
    finally:
        release.set()
        scheduler.stop()
    assert not scheduler.status()["running"]
    assert refresh.ensure.call_count == 1


def test_scheduler_can_restart_after_stop(settings):
    called = Event()
    refresh = Mock(ensure=Mock(side_effect=lambda *args, **kwargs: called.set()))
    scheduler = MealScheduler(refresh, settings)
    for _ in range(2):
        called.clear()
        scheduler.start()
        try:
            assert called.wait(1)
        finally:
            scheduler.stop()
    assert refresh.ensure.call_count == 2


@pytest.mark.parametrize("name,value", [
    ("max_date_range_days", 0), ("schedule_interval_minutes", 0),
    ("schedule_days", 0), ("meal_refresh_hours", -1),
    ("meal_empty_refresh_minutes", -1), ("refresh_retry_minutes", -1),
    ("schedule_start_offset_days", True), ("schedule_enabled", "true"),
    ("schedule_interval_minutes", True), ("schedule_days", 1.5),
])
def test_invalid_schedule_settings_fail_before_making_data_directory(tmp_path, name, value):
    target = tmp_path / "invalid-settings"
    with pytest.raises(ValueError, match=name):
        Settings(data_dir=target, database_url="", api_key="", **{name: value})
    assert not target.exists()


def test_schedule_range_must_fit_max_date_range(settings, tmp_path):
    with pytest.raises(ValueError, match="schedule_days"):
        Settings(data_dir=tmp_path, database_url="", api_key="", max_date_range_days=5, schedule_days=6)


@pytest.mark.parametrize("raw,expected", [("true", True), ("1", True), (" YES ", True),
    ("false", False), ("0", False), ("off", False)])
def test_schedule_boolean_environment(settings, tmp_path, monkeypatch, raw, expected):
    monkeypatch.setenv("SCHEDULE_ENABLED", raw)
    actual = Settings(data_dir=tmp_path, database_url="", api_key="")
    assert actual.schedule_enabled is expected


def test_invalid_boolean_environment_is_rejected(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("SCHEDULE_ENABLED", "maybe")
    with pytest.raises(ValueError, match="SCHEDULE_ENABLED"):
        Settings(data_dir=tmp_path, database_url="", api_key="")


def test_refresh_and_schedule_environment_values(settings, tmp_path, monkeypatch):
    values = {"MAX_DATE_RANGE_DAYS": "20", "MEAL_REFRESH_HOURS": "0",
        "MEAL_EMPTY_REFRESH_MINUTES": "15", "REFRESH_RETRY_MINUTES": "2",
        "SCHEDULE_INTERVAL_MINUTES": "30", "SCHEDULE_START_OFFSET_DAYS": "1",
        "SCHEDULE_DAYS": "20"}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    actual = Settings(data_dir=tmp_path, database_url="", api_key="")
    assert (actual.max_date_range_days, actual.meal_refresh_hours, actual.meal_empty_refresh_minutes,
        actual.refresh_retry_minutes, actual.schedule_interval_minutes,
        actual.schedule_start_offset_days, actual.schedule_days) == (20, 0, 15, 2, 30, 1, 20)
