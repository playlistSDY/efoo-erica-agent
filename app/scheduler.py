"""A process-local, stoppable scheduler for public meal refreshes.

Use one API worker with this scheduler enabled. Multiple worker processes need
an external scheduler or a distributed lease rather than one thread per worker.
"""

from datetime import date, datetime, timedelta
from threading import Event, Lock, Thread, current_thread
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")


def _today_kst():
    return datetime.now(KST).date()


class MealScheduler:
    """Periodically call ``refresh.ensure(start, end, force=False)``.

    Starting performs the first check in the background immediately. Subsequent
    checks run one interval after the previous check finishes. ``stop`` wakes
    the wait immediately; an already-running refresh is allowed to finish.
    """

    def __init__(self, refresh, settings, today_fn=_today_kst):
        self.refresh = refresh
        self.settings = settings
        self.today_fn = today_fn
        self._stop_event = Event()
        self._state_lock = Lock()
        self._run_lock = Lock()
        self._thread = None
        self._in_progress = False
        self._last_started_at = None
        self._last_completed_at = None
        self._next_run_at = None
        self._last_error = None
        self._last_summary = None

    @staticmethod
    def _now():
        return datetime.now(KST)

    def start(self):
        with self._state_lock:
            if not self.settings.schedule_enabled:
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._next_run_at = self._now().isoformat()
            self._thread = Thread(target=self._loop, name="meal-refresh", daemon=True)
            self._thread.start()

    def stop(self):
        with self._state_lock:
            self._stop_event.set()
            self._next_run_at = None
            thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=1)

    def status(self):
        with self._state_lock:
            running = self._thread is not None and self._thread.is_alive()
            return {
                "enabled": self.settings.schedule_enabled,
                "running": running,
                "in_progress": self._in_progress,
                "stopping": running and self._stop_event.is_set(),
                "interval_minutes": self.settings.schedule_interval_minutes,
                "start_offset_days": self.settings.schedule_start_offset_days,
                "days": self.settings.schedule_days,
                "last_started_at": self._last_started_at,
                "last_completed_at": self._last_completed_at,
                "next_run_at": self._next_run_at,
                "last_error": self._last_error,
                "last_summary": self._last_summary,
            }

    def run_once(self):
        """Return the refresh result, or None after an error/overlapping call."""
        if not self._run_lock.acquire(blocking=False):
            return None
        with self._state_lock:
            self._in_progress = True
            self._last_started_at = self._now().isoformat()
            self._last_error = None
        try:
            today = self.today_fn()
            if not isinstance(today, date) or isinstance(today, datetime):
                raise TypeError("today_fn must return a date")
            start = today + timedelta(days=self.settings.schedule_start_offset_days)
            end = start + timedelta(days=self.settings.schedule_days - 1)
            report=self.refresh.ensure(start, end, force=False)
            if isinstance(report,dict):
                with self._state_lock:
                    self._last_summary={key:report.get(key) for key in (
                        "start_date","end_date","status","checked","updated","fresh","skipped","indexed","index_status")}
                    self._last_summary["error_count"]=len(report.get("errors",[]))
                    if report.get("errors"):
                        self._last_error="RefreshIncomplete"
            return report
        except Exception as error:
            # Error messages may include URLs, credentials or response bodies.
            # Only the exception class is safe operational status metadata.
            with self._state_lock:
                self._last_error = type(error).__name__
            return None
        finally:
            with self._state_lock:
                self._in_progress = False
                self._last_completed_at = self._now().isoformat()
            self._run_lock.release()

    def _loop(self):
        try:
            while not self._stop_event.is_set():
                self.run_once()
                with self._state_lock:
                    if self._stop_event.is_set():
                        break
                    interval = self.settings.schedule_interval_minutes * 60
                    self._next_run_at = (self._now() + timedelta(seconds=interval)).isoformat()
                if self._stop_event.wait(interval):
                    break
        finally:
            with self._state_lock:
                self._next_run_at = None
