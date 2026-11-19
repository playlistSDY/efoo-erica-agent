"""Bounded, single-flight refresh of requested official daily meal pages."""

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import threading
import time

from app.assets import archive_assets
from app.crawler import MealCrawler
from app.domain import RESTAURANTS
from app.indexing import build_chunks, index_pending, persist_raw


@dataclass
class _TargetState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    completed: float = 0.0
    last_result: dict | None = None
    crawl_retry_at: datetime | None = None
    crawl_error: str | None = None
    index_retry_at: datetime | None = None
    index_error: str | None = None
    touched: float = field(default_factory=time.monotonic)


class MealRefreshService:
    _registry_guard = threading.Lock()
    _states = {}
    _index_locks = {}

    def __init__(self, store, settings, crawler_factory=MealCrawler, indexer=index_pending, *, now=None):
        self.store, self.settings = store, settings
        self.crawler_factory, self.indexer = crawler_factory, indexer
        self._now = now or (lambda: datetime.now(timezone.utc))
        database = getattr(settings, "database_url", "")
        identity = "postgres:" + database if database else "sqlite:" + str(Path(settings.data_dir).resolve())
        # A database URL may contain credentials; never retain it in reports or
        # lock names that could be surfaced to a caller.
        self._backend = sha256(identity.encode()).hexdigest()
        with self._registry_guard:
            self._index_lock = self._index_locks.setdefault(self._backend, threading.Lock())

    def _utcnow(self):
        value = self._now()
        if not isinstance(value, datetime):
            raise ValueError("now는 UTC datetime을 반환해야 합니다")
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

    def _state(self, day, code):
        key = (self._backend, day, code)
        with self._registry_guard:
            # Success entries only coordinate overlapping requests. Keep a
            # bounded cache while retaining recent retry cooldowns.
            if len(self._states) > 4096:
                cutoff = time.monotonic() - 3600
                for old_key, old in list(self._states.items()):
                    if old.touched < cutoff and not old.lock.locked():
                        self._states.pop(old_key, None)
            state = self._states.setdefault(key, _TargetState())
            state.touched = time.monotonic()
            return state

    @staticmethod
    def _timestamp(value):
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    def _is_fresh(self, collection, now):
        fetched = self._timestamp(collection.get("fetched_at"))
        if fetched is None:
            return False
        if not collection.get("meals"):
            ttl = timedelta(minutes=float(getattr(self.settings, "meal_empty_refresh_minutes", 60)))
        else:
            ttl = timedelta(hours=float(getattr(self.settings, "meal_refresh_hours", 6)))
        # Store's freshness gate must not reject a snapshot that this service
        # just declared fresh, even if the configured collection TTL is longer.
        ttl = min(ttl, timedelta(hours=float(getattr(self.settings, "max_age_hours", 24))))
        return timedelta(0) <= now - fetched < ttl

    def _retry_time(self, now):
        return now + timedelta(minutes=float(getattr(self.settings, "refresh_retry_minutes", 5)))

    @staticmethod
    def _base(day, code, collection=None):
        return {"date": day, "code": code, "status": "error", "usable": False,
                "empty": not bool(collection["meals"]) if collection is not None else None,
                "fetched_at": collection.get("fetched_at") if collection is not None else None,
                "index_status": "not_needed"}

    def _collect(self, target, code, state, get_crawler, request_started, force):
        day = target.isoformat()
        with state.lock:
            # A refresh completed after this ensure call began. Share its
            # outcome even for force=True or zero TTL; do not issue duplicate
            # requests for overlapping callers.
            if state.last_result is not None and state.completed >= request_started:
                result = {**state.last_result, "shared": True}
                if result["usable"]:
                    result["status"] = "fresh"
                return result
            now = self._utcnow()
            if state.crawl_retry_at is not None and now < state.crawl_retry_at:
                return {**self._base(day, code), "status": "cooldown", "retry_at": state.crawl_retry_at.isoformat(),
                        "error": {"date": day, "code": code, "stage": "crawl", "operation": "retry",
                                  "error": "RetryCooldown", "cause": state.crawl_error,
                                  "retry_at": state.crawl_retry_at.isoformat()}}
            operation, previous = "read_cache", None
            try:
                rows = self.store.collections(day, code)
                previous = rows[0] if rows else None
                if previous is not None and not force and self._is_fresh(previous, now):
                    return {**self._base(day, code, previous), "status": "fresh", "usable": True,
                            "index_status": "pending"}
                operation = "fetch"
                crawler = get_crawler()
                result = crawler.fetch(code, target)
                if result.code != code or result.date != day:
                    raise ValueError("요청 날짜 또는 식당과 수집 결과가 다릅니다")
                operation = "archive_assets"
                assets = archive_assets(result.html, result.source_url, self.settings.data_dir, session=crawler.session)
                result = replace(result, assets=assets)
                operation = "persist_raw"
                persist_raw(result, self.settings)
                operation = "replace_collection"
                # Never iterate weekly_meals here: each target must have its
                # own verified official response and photos.
                self.store.replace_collection(result, build_chunks(result))
                state.crawl_retry_at, state.crawl_error = None, None
                outcome = {**self._base(day, code), "status": "updated", "usable": True,
                           "empty": not bool(result.meals), "fetched_at": result.fetched_at,
                           "index_status": "pending"}
            except Exception as exc:
                state.crawl_retry_at = self._retry_time(now)
                state.crawl_error = type(exc).__name__
                outcome = {**self._base(day, code, previous), "retry_at": state.crawl_retry_at.isoformat(),
                           "error": {"date": day, "code": code, "stage": "crawl", "operation": operation,
                                     "error": state.crawl_error, "retry_at": state.crawl_retry_at.isoformat()}}
            state.last_result, state.completed = outcome, time.monotonic()
            return {**outcome}

    def _pending_targets(self, requested):
        return {(row["day"], row["code"]) for row in self.store.pending()
                if row.get("doc_type", "menu") == "menu" and (row["day"], row["code"]) in requested}

    def _index(self, targets, states, report):
        valid = {(item["date"], item["code"]): item for item in targets if item["usable"]}
        if not valid:
            return
        if not getattr(self.settings, "api_key", ""):
            for target in valid.values():
                target["index_status"] = "not_configured"
            return
        with self._index_lock:
            try:
                pending = self._pending_targets(valid)
            except Exception as exc:
                for key, target in valid.items():
                    target["index_status"] = "error"
                    report["errors"].append({"date": key[0], "code": key[1], "stage": "index",
                                             "operation": "read_pending", "error": type(exc).__name__})
                return
            eligible = []
            now = self._utcnow()
            for key, target in valid.items():
                state = states[key]
                if key not in pending:
                    target["index_status"] = "indexed"
                    state.index_retry_at, state.index_error = None, None
                elif state.index_retry_at is not None and now < state.index_retry_at:
                    target["index_status"] = "cooldown"
                    target["index_retry_at"] = state.index_retry_at.isoformat()
                    report["errors"].append({"date": key[0], "code": key[1], "stage": "index",
                        "operation": "retry", "error": "RetryCooldown", "cause": state.index_error,
                        "retry_at": state.index_retry_at.isoformat()})
                else:
                    eligible.append(key)
            if not eligible:
                return
            try:
                report["indexed"] += int(self.indexer(self.store, self.settings, menu_targets=eligible))
                remaining = self._pending_targets(eligible)
                for key in eligible:
                    valid[key]["index_status"] = "pending" if key in remaining else "indexed"
                    states[key].index_retry_at, states[key].index_error = None, None
            except Exception as exc:
                retry_at = self._retry_time(self._utcnow())
                for key in eligible:
                    state = states[key]
                    state.index_retry_at, state.index_error = retry_at, type(exc).__name__
                    valid[key]["index_status"] = "error"
                    valid[key]["index_retry_at"] = retry_at.isoformat()
                    report["errors"].append({"date": key[0], "code": key[1], "stage": "index",
                        "operation": "embed", "error": state.index_error, "retry_at": retry_at.isoformat()})

    def ensure(self, start: date, end: date, restaurant_codes=None, force=False):
        request_started = time.monotonic()
        if any(not isinstance(value, date) or isinstance(value, datetime) for value in (start, end)):
            raise ValueError("조회 범위는 datetime.date여야 합니다")
        days = (end - start).days + 1
        if not 1 <= days <= int(getattr(self.settings, "max_date_range_days", 31)):
            raise ValueError("조회 날짜 범위가 올바르지 않습니다")
        if isinstance(restaurant_codes, str):
            raise ValueError("식당 코드는 목록으로 전달해 주세요")
        codes = list(dict.fromkeys(RESTAURANTS if restaurant_codes is None else restaurant_codes))
        if not codes or any(code not in RESTAURANTS for code in codes):
            raise ValueError("알 수 없는 식당 코드입니다")
        report = {"start_date": start.isoformat(), "end_date": end.isoformat(),
                  "dates": [(start + timedelta(days=offset)).isoformat() for offset in range(days)],
                  "checked": 0, "updated": 0, "fresh": 0, "skipped": 0,
                  "errors": [], "indexed": 0, "targets": []}
        crawler = None

        def get_crawler():
            nonlocal crawler
            if crawler is None:
                crawler = self.crawler_factory()
            return crawler

        states = {}
        try:
            for offset in range(days):
                target = start + timedelta(days=offset)
                for code in codes:
                    key = (target.isoformat(), code)
                    state = states[key] = self._state(*key)
                    outcome = self._collect(target, code, state, get_crawler, request_started, force)
                    error = outcome.pop("error", None)
                    if error:
                        report["errors"].append(error)
                    report["targets"].append(outcome)
                    report["checked"] += 1
                    if outcome["status"] == "updated":
                        report["updated"] += 1
                    elif outcome["status"] == "fresh":
                        report["fresh"] += 1
                    elif outcome["status"] == "cooldown":
                        report["skipped"] += 1
        finally:
            if crawler is not None:
                crawler.close()
        self._index(report["targets"], states, report)
        report["status"] = ("partial" if any(item["usable"] for item in report["targets"]) else "error") if report["errors"] else "ready"
        index_states = {item["index_status"] for item in report["targets"] if item["usable"]}
        report["index_status"] = next((value for value in ("error", "cooldown", "pending", "not_configured", "indexed") if value in index_states), "not_needed")
        return report
