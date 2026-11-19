"""Refresh only caller-selected public facility profiles with safe provenance."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse

import requests

from app.assets import archive_assets
from app.crawler import parse_facility_html
from app.domain import RESTAURANTS
from app.indexing import index_facility, index_pending


BASE = "https://life.hanyang.ac.kr"


@dataclass
class _FacilityState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    completed: float = 0.0
    result: dict | None = None
    retry_at: datetime | None = None
    error: str | None = None
    index_retry_at: datetime | None = None
    index_error: str | None = None


class FacilityRefreshService:
    _guard = threading.Lock()
    _states = {}
    _index_locks = {}

    def __init__(self, store, settings, *, session_factory=requests.Session, indexer=index_pending, now=None):
        self.store, self.settings = store, settings
        self.session_factory, self.indexer = session_factory, indexer
        self._now = now or (lambda: datetime.now(timezone.utc))
        database = getattr(settings, "database_url", "")
        identity = "postgres:" + database if database else "sqlite:" + str(Path(settings.data_dir).resolve())
        self._backend = sha256(identity.encode()).hexdigest()
        with self._guard:
            self._index_lock = self._index_locks.setdefault(self._backend, threading.Lock())

    def _utcnow(self):
        value = self._now()
        if not isinstance(value, datetime):
            raise ValueError("now는 datetime을 반환해야 합니다")
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

    def _state(self, identifier):
        with self._guard:
            return self._states.setdefault((self._backend, identifier), _FacilityState())

    @staticmethod
    def _code(identifier):
        return next((code for code, info in RESTAURANTS.items() if info["facility_id"] == str(identifier)), f"fac_{identifier}")

    @staticmethod
    def _url(record):
        page = "detail.php" if record["category"] == "구내식당" else "shop-detail.php"
        return f"{BASE}/theme/pages/facilities/{page}?id={record['id']}"

    @staticmethod
    def _verified_url(actual, expected):
        try:
            parsed, requested = urlparse(actual), urlparse(expected)
            return (parsed.scheme == "https" and parsed.hostname == "life.hanyang.ac.kr"
                and parsed.port in {None, 443} and not parsed.username and not parsed.password
                and parsed.path == requested.path
                and parse_qs(parsed.query).get("id") == parse_qs(requested.query).get("id"))
        except (TypeError, ValueError):
            return False

    def _existing(self, identifier):
        # Avoid loading every profile's raw payload just to check one timestamp.
        with self.store.connection() as conn:
            row = conn.execute(self.store.sql(
                "SELECT scope,code,name,url,fetched_at FROM knowledge_sources WHERE scope=?"),
                (f"facility:{identifier}",)).fetchone()
        return dict(row) if row is not None else None

    def _fresh(self, existing, record, now):
        if existing is None or existing["code"] != self._code(record["id"]) or not self._verified_url(existing["url"], self._url(record)):
            return False
        try:
            fetched = datetime.fromisoformat(existing["fetched_at"])
        except (TypeError, ValueError):
            return False
        fetched = fetched.replace(tzinfo=timezone.utc) if fetched.tzinfo is None else fetched.astimezone(timezone.utc)
        return timedelta(0) <= now - fetched < timedelta(hours=float(getattr(self.settings, "max_age_hours", 24)))

    def _retry_time(self):
        return self._utcnow() + timedelta(minutes=float(getattr(self.settings, "refresh_retry_minutes", 5)))

    @staticmethod
    def _target(record, *, status="error", usable=False, fetched_at=None):
        return {"id": record["id"], "code": FacilityRefreshService._code(record["id"]),
                "status": status, "usable": usable, "fetched_at": fetched_at, "index_status": "not_needed"}

    def _refresh_one(self, record, state, get_session, started, force):
        with state.lock:
            if state.result is not None and state.completed >= started:
                shared = {**state.result, "shared": True}
                if shared["usable"]:
                    shared["status"] = "fresh"
                return shared
            if state.retry_at is not None and self._utcnow() < state.retry_at:
                return {**self._target(record), "status": "cooldown", "retry_at": state.retry_at.isoformat(),
                    "error": {"id": record["id"], "code": self._code(record["id"]), "stage": "crawl",
                        "operation": "retry", "error": "RetryCooldown", "cause": state.error,
                        "retry_at": state.retry_at.isoformat()}}
            operation, existing = "read_cache", None
            try:
                existing = self._existing(record["id"])
                if not force and self._fresh(existing, record, self._utcnow()):
                    return {**self._target(record, status="fresh", usable=True, fetched_at=existing["fetched_at"]),
                            "index_status": "pending"}
                session = get_session()
                url = self._url(record)
                operation = "fetch"
                response = session.get(url, timeout=(3, 12), allow_redirects=False,
                    headers={"User-Agent": "EfooWebAcademicCrawler/1.0"})
                response.raise_for_status()
                if 300 <= response.status_code < 400 or not self._verified_url(response.url or url, url):
                    raise ValueError("공식 시설 상세 페이지를 확인할 수 없습니다")
                response.encoding = "utf-8"
                html = response.text
                operation = "parse"
                facility = parse_facility_html(html, expected_facility_id=str(record["id"]), base_url=BASE)
                facility["facility_id"] = record["id"]
                operation = "archive_assets"
                assets = archive_assets(html, url, self.settings.data_dir, session=session)
                fetched_at = self._utcnow().isoformat()
                operation = "persist_raw"
                folder = self.settings.data_dir / "portal"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"facility-{record['id']}.html").write_text(html, encoding="utf-8")
                operation = "index_facility"
                index_facility(self.store, facility, assets, fetched_at, url, record)
                state.retry_at, state.error = None, None
                outcome = {**self._target(record, status="updated", usable=True, fetched_at=fetched_at),
                           "index_status": "pending"}
            except Exception as exc:
                state.retry_at, state.error = self._retry_time(), type(exc).__name__
                outcome = {**self._target(record, fetched_at=existing["fetched_at"] if existing else None),
                    "retry_at": state.retry_at.isoformat(),
                    "error": {"id": record["id"], "code": self._code(record["id"]), "stage": "crawl",
                        "operation": operation, "error": state.error, "retry_at": state.retry_at.isoformat()}}
            state.result, state.completed = outcome, time.monotonic()
            return {**outcome}

    def _pending_scopes(self, requested):
        return {row.get("scope") for row in self.store.pending() if row.get("scope") in requested}

    def _index(self, report, states):
        valid = {f"facility:{target['id']}": target for target in report["targets"] if target["usable"]}
        if not valid:
            return
        if self.indexer is None or not getattr(self.settings, "api_key", ""):
            for target in valid.values():
                target["index_status"] = "not_requested" if self.indexer is None else "not_configured"
            return
        with self._index_lock:
            eligible = []
            try:
                pending = self._pending_scopes(valid)
                for scope, target in valid.items():
                    state = states[target["id"]]
                    if scope not in pending:
                        target["index_status"] = "indexed"
                        state.index_retry_at, state.index_error = None, None
                    elif state.index_retry_at is not None and self._utcnow() < state.index_retry_at:
                        target["index_status"] = "cooldown"
                        report["errors"].append({"id": target["id"], "code": target["code"], "stage": "index",
                            "operation": "retry", "error": "RetryCooldown", "cause": state.index_error,
                            "retry_at": state.index_retry_at.isoformat()})
                    else:
                        eligible.append(scope)
                if not eligible:
                    return
                report["indexed"] += int(self.indexer(self.store, self.settings, scopes=eligible))
                remaining = self._pending_scopes(eligible)
                for scope in eligible:
                    target = valid[scope]
                    target["index_status"] = "pending" if scope in remaining else "indexed"
                    states[target["id"]].index_retry_at = None
                    states[target["id"]].index_error = None
            except Exception as exc:
                retry_at = self._retry_time()
                for scope in eligible or valid:
                    target = valid[scope]
                    state = states[target["id"]]
                    state.index_retry_at, state.index_error = retry_at, type(exc).__name__
                    target["index_status"] = "error"
                    report["errors"].append({"id": target["id"], "code": target["code"], "stage": "index",
                        "operation": "embed" if eligible else "read_pending", "error": state.index_error,
                        "retry_at": retry_at.isoformat()})

    def ensure(self, records: list[dict], force=False):
        started = time.monotonic()
        if not isinstance(records, list) or len(records) > 150:
            raise ValueError("공식 시설 목록은 한 번에 최대 150개까지 확인할 수 있습니다")
        identifiers = set()
        for record in records:
            if (not isinstance(record, dict) or isinstance(record.get("id"), bool) or not isinstance(record.get("id"), int)
                or record["id"] < 1 or record["id"] in identifiers or not isinstance(record.get("name"), str)
                or not record["name"].strip() or not isinstance(record.get("category"), str) or not record["category"].strip()):
                raise ValueError("공식 시설의 식별자와 이름, 분류를 확인해 주세요")
            identifiers.add(record["id"])
        report = {"checked": 0, "updated": 0, "fresh": 0, "errors": [], "targets": [], "indexed": 0}
        session = None

        def get_session():
            nonlocal session
            if session is None:
                session = self.session_factory()
            return session

        states = {}
        try:
            for record in records:
                state = states[record["id"]] = self._state(record["id"])
                outcome = self._refresh_one(record, state, get_session, started, force)
                error = outcome.pop("error", None)
                if error:
                    report["errors"].append(error)
                report["targets"].append(outcome)
                report["checked"] += 1
                if outcome["status"] in {"fresh", "updated"}:
                    report[outcome["status"]] += 1
        finally:
            if session is not None:
                session.close()
        self._index(report, states)
        report["status"] = ("partial" if any(target["usable"] for target in report["targets"]) else "error") if report["errors"] else "ready"
        statuses = {target["index_status"] for target in report["targets"] if target["usable"]}
        report["index_status"] = next((value for value in ("error", "cooldown", "pending", "not_configured", "not_requested", "indexed") if value in statuses), "not_needed")
        return report
