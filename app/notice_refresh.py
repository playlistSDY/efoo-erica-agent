"""Refresh the public Korean notice board without reusing unverified policy."""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup
import requests

from app.assets import archive_assets
from app.indexing import index_notice, index_pending
from app.portal import embedded_array, parse_notice


BASE = "https://life.hanyang.ac.kr"
INDEX_URL = BASE + "/theme/pages/notice/index.php?board_lang=ko"
INDEX_SCOPE = "page:notice-index"


@dataclass
class _NoticeState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    completed: float = 0.0
    report: dict | None = None
    board_retry: datetime | None = None
    board_error: str | None = None
    detail_retries: dict = field(default_factory=dict)
    index_retries: dict = field(default_factory=dict)


class NoticeRefreshService:
    _guard = threading.Lock()
    _states = {}

    def __init__(self, store, settings, session_factory=requests.Session, indexer=index_pending, now=None):
        self.store, self.settings = store, settings
        self.session_factory, self.indexer = session_factory, indexer
        self._now = now or (lambda: datetime.now(timezone.utc))
        database = getattr(settings, "database_url", "")
        identity = "postgres:" + database if database else "sqlite:" + str(Path(settings.data_dir).resolve())
        with self._guard:
            self._state = self._states.setdefault(sha256(identity.encode()).hexdigest(), _NoticeState())

    def _utcnow(self):
        value = self._now()
        if not isinstance(value, datetime):
            raise ValueError("now는 datetime을 반환해야 합니다")
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

    def _fresh(self, row):
        try:
            fetched = datetime.fromisoformat(row["fetched_at"])
            fetched = fetched.replace(tzinfo=timezone.utc) if fetched.tzinfo is None else fetched.astimezone(timezone.utc)
            return timedelta(0) <= self._utcnow() - fetched < timedelta(hours=float(getattr(self.settings, "max_age_hours", 24)))
        except (KeyError, TypeError, ValueError):
            return False

    def _retry_at(self):
        return self._utcnow() + timedelta(minutes=float(getattr(self.settings, "refresh_retry_minutes", 5)))

    def _sources(self):
        with self.store.connection() as conn:
            rows = conn.execute("SELECT scope,code,name,url,fetched_at,payload FROM knowledge_sources WHERE scope='page:notice-index' OR scope LIKE 'notice:%'").fetchall()
        return {row["scope"]: {**dict(row), "payload": json.loads(row["payload"])} for row in rows}

    @staticmethod
    def _records(value):
        if not isinstance(value, list) or len(value) > 100:
            raise ValueError("공지 목록의 형식을 확인할 수 없습니다")
        seen = set()
        for record in value:
            if (not isinstance(record, dict) or isinstance(record.get("id"), bool) or not isinstance(record.get("id"), int)
                or record["id"] < 1 or record["id"] in seen or not isinstance(record.get("title"), str) or not record["title"].strip()):
                raise ValueError("공지 식별자와 제목을 확인할 수 없습니다")
            seen.add(record["id"])
        return value

    @staticmethod
    def _verified_url(actual, expected):
        try:
            parsed, wanted = urlparse(actual), urlparse(expected)
            query, requested = parse_qs(parsed.query), parse_qs(wanted.query)
            return (parsed.scheme == "https" and parsed.hostname == "life.hanyang.ac.kr" and parsed.port in {None, 443}
                and not parsed.username and not parsed.password and parsed.path == wanted.path
                and query.get("id") == requested.get("id") and query.get("board_lang") == ["ko"])
        except (TypeError, ValueError):
            return False

    def _get(self, session, url):
        response = session.get(url, timeout=(3, 12), allow_redirects=False,
            headers={"User-Agent": "EfooWebAcademicCrawler/1.0"})
        response.raise_for_status()
        if 300 <= response.status_code < 400 or not self._verified_url(response.url or url, url):
            raise ValueError("공식 한국어 공지 페이지를 확인할 수 없습니다")
        response.encoding = "utf-8"
        return response.text

    def _board(self, cached, get_session):
        if cached and self._fresh(cached) and cached["payload"].get("verified") is True:
            try:
                records = self._records(cached["payload"].get("records"))
                if cached["payload"].get("count") == len(records) and self._verified_url(cached["url"], INDEX_URL):
                    return cached, records, "fresh"
            except ValueError:
                pass
        if self._state.board_retry and self._utcnow() < self._state.board_retry:
            raise _Cooldown(self._state.board_error)
        html = self._get(get_session(), INDEX_URL)
        soup = BeautifulSoup(html, "html.parser")
        main = soup.select_one("main")
        if main is None or main.select_one("#notice-list-container, .notice-row-wrapper") is None:
            raise ValueError("공식 공지 목록의 구조가 변경되었습니다")
        records = self._records(embedded_array(html, "dbNotices"))
        fetched_at = self._utcnow().isoformat()
        assets = archive_assets(html, INDEX_URL, self.settings.data_dir, session=get_session())
        payload = {"verified": True, "records": records, "count": len(records), "verified_empty": not records,
            "source_url": INDEX_URL, "fetched_at": fetched_at, "language": "ko", "assets": assets,
            "scope": "public_korean_board_embedded_records",
            "page_metadata": {"title": soup.title.get_text(" ", strip=True) if soup.title else "",
                              "meta": [dict(tag.attrs) for tag in soup.select("meta")]}}
        folder = self.settings.data_dir / "portal"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "notice-index.html").write_text(html, encoding="utf-8")
        (folder / "notice-index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # This source proves the listing, including a verified empty list.
        # Aggregation does not need a generated embedding for the marker.
        self.store.replace_knowledge(INDEX_SCOPE, code="campus", name="공식 한국어 공지 게시판 목록", url=INDEX_URL,
            fetched_at=fetched_at, payload=payload, chunks=[])
        self._state.board_retry, self._state.board_error = None, None
        return {"scope": INDEX_SCOPE, "url": INDEX_URL, "fetched_at": fetched_at, "payload": payload}, records, "updated"

    def _detail(self, record, cached, get_session):
        identifier = record["id"]
        scope = f"notice:{identifier}"
        url = f"{BASE}/theme/pages/notice/detail.php?id={identifier}&board_lang=ko"
        target = {"id": identifier, "scope": scope, "status": "error", "usable": False,
                  "fetched_at": cached.get("fetched_at") if cached else None, "index_status": "not_needed"}
        previous_record=(cached or {}).get("payload",{}).get("raw_record",{})
        changed=any(key in record and key in previous_record and record[key]!=previous_record[key]
                    for key in ("title","content","updated_at"))
        if cached and not changed and self._fresh(cached) and self._verified_url(cached["url"], url) and cached["name"].strip()==record["title"].strip():
            return {**target, "status": "fresh", "usable": True, "index_status": "pending"}
        retry = self._state.detail_retries.get(scope)
        if retry and self._utcnow() < retry[0]:
            return {**target, "status": "cooldown", "error": {"id": identifier, "scope": scope, "stage": "crawl",
                "operation": "retry", "error": "RetryCooldown", "cause": retry[1], "retry_at": retry[0].isoformat()}}
        operation = "fetch_detail"
        try:
            html = self._get(get_session(), url)
            operation = "parse_detail"
            soup = BeautifulSoup(html, "html.parser")
            links = soup.select('link[rel="alternate"][hreflang="ko"][href], a.hy-lang-btn.is-active[lang="ko"][href]')
            if not links or any(not self._verified_url(urljoin(url, link["href"]), url) for link in links):
                raise ValueError("요청한 공지의 식별 정보를 확인할 수 없습니다")
            fetched_at = self._utcnow().isoformat()
            notice = parse_notice(html, url, fetched_at)
            if not notice["title"].strip() or not notice["text"].strip():
                raise ValueError("공지 본문을 확인할 수 없습니다")
            notice["raw_record"] = record
            operation = "archive_assets"
            notice["assets"] = archive_assets(html, url, self.settings.data_dir, session=get_session())
            operation = "persist_raw"
            folder = self.settings.data_dir / "portal"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"notice-{identifier}.html").write_text(html, encoding="utf-8")
            operation = "store_notice"
            index_notice(self.store, notice, identifier)
            self._state.detail_retries.pop(scope, None)
            return {**target, "status": "updated", "usable": True, "fetched_at": fetched_at, "index_status": "pending"}
        except Exception as exc:
            retry_at = self._retry_at()
            self._state.detail_retries[scope] = (retry_at, type(exc).__name__)
            return {**target, "error": {"id": identifier, "scope": scope, "stage": "crawl", "operation": operation,
                "error": type(exc).__name__, "retry_at": retry_at.isoformat()}}

    def _embed(self, report):
        targets = {target["scope"]: target for target in report["targets"] if target["usable"]}
        if not targets:
            return
        if self.indexer is None or not getattr(self.settings, "api_key", ""):
            for target in targets.values():
                target["index_status"] = "not_requested" if self.indexer is None else "not_configured"
            return
        eligible = []
        try:
            pending = {row.get("scope") for row in self.store.pending() if row.get("scope") in targets}
            for scope, target in targets.items():
                retry = self._state.index_retries.get(scope)
                if scope not in pending:
                    target["index_status"] = "indexed"
                    self._state.index_retries.pop(scope, None)
                elif retry and self._utcnow() < retry[0]:
                    target["index_status"] = "cooldown"
                    report["errors"].append({"id": target["id"], "scope": scope, "stage": "index", "operation": "retry",
                        "error": "RetryCooldown", "cause": retry[1], "retry_at": retry[0].isoformat()})
                else:
                    eligible.append(scope)
            if not eligible:
                return
            report["indexed"] += int(self.indexer(self.store, self.settings, scopes=eligible))
            remaining = {row.get("scope") for row in self.store.pending() if row.get("scope") in eligible}
            for scope in eligible:
                targets[scope]["index_status"] = "pending" if scope in remaining else "indexed"
                self._state.index_retries.pop(scope, None)
        except Exception as exc:
            retry_at = self._retry_at()
            for scope in eligible or targets:
                target = targets[scope]
                self._state.index_retries[scope] = (retry_at, type(exc).__name__)
                target["index_status"] = "error"
                report["errors"].append({"id": target["id"], "scope": scope, "stage": "index",
                    "operation": "embed" if eligible else "read_pending", "error": type(exc).__name__, "retry_at": retry_at.isoformat()})

    def ensure(self):
        started = time.monotonic()
        with self._state.lock:
            if self._state.report is not None and self._state.completed >= started:
                report = deepcopy(self._state.report)
                report["shared"] = True
                report["fresh"] += report["updated"]
                report["updated"], report["indexed"] = 0, 0
                for target in report["targets"]:
                    if target["status"] == "updated":
                        target["status"] = "fresh"
                if report["index"]["status"] == "updated":
                    report["index"]["status"] = "fresh"
                return report
            report = {"checked": 0, "updated": 0, "fresh": 0, "indexed": 0, "errors": [], "targets": [],
                      "listed_ids": None, "unusable_scopes": [], "index": {"scope": INDEX_SCOPE, "status": "error", "usable": False,
                          "source_url": INDEX_URL, "fetched_at": None, "count": None, "verified_empty": False}}
            session = None

            def get_session():
                nonlocal session
                if session is None:
                    session = self.session_factory()
                return session

            existing = {}
            try:
                existing = self._sources()
                board, records, status = self._board(existing.get(INDEX_SCOPE), get_session)
                report["listed_ids"] = [record["id"] for record in records]
                report["index"].update(status=status, usable=True, fetched_at=board["fetched_at"], count=len(records), verified_empty=not records)
                text = f"공식 한국어 게시판이 공개한 목록의 게시글은 {len(records)}개입니다. 학교 전체 공지의 부재를 뜻하지는 않습니다."
                text += "\n" + "\n".join(f"ID {record['id']}: {record['title']}" for record in records)
                report["index"]["source"] = {"id": "S1", "title": "공식 한국어 공지 게시판 목록", "url": INDEX_URL,
                    "fetched_at": board["fetched_at"], "date": board["fetched_at"][:10], "restaurant_code": "campus",
                    "doc_type": "page", "text": text, "images": [], "metadata": {"count": len(records), "verified_empty": not records}}
                for record in records:
                    target = self._detail(record, existing.get(f"notice:{record['id']}"), get_session)
                    error = target.pop("error", None)
                    if error:
                        report["errors"].append(error)
                    report["targets"].append(target)
                    report["checked"] += 1
                    if target["status"] in {"fresh", "updated"}:
                        report[target["status"]] += 1
                listed = {f"notice:{identifier}" for identifier in report["listed_ids"]}
                for scope, old in existing.items():
                    if scope.startswith("notice:") and scope not in listed:
                        report["targets"].append({"id": int(scope.split(":", 1)[1]), "scope": scope, "status": "not_listed",
                            "usable": False, "fetched_at": old["fetched_at"], "index_status": "not_needed"})
                self._embed(report)
            except Exception as exc:
                cooldown = isinstance(exc, _Cooldown)
                if not cooldown:
                    self._state.board_retry, self._state.board_error = self._retry_at(), type(exc).__name__
                report["index"]["status"] = "cooldown" if cooldown else "error"
                report["errors"].append({"scope": INDEX_SCOPE, "stage": "crawl", "operation": "retry" if cooldown else "refresh_index",
                    "error": "RetryCooldown" if cooldown else type(exc).__name__, "retry_at": self._state.board_retry.isoformat()})
                report["targets"] = [{"id": int(scope.split(":", 1)[1]), "scope": scope, "status": report["index"]["status"],
                    "usable": False, "fetched_at": old["fetched_at"], "index_status": "not_needed"}
                    for scope, old in existing.items() if scope.startswith("notice:")]
            finally:
                if session is not None:
                    session.close()
            report["unusable_scopes"] = sorted(target["scope"] for target in report["targets"] if not target["usable"])
            if not report["index"]["usable"]:
                report["unusable_scopes"].append(INDEX_SCOPE)
            report["status"] = ("partial" if report["index"]["usable"] else "error") if report["errors"] else "ready"
            statuses = {target["index_status"] for target in report["targets"] if target["usable"]}
            report["index_status"] = next((value for value in ("error", "cooldown", "pending", "not_configured", "not_requested", "indexed") if value in statuses), "not_needed")
            self._state.report, self._state.completed = deepcopy(report), time.monotonic()
            return report


class _Cooldown(Exception):
    pass
