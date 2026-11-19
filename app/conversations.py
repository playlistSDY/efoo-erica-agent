"""Persistent browser-owned conversations on the application's database.

Only a SHA256 digest of the opaque browser token is persisted. The API supplies
the token through its cookie and passes public answer/request metadata here;
credentials and HTTP headers must never be included in response payloads.
"""

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import re
import secrets
import threading
from uuid import uuid4
from zoneinfo import ZoneInfo

from app.text_normalization import normalize_korean_input


_SEOUL = ZoneInfo("Asia/Seoul")
_EXPLICIT_DAY = re.compile(r"20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}월\s*\d{1,2}일")
_RELATIVE_DAY = re.compile(
    r"(?<![A-Za-z0-9가-힣])(?P<word>모레|내일|어제|오늘)"
    r"(?=$|[\s,.;:!?/()\[\]{}~·]|[은는도에의와과랑]|까지|부터|학생|교직원|창의|창업|기숙사|"
    r"학식|식당|메뉴|아침|점심|저녁|조식|중식|석식|가격|운영|몇\s*시)"
)


def _historical_content(content, created_at, reference_day):
    if reference_day is None or _EXPLICIT_DAY.search(content):
        return content
    try:
        created = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        return content
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    original_day = created.astimezone(_SEOUL).date()
    if original_day == reference_day:
        return content
    # IME compatibility jamo and NFD syllables must resolve against the date
    # of the old request too. Normalize only this internal history copy, and
    # retain the exact original for explicit dates or unrelated names.
    normalized = normalize_korean_input(content)
    if _EXPLICIT_DAY.search(normalized) or not _RELATIVE_DAY.search(normalized):
        return content
    offsets = {"어제": -1, "오늘": 0, "내일": 1, "모레": 2}
    return _RELATIVE_DAY.sub(lambda match: (original_day + timedelta(days=offsets[match["word"]])).isoformat(), normalized)


def _now():
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    # Share locks between service instances in the same process. Entries are
    # removed when the final holder/waiter exits, so abandoned chats do not
    # accumulate locks indefinitely. Database transactions protect other workers.
    _lock_guard = threading.Lock()
    _locks = {}

    def __init__(self, store):
        self.store = store
        with store.connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS chat_sessions (
              token_hash TEXT PRIMARY KEY,
              current_conversation_id TEXT,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
              id TEXT PRIMARY KEY, owner_hash TEXT NOT NULL,
              title TEXT NOT NULL, created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
              id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
              turn_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
              role TEXT NOT NULL CHECK(role IN ('user','assistant')),
              content TEXT NOT NULL, payload TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(conversation_id,turn_id,role),
              UNIQUE(conversation_id,ordinal))""")
            conn.execute("CREATE INDEX IF NOT EXISTS conversations_owner_updated ON conversations(owner_hash,updated_at)")

    def _execute(self, conn, query, args=()):
        return conn.execute(self.store.sql(query), args)

    def _write_start(self, conn):
        if not self.store.postgres:
            conn.execute("BEGIN IMMEDIATE")

    def _session(self, conn, owner_hash, *, lock=False):
        suffix = " FOR UPDATE" if lock and self.store.postgres else ""
        row = self._execute(conn, "SELECT * FROM chat_sessions WHERE token_hash=?" + suffix, (owner_hash,)).fetchone()
        if row is None:
            raise KeyError("대화 세션을 찾을 수 없습니다")
        return dict(row)

    def _owned(self, conn, owner_hash, conversation_id, *, lock=False):
        suffix = " FOR UPDATE" if lock and self.store.postgres else ""
        row = self._execute(conn, "SELECT * FROM conversations WHERE id=? AND owner_hash=?" + suffix,
                            (conversation_id, owner_hash)).fetchone()
        if row is None:
            raise KeyError("대화를 찾을 수 없습니다")
        return dict(row)

    def _make_current(self, conn, owner_hash, conversation_id, timestamp):
        self._execute(conn, "UPDATE chat_sessions SET current_conversation_id=?,updated_at=? WHERE token_hash=?",
                      (conversation_id, timestamp, owner_hash))

    def _create(self, conn, owner_hash):
        timestamp, identifier = _now(), str(uuid4())
        self._execute(conn, "INSERT INTO conversations(id,owner_hash,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                      (identifier, owner_hash, "새 대화", timestamp, timestamp))
        self._make_current(conn, owner_hash, identifier, timestamp)
        return {"id": identifier, "owner_hash": owner_hash, "title": "새 대화",
                "created_at": timestamp, "updated_at": timestamp}

    def _view(self, conn, conversation):
        rows = self._execute(conn,
            "SELECT id,turn_id,ordinal,role,content,payload,created_at FROM chat_messages WHERE conversation_id=? ORDER BY ordinal",
            (conversation["id"],)).fetchall()
        messages = []
        for row in rows:
            message = dict(row)
            message["payload"] = json.loads(message["payload"])
            messages.append(message)
        return {key: conversation[key] for key in ("id", "title", "created_at", "updated_at")} | {"messages": messages}

    def ensure_owner(self, token):
        """Reuse a known valid token; rotate unknown or malformed identifiers."""
        token_hash = sha256(token.encode()).hexdigest() if isinstance(token, str) and re.fullmatch(r"[A-Za-z0-9_-]{43}", token) else None
        with self.store.connection() as conn:
            if token_hash and self._execute(conn, "SELECT token_hash FROM chat_sessions WHERE token_hash=?", (token_hash,)).fetchone():
                self._execute(conn, "UPDATE chat_sessions SET updated_at=? WHERE token_hash=?", (_now(), token_hash))
                return token, token_hash
            token = secrets.token_urlsafe(32)
            token_hash, timestamp = sha256(token.encode()).hexdigest(), _now()
            self._execute(conn, "INSERT INTO chat_sessions(token_hash,current_conversation_id,created_at,updated_at) VALUES(?,?,?,?)",
                          (token_hash, None, timestamp, timestamp))
        return token, token_hash

    def current(self, owner_hash):
        """Return the current conversation, creating the browser's first one."""
        with self.store.connection() as conn:
            self._write_start(conn)
            session = self._session(conn, owner_hash, lock=True)
            conversation = None
            if session["current_conversation_id"]:
                row = self._execute(conn, "SELECT * FROM conversations WHERE id=? AND owner_hash=?",
                                    (session["current_conversation_id"], owner_hash)).fetchone()
                conversation = dict(row) if row is not None else None
            if conversation is None:
                conversation = self._create(conn, owner_hash)
            return self._view(conn, conversation)

    def create(self, owner_hash):
        with self.store.connection() as conn:
            self._write_start(conn)
            self._session(conn, owner_hash, lock=True)
            return self._view(conn, self._create(conn, owner_hash))

    def list(self, owner_hash):
        with self.store.connection() as conn:
            self._session(conn, owner_hash)
            rows = self._execute(conn, """SELECT c.id,c.title,c.created_at,c.updated_at,
              (SELECT COUNT(*) FROM chat_messages m WHERE m.conversation_id=c.id) AS message_count,
              (SELECT content FROM chat_messages m WHERE m.conversation_id=c.id ORDER BY ordinal DESC LIMIT 1) AS last_message
              FROM conversations c WHERE owner_hash=? ORDER BY updated_at DESC,created_at DESC,id DESC LIMIT 50""",
              (owner_hash,)).fetchall()
        return [{**dict(row), "last_message": (row["last_message"] or "")[:160]} for row in rows]

    def get(self, owner_hash, conversation_id, *, mark_current=True):
        with self.store.connection() as conn:
            if mark_current:
                self._write_start(conn)
            self._session(conn, owner_hash, lock=mark_current)
            conversation = self._owned(conn, owner_hash, conversation_id)
            if mark_current:
                self._make_current(conn, owner_hash, conversation_id, _now())
            return self._view(conn, conversation)

    def recent_history(self, owner_hash, conversation_id, limit=12, *, reference_date=None):
        limit = min(max(int(limit), 0), 12)
        if isinstance(reference_date, datetime):
            reference_date = reference_date.date()
        elif isinstance(reference_date, str):
            reference_date = date.fromisoformat(reference_date)
        elif reference_date is not None and not isinstance(reference_date, date):
            raise ValueError("대화 기준 날짜가 올바르지 않습니다")
        with self.store.connection() as conn:
            self._owned(conn, owner_hash, conversation_id)
            rows = self._execute(conn,
                "SELECT role,content,created_at,payload,turn_id FROM chat_messages WHERE conversation_id=? ORDER BY ordinal DESC LIMIT ?",
                (conversation_id, limit)).fetchall()
        ranges={}
        for row in rows:
            if row["role"]=="assistant":
                payload=json.loads(row["payload"])
                span=payload.get("date_range")
                if isinstance(span,dict) and span.get("start") and span.get("end"):
                    ranges[row["turn_id"]]={"start":span["start"],"end":span["end"]}
        history=[]
        for row in reversed(rows):
            item={"role":row["role"],"content":(
                _historical_content(row["content"],row["created_at"],reference_date) if row["role"]=="user" else row["content"]
            )[:5000]}
            if row["role"]=="user" and row["turn_id"] in ranges:
                item["date_range"]=ranges[row["turn_id"]]
            if row["role"]=="assistant":
                payload=json.loads(row["payload"])
                for key in ("selected_facility_ids","reference_facility_ids"):
                    values=payload.get(key)
                    if isinstance(values,list):
                        item[key]=[str(value) for value in values[:150] if isinstance(value,(str,int))]
                directory=payload.get("directory")
                if isinstance(directory,dict) and isinstance(directory.get("records"),list):
                    item["directory"]={"records":[{key:record[key] for key in ("id","name","location") if key in record}
                        for record in directory["records"][:150] if isinstance(record,dict)]}
                context=payload.get("facility_context")
                if isinstance(context,dict):
                    item["facility_context"]={key:context[key] for key in ("selected_facility_ids","reference_facility_ids","candidate_ids") if key in context}
            history.append(item)
        return history

    def _existing_turn(self, conn, conversation_id, turn_id, message=None):
        rows = self._execute(conn,
            "SELECT role,content,payload FROM chat_messages WHERE conversation_id=? AND turn_id=? ORDER BY ordinal",
            (conversation_id, turn_id)).fetchall()
        if not rows:
            return None
        by_role = {row["role"]: row for row in rows}
        if set(by_role) != {"user", "assistant"}:
            raise ValueError("저장된 대화 요청이 완전하지 않습니다")
        if message is not None and by_role["user"]["content"] != message:
            raise ValueError("같은 요청 ID를 다른 질문에 사용할 수 없습니다")
        return json.loads(by_role["assistant"]["payload"])

    def get_turn(self, owner_hash, conversation_id, turn_id, message=None):
        """Return a saved answer; optional message validates a retry's text."""
        with self.store.connection() as conn:
            self._owned(conn, owner_hash, conversation_id)
            return self._existing_turn(conn, conversation_id, turn_id, message)

    def append_turn(self, owner_hash, conversation_id, message, response, turn_id):
        """Atomically store exactly one ordered user/assistant pair per turn ID."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("질문을 입력해 주세요")
        if not isinstance(turn_id, str) or not turn_id.strip() or len(turn_id) > 128:
            raise ValueError("요청 ID가 올바르지 않습니다")
        if not isinstance(response, dict) or not isinstance(response.get("answer"), str):
            raise ValueError("답변 데이터가 올바르지 않습니다")
        with self.store.connection() as conn:
            self._write_start(conn)
            conversation = self._owned(conn, owner_hash, conversation_id, lock=True)
            if self._existing_turn(conn, conversation_id, turn_id, message) is not None:
                return self._view(conn, conversation)
            # Serialize before inserting either message. Unserializable model
            # metadata cannot leave a half-persisted turn.
            payload = json.dumps(response, ensure_ascii=False, allow_nan=False)
            last = self._execute(conn, "SELECT MAX(ordinal) AS position FROM chat_messages WHERE conversation_id=?",
                                 (conversation_id,)).fetchone()["position"]
            first_ordinal = 0 if last is None else last + 1
            timestamp = _now()
            for offset, role, content, stored in ((0, "user", message, "{}"), (1, "assistant", response["answer"], payload)):
                self._execute(conn, """INSERT INTO chat_messages
                  (id,conversation_id,turn_id,ordinal,role,content,payload,created_at) VALUES(?,?,?,?,?,?,?,?)""",
                  (str(uuid4()), conversation_id, turn_id, first_ordinal + offset, role, content, stored, timestamp))
            title = conversation["title"]
            if last is None:
                title = re.sub(r"\s+", " ", message).strip()[:60]
            self._execute(conn, "UPDATE conversations SET title=?,updated_at=? WHERE id=? AND owner_hash=?",
                          (title, timestamp, conversation_id, owner_hash))
            conversation.update(title=title, updated_at=timestamp)
            return self._view(conn, conversation)

    @contextmanager
    def turn_lock(self, owner_hash, conversation_id):
        """Serialize generation and persistence for this chat in one process.

        append_turn's database row/write lock independently protects ordinals
        and retry idempotency across API workers. To serialize model generation
        across workers too, deploy one worker or add a distributed request lock.
        """
        key = (owner_hash, conversation_id)
        with self._lock_guard:
            entry = self._locks.setdefault(key, [threading.RLock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                with self.store.connection() as conn:
                    self._owned(conn, owner_hash, conversation_id)
                yield
        finally:
            with self._lock_guard:
                entry[1] -= 1
                if not entry[1]:
                    self._locks.pop(key, None)
