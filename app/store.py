"""Small-corpus exact cosine search. PostgreSQL uses pgvector in production."""
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import json
import math
import sqlite3
from zoneinfo import ZoneInfo

from app.settings import Settings


def cosine(a, b):
    if not a or len(a) != len(b):
        return 0.0
    norm = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / norm if norm else 0.0


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.postgres = bool(settings.database_url)
        with self.connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS collections (
              code TEXT NOT NULL, day TEXT NOT NULL, name TEXT NOT NULL,
              url TEXT NOT NULL, fetched_at TEXT NOT NULL, meals TEXT NOT NULL,
              PRIMARY KEY (code, day))""")
            if self.postgres:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            vector_type = "vector(1536)" if self.postgres else "TEXT"
            conn.execute(f"""CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY, code TEXT NOT NULL, day TEXT NOT NULL,
              meal_type TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL,
              url TEXT NOT NULL, fetched_at TEXT NOT NULL,
              embedding {vector_type}, embedding_model TEXT)""")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_day_code ON chunks(day, code)")
            # Additive migration keeps existing meal embeddings and snapshots.
            additions = {"doc_type": "TEXT NOT NULL DEFAULT 'menu'", "images": "TEXT NOT NULL DEFAULT '[]'",
                         "metadata": "TEXT NOT NULL DEFAULT '{}'", "scope": "TEXT NOT NULL DEFAULT ''"}
            if self.postgres:
                for name, definition in additions.items():
                    conn.execute(f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS {name} {definition}")
                conn.execute("ALTER TABLE collections ADD COLUMN IF NOT EXISTS payload TEXT NOT NULL DEFAULT '{}'")
            else:
                names = {r["name"] for r in conn.execute("PRAGMA table_info(chunks)")}
                for name, definition in additions.items():
                    if name not in names:
                        conn.execute(f"ALTER TABLE chunks ADD COLUMN {name} {definition}")
                if "payload" not in {r["name"] for r in conn.execute("PRAGMA table_info(collections)")}:
                    conn.execute("ALTER TABLE collections ADD COLUMN payload TEXT NOT NULL DEFAULT '{}'")
            conn.execute("""CREATE TABLE IF NOT EXISTS knowledge_sources (
              scope TEXT PRIMARY KEY, code TEXT NOT NULL, name TEXT NOT NULL,
              url TEXT NOT NULL, fetched_at TEXT NOT NULL, payload TEXT NOT NULL)""")

    @contextmanager
    def connection(self):
        if self.postgres:
            import psycopg
            from psycopg.rows import dict_row
            conn = psycopg.connect(self.settings.database_url, row_factory=dict_row)
        else:
            conn = sqlite3.connect(self.settings.data_dir / "efoo.sqlite3", timeout=20)
            conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def sql(self, query):
        return query.replace("?", "%s") if self.postgres else query

    def replace_collection(self, result, chunks):
        """Atomic snapshot replacement; a failed crawl never reaches this method."""
        with self.connection() as conn:
            old = {r["id"]: dict(r) for r in conn.execute(self.sql(
                "SELECT * FROM chunks WHERE code=? AND day=? AND doc_type='menu'"), (result.code, result.date)).fetchall()}
            conn.execute(self.sql("DELETE FROM chunks WHERE code=? AND day=? AND doc_type='menu'"), (result.code, result.date))
            conn.execute(self.sql("""INSERT INTO collections(code,day,name,url,fetched_at,meals)
              VALUES(?,?,?,?,?,?) ON CONFLICT(code,day) DO UPDATE SET
              name=excluded.name,url=excluded.url,fetched_at=excluded.fetched_at,meals=excluded.meals"""),
              (result.code, result.date, result.name, result.source_url, result.fetched_at,
               json.dumps(result.meals, ensure_ascii=False)))
            for chunk in chunks:
                previous = old.get(chunk["id"], {})
                embedding = previous.get("embedding")
                conn.execute(self.sql("""INSERT INTO chunks
                  (id,code,day,meal_type,title,text,url,fetched_at,embedding,embedding_model,images,metadata)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"""),
                  (chunk["id"], result.code, result.date, chunk["meal_type"], chunk["title"],
                   chunk["text"], result.source_url, result.fetched_at, embedding,
                   previous.get("embedding_model"),json.dumps(chunk.get("images",[]),ensure_ascii=False),
                   json.dumps(chunk.get("metadata",{}),ensure_ascii=False)))
            from dataclasses import asdict
            payload = asdict(result)
            payload.pop("html",None)
            conn.execute(self.sql("UPDATE collections SET payload=? WHERE code=? AND day=?"),
                         (json.dumps(payload,ensure_ascii=False),result.code,result.date))

    def replace_knowledge(self, scope, *, code, name, url, fetched_at, payload, chunks):
        """Replace one public page atomically; reuse unchanged text embeddings."""
        with self.connection() as conn:
            old = {r["id"]:dict(r) for r in conn.execute(self.sql("SELECT * FROM chunks WHERE scope=?"),(scope,)).fetchall()}
            conn.execute(self.sql("DELETE FROM chunks WHERE scope=?"),(scope,))
            conn.execute(self.sql("""INSERT INTO knowledge_sources(scope,code,name,url,fetched_at,payload)
              VALUES(?,?,?,?,?,?) ON CONFLICT(scope) DO UPDATE SET code=excluded.code,name=excluded.name,
              url=excluded.url,fetched_at=excluded.fetched_at,payload=excluded.payload"""),
              (scope,code,name,url,fetched_at,json.dumps(payload,ensure_ascii=False)))
            for chunk in chunks:
                previous=old.get(chunk["id"],{})
                conn.execute(self.sql("""INSERT INTO chunks
                  (id,code,day,meal_type,title,text,url,fetched_at,embedding,embedding_model,doc_type,images,metadata,scope)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""),
                  (chunk["id"],code,fetched_at[:10],"",chunk["title"],chunk["text"],url,fetched_at,
                   previous.get("embedding"),previous.get("embedding_model"),chunk["doc_type"],
                   json.dumps(chunk.get("images",[]),ensure_ascii=False),json.dumps(chunk.get("metadata",{}),ensure_ascii=False),scope))

    def knowledge(self):
        with self.connection() as conn:
            rows=[dict(r) for r in conn.execute("SELECT * FROM knowledge_sources ORDER BY code,scope").fetchall()]
        for row in rows:
            row["payload"]=json.loads(row["payload"])
        return rows

    def lookup_code(self,text):
        choices=[(row["name"],row["code"]) for row in self.knowledge() if row["scope"].startswith("facility:")]
        aliases=[]
        for name,code in choices:
            if "우편" in name:
                aliases.extend((word,code) for word in ("우체국","우편취급국","우편 취급국"))
            if "신한은행" in name:
                aliases.append(("신한은행",code))
        choices+=aliases
        matches=[(name,code) for name,code in choices if name and name.casefold() in text.casefold()]
        if matches:
            longest=max(len(name) for name,_ in matches)
            codes={code for name,code in matches if len(name)==longest}
            # Unqualified chain names may refer to several campus branches.
            return next(iter(codes)) if len(codes)==1 else None
        return None

    def pending(self):
        with self.connection() as conn:
            return [dict(r) for r in conn.execute(self.sql(
                "SELECT * FROM chunks WHERE embedding IS NULL OR embedding_model != ?"),
                (self.settings.embedding_model,)).fetchall()]

    def save_embeddings(self, rows, vectors):
        if len(rows) != len(vectors):
            raise ValueError("임베딩 응답 개수가 청크 개수와 다릅니다")
        if any(len(v) != self.settings.embedding_dimensions or
               not all(math.isfinite(x) for x in v) for v in vectors):
            raise ValueError("임베딩 차원 또는 숫자 형식이 올바르지 않습니다")
        with self.connection() as conn:
            for row, vector in zip(rows, vectors):
                conn.execute(self.sql("UPDATE chunks SET embedding=?,embedding_model=? WHERE id=?"),
                    (json.dumps(vector), self.settings.embedding_model, row["id"]))

    def stats(self):
        with self.connection() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            indexed = conn.execute(self.sql(
                "SELECT COUNT(*) AS n FROM chunks WHERE embedding IS NOT NULL AND embedding_model=?"),
                (self.settings.embedding_model,)).fetchone()["n"]
            info = dict(conn.execute("SELECT COUNT(*) AS n,MAX(fetched_at) AS last FROM collections").fetchone())
            knowledge=conn.execute("SELECT COUNT(*) AS n,MAX(fetched_at) AS last FROM knowledge_sources").fetchone()
            facility=conn.execute("SELECT COUNT(*) AS n FROM knowledge_sources WHERE scope LIKE 'facility:%'").fetchone()["n"]
            notice=conn.execute("SELECT COUNT(*) AS n FROM knowledge_sources WHERE scope LIKE 'notice:%'").fetchone()["n"]
        return {"document_count": info["n"]+knowledge["n"], "meal_snapshot_count":info["n"],
                "facility_count":facility,"notice_count":notice,"chunk_count": total,
                "indexed_count": indexed, "last_updated": max(filter(None,[info["last"],knowledge["last"]]),default=None)}

    def collections(self, day, code=None):
        query, args = "SELECT * FROM collections WHERE day=?", [day]
        if code:
            query += " AND code=?"
            args.append(code)
        with self.connection() as conn:
            rows = [dict(r) for r in conn.execute(self.sql(query + " ORDER BY code"), args).fetchall()]
        for row in rows:
            row["meals"] = json.loads(row["meals"])
            row["payload"] = json.loads(row["payload"])
        return rows

    def candidates(self, day, code=None, meal_type=None, *, indexed=False, query_vector=None, doc_types=None,
                   end_day=None, exclude_targets=None, exclude_scopes=None):
        end_day = end_day or day
        if doc_types and set(doc_types)!={"menu"}:
            query,args="SELECT * FROM chunks WHERE (day BETWEEN ? AND ? OR doc_type!='menu')",[day,end_day]
        else:
            query,args="SELECT * FROM chunks WHERE day BETWEEN ? AND ?",[day,end_day]
        if doc_types:
            query+=" AND doc_type IN ("+",".join("?" for _ in doc_types)+")"
            args.extend(doc_types)
        if code:
            query += " AND (code=? OR (doc_type='notice' AND metadata LIKE ?))"
            args.extend([code,'%"'+code+'"%'])
        if meal_type and (not doc_types or "menu" in doc_types):
            query += " AND (doc_type!='menu' OR meal_type=?)"
            args.append(meal_type)
        current_day = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.settings.max_age_hours)
        query += " AND ((doc_type='menu' AND day<?) OR fetched_at>=?)"
        args.extend([current_day, cutoff.isoformat()])
        for target_day, target_code in sorted(set(exclude_targets or [])):
            query += " AND NOT (doc_type='menu' AND day=? AND code=?)"
            args.extend([str(target_day), target_code])
        scopes=sorted(set(exclude_scopes or []))
        if scopes:
            query += " AND scope NOT IN ("+",".join("?" for _ in scopes)+")"
            args.extend(scopes)
        if indexed:
            query += " AND embedding IS NOT NULL AND embedding_model=?"
            args.append(self.settings.embedding_model)
        if self.postgres and query_vector is not None:
            query = query.replace("SELECT *", "SELECT *, 1 - (embedding <=> ?::vector) AS score")
            args.insert(0, json.dumps(query_vector))
            # The dated cafeteria corpus is deliberately small. Defer the
            # context cap until after per-cafeteria coverage selection.
            query += " ORDER BY embedding <=> ?::vector"
            args.append(json.dumps(query_vector))
        else:
            query += " ORDER BY code,meal_type,id"
        with self.connection() as conn:
            rows = [dict(r) for r in conn.execute(self.sql(query), args).fetchall()]
        return rows
