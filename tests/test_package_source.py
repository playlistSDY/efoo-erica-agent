import importlib.util
from pathlib import Path
import sqlite3
import zipfile


spec=importlib.util.spec_from_file_location("package_source",Path(__file__).parents[1]/"scripts/package_source.py")
packaging=importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


def test_archive_keeps_public_knowledge_and_removes_private_rows_and_secrets(tmp_path):
    project=tmp_path/"project"
    (project/"data").mkdir(parents=True)
    secret=b"PRIVATE_CHAT_MARKER_f94b3b9b"
    with sqlite3.connect(project/"data/efoo.sqlite3") as conn:
        conn.execute("CREATE TABLE chunks(id TEXT, text TEXT)")
        conn.execute("INSERT INTO chunks VALUES('public','official campus facts')")
        for table in ("chat_sessions","conversations","chat_messages","future_private_table"):
            conn.execute(f"CREATE TABLE {table}(text TEXT)")
            conn.execute(f"INSERT INTO {table} VALUES(?)",(secret.decode(),))
    (project/".env").write_text("OPENAI_API_KEY=private-key")
    (project/".env.local").write_text("ANOTHER_SECRET=yes")
    (project/".env.example").write_text("OPENAI_API_KEY=")
    (project/"app.py").write_text("print('app')")
    target=packaging.package(project,tmp_path/"source.zip")
    with zipfile.ZipFile(target) as archive:
        names=archive.namelist()
        assert "project/.env.example" in names
        assert "project/.env" not in names and "project/.env.local" not in names
        database=archive.read("project/data/efoo.sqlite3")
    assert secret not in database
    safe=tmp_path/"public.sqlite3"
    safe.write_bytes(database)
    with sqlite3.connect(safe) as conn:
        assert conn.execute("SELECT * FROM chunks").fetchall()==[("public","official campus facts")]
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()==[("chunks",)]
    with sqlite3.connect(project/"data/efoo.sqlite3") as conn:
        assert conn.execute("SELECT text FROM chat_messages").fetchone()[0]==secret.decode()
