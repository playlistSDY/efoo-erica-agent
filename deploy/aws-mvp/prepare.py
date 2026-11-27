"""Build a deployable ZIP with only application code and public evidence."""
import argparse
import json
from pathlib import Path
import re
import sqlite3
import sys
from tempfile import TemporaryDirectory
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.package_source import PUBLIC_TABLES, public_database_copy

MEDIA_NAME = re.compile(r"[0-9a-f]{64}\.(?:jpg|jpeg|png|gif|webp|avif)")


def prepare(output: Path, data: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp:
        seed = Path(tmp) / "efoo.sqlite3"
        public_database_copy(data / "efoo.sqlite3", seed)
        with sqlite3.connect(seed) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables - {"sqlite_sequence"} != PUBLIC_TABLES:
                raise ValueError("Unexpected public seed schema")
            chunk_count, indexed = db.execute(
                "SELECT COUNT(*), COUNT(embedding) FROM chunks"
            ).fetchone()
            if not indexed or indexed != chunk_count:
                raise ValueError("Complete public embeddings before packaging")
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((ROOT / "app").rglob("*")):
                rel = path.relative_to(ROOT)
                if path.is_file() and not path.is_symlink() and not any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                    archive.write(path, rel)
            archive.write(ROOT / "requirements.txt", "requirements.txt")
            archive.write(seed, "seed/efoo.sqlite3")
            for path in sorted((data / "media").iterdir()):
                if path.is_file() and not path.is_symlink() and MEDIA_NAME.fullmatch(path.name):
                    archive.write(path, "seed/media/" + path.name)
            for name in ("Dockerfile", "README.md", "deploy.sh", "templates.py"):
                archive.write(ROOT / "deploy/aws-mvp" / name, name)
            for name in ("bootstrap.json", "environment.json", "frontend-probe.json"):
                archive.write(ROOT / "deploy/aws-test" / name, name)
        print(json.dumps({"bundle": str(output), "chunks": chunk_count, "indexed": indexed}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    prepare(args.output.resolve(), args.data.resolve())
