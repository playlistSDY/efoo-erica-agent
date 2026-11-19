"""Package source/public crawl data without secrets or private conversations."""
from argparse import ArgumentParser
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import zipfile

PUBLIC_TABLES = {"collections", "chunks", "knowledge_sources"}


def public_database_copy(source, target):
    with sqlite3.connect(Path(source).resolve().as_uri()+"?mode=ro",uri=True) as original:
        with sqlite3.connect(target) as public:
            original.backup(public)
            tables=public.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
            for (table,) in tables:
                if table not in PUBLIC_TABLES:
                    quoted=table.replace('"','""')
                    public.execute(f'DROP TABLE "{quoted}"')
            public.commit()
            # Dropped rows must also disappear from SQLite's free pages.
            public.execute("VACUUM")


def package(project, output):
    project,output=Path(project).resolve(),Path(output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix="efoo-public-package-") as temporary:
        sanitized=Path(temporary)/"efoo.sqlite3"
        database=project/"data/efoo.sqlite3"
        if database.exists():
            public_database_copy(database,sanitized)
        with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as archive:
            for source in sorted(project.rglob("*")):
                if not source.is_file() or source.is_symlink() or source.resolve()==output:
                    continue
                if any(part in {".git",".venv","__pycache__",".pytest_cache"} for part in source.relative_to(project).parts):
                    continue
                if source.name.startswith(".env") and source.name!=".env.example":
                    continue
                if source.name==".DS_Store" or source.suffix in {".pyc",".zip"} or source.name.endswith(("-wal","-shm","-journal")):
                    continue
                if source.suffix in {".db",".sqlite",".sqlite3"}:
                    if source!=database:
                        continue
                    archive.write(sanitized,Path(project.name)/source.relative_to(project))
                else:
                    archive.write(source,Path(project.name)/source.relative_to(project))
            if archive.testzip() is not None:
                raise ValueError("소스 압축 파일 검증 실패")
    return output


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--project",default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(package(args.project,args.output))
