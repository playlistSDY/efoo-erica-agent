"""EFOO Week 2 infrastructure probe; no chat history or model credentials."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

LOGGER = logging.getLogger("efoo.environment")
MAX_MANIFEST_BYTES = 64 * 1024
ROOT = Path(__file__).resolve().parent


def validate_manifest(raw: bytes) -> list[dict[str, str]]:
    """Require a small EFOO source manifest, not a claim of indexed documents."""
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("project") != "EFOO":
        raise ValueError("The manifest must belong to EFOO")
    if manifest.get("kind") != "source-metadata-only":
        raise ValueError("Unsupported manifest kind")
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not 1 <= len(documents) <= 100:
        raise ValueError("The manifest must contain between 1 and 100 sources")
    result = []
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Invalid source record")
        item = {}
        for key in ("id", "title", "source_url", "category"):
            value = document.get(key)
            if not isinstance(value, str) or not value.strip() or len(value) > 1000:
                raise ValueError("Invalid source field")
            item[key] = value
        source = urlsplit(item["source_url"])
        if source.scheme != "https" or source.hostname != "life.hanyang.ac.kr":
            raise ValueError("Only official HTTPS campus source URLs are allowed")
        if source.username or source.password:
            raise ValueError("Source URLs must not contain credentials")
        result.append(item)
    return result


def environment_result(s3_client=None) -> tuple[int, dict]:
    """Actually GET the configured S3 object on each request; never use fixtures."""
    bucket = os.getenv("DOCUMENT_BUCKET", "")
    key = os.getenv("DOCUMENT_KEY", "samples/manifest.json")
    region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"))
    result = {
        "project": "EFOO",
        "stage": "week2",
        "status": "error",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "region": region,
        "version": os.getenv("APP_VERSION", "local-draft"),
        "rag_enabled": False,
        "scope": "AWS environment verification; source metadata only",
        "storage": {"type": "s3", "bucket": bucket, "key": key, "read_ok": False},
    }
    if not bucket:
        result["error"] = "DOCUMENT_BUCKET is not configured"
        return 503, result
    try:
        client = s3_client or boto3.client(
            "s3", region_name=region,
            config=Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 1}),
        )
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            if response.get("ContentLength", 0) > MAX_MANIFEST_BYTES:
                raise ValueError("Manifest is too large")
            raw = body.read(MAX_MANIFEST_BYTES + 1)
        finally:
            body.close()
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ValueError("Manifest is too large")
        documents = validate_manifest(raw)
        result.update(status="ok", source_count=len(documents), sources=documents)
        result["storage"].update(
            read_ok=True,
            etag=response.get("ETag", "").strip('"'),
            version_id=response.get("VersionId"),
        )
        if response.get("LastModified"):
            result["storage"]["last_modified"] = response["LastModified"].isoformat()
        return 200, result
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", "S3Error"))
        LOGGER.warning("s3_read_failed code=%s", code)
        result["error"] = "S3 object could not be read; check object existence and task permissions"
        result["error_code"] = code
    except (BotoCoreError, ValueError, TypeError, KeyError, OSError):
        LOGGER.warning("environment_verification_failed", exc_info=False)
        result["error"] = "S3 manifest connection or validation failed"
    return 503, result


class Handler(BaseHTTPRequestHandler):
    server_version = "EFOO-Week2"
    sys_version = ""

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/healthz":
            self.send_json(200, {"status": "ok", "project": "EFOO", "stage": "week2", "rag_enabled": False})
        elif path == "/api/environment":
            status, result = environment_result()
            self.send_json(status, result)
        elif path == "/":
            self.send_content(200, ROOT.joinpath("index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_json(404, {"error": "Not found"})

    def send_json(self, status: int, value: dict):
        self.send_content(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def send_content(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Never log arbitrary URLs, query strings, request bodies, or headers.
        LOGGER.info("http_request method=%s status=%s", self.command, args[1] if len(args) > 1 else "-")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    LOGGER.info("server_started project=EFOO stage=week2 port=%s rag_enabled=false", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
