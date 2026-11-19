"""Audit public restaurant-page resources and archive bounded safe assets.

Navigation, scripts, stylesheets, external media and unsafe formats remain in
the manifest. Only approved official image/document files are downloaded.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import unquote, urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup
import requests


MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
MAX_FILES = 150
MAX_REDIRECTS = 3
OFFICIAL_HOST = "life.hanyang.ac.kr"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".svg", ".ico"}
_DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}
_MIME_EXTENSIONS = {
    "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
    "image/gif": ".gif", "image/webp": ".webp", "image/avif": ".avif",
    "application/pdf": ".pdf", "application/msword": ".doc",
    "application/vnd.ms-excel": ".xls", "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}
_CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.I)
_CACHE_FILE = ".asset-cache.json"


def _extension(url):
    try:
        return Path(unquote(urlparse(url).path)).suffix.lower()
    except ValueError:
        return ""


def _classify(url, fallback="link"):
    extension = _extension(url)
    if extension in _IMAGE_EXTENSIONS:
        return "image"
    if extension in _DOCUMENT_EXTENSIONS:
        return "document"
    return fallback


def _selector(tag):
    result = tag.name
    if tag.get("id"):
        result += "#" + str(tag["id"])
    for item in tag.get("class", [])[:4]:
        result += "." + str(item)
    return result


def _role(tag, kind):
    labels = " ".join(" ".join(item.get("class", [])) for item in [tag, *list(tag.parents)[:3]] if getattr(item, "attrs", None))
    if kind == "image":
        if "menu-card" in labels or "menu-item" in labels:
            return "menu"
        if "fac-cover" in labels:
            return "cover"
        if any(word in labels for word in ("gallery", "swiper", "slider", "carousel")):
            return "gallery"
        return "page_image"
    return {"stylesheet": "style", "script": "script"}.get(kind, kind)


def _source_values(value):
    """Extract obvious URL values in gallery JSON without treating labels as URLs."""
    if isinstance(value, list):
        for item in value:
            yield from _source_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _source_values(item)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("http://", "https://", "//", "/", "./", "../")) or _extension(stripped) in _IMAGE_EXTENSIONS | _DOCUMENT_EXTENSIONS:
            yield stripped


def _discover(html, source_url):
    soup = BeautifulSoup(html, "html.parser")
    found = {}

    def add(raw, kind, tag, attribute):
        if not isinstance(raw, str) or not raw.strip():
            return
        raw = raw.strip()
        try:
            url = urldefrag(urljoin(source_url, raw))[0]
        except ValueError:
            # Keep malformed references for the audit; never request them.
            url = raw
        kind = _classify(url, kind)
        occurrence = {"selector": _selector(tag), "attribute": attribute}
        if tag.get("alt"):
            occurrence["alt"] = tag["alt"]
        if tag.name == "a" and tag.get_text(" ", strip=True):
            occurrence["text"] = tag.get_text(" ", strip=True)
        if url in found:
            item = found[url]
            if occurrence not in item["occurrences"]:
                item["occurrences"].append(occurrence)
            # A later explicit image occurrence can enrich a navigation link.
            if item["kind"] == "link" and kind != "link":
                item["kind"], item["role"] = kind, _role(tag, kind)
            if tag.get("alt") and not item.get("alt"):
                item["alt"] = tag["alt"]
            return
        found[url] = {"url": url, "kind": kind, "role": _role(tag, kind),
            "selector": _selector(tag), "alt": tag.get("alt", ""),
            "occurrences": [occurrence], "downloaded": False,
            "local_path": None, "sha256": None, "content_type": None,
            "size": 0, "status": "manifest_only"}

    def srcset(value, tag, attribute):
        # Source URLs containing whitespace must be percent-encoded in srcset.
        for candidate in re.findall(r"(?:^|,)\s*([^\s,]+)", value or ""):
            add(candidate, "image", tag, attribute)

    for tag in soup.find_all(True):
        if tag.name in {"img", "source"}:
            for attribute in ("src", "data-src", "data-original", "data-lazy-src"):
                if tag.get(attribute):
                    add(tag[attribute], "image" if tag.name == "img" or tag.find_parent("picture") else "media", tag, attribute)
            for attribute in ("srcset", "data-srcset"):
                if tag.get(attribute):
                    srcset(tag[attribute], tag, attribute)
        if tag.name == "a" and tag.get("href"):
            add(tag["href"], "link", tag, "href")
        if tag.name == "link" and tag.get("href"):
            relations = tag.get("rel", [])
            kind = "stylesheet" if "stylesheet" in relations else "image" if any("icon" in rel for rel in relations) or tag.get("as") == "image" else "link"
            add(tag["href"], kind, tag, "href")
            if tag.get("imagesrcset"):
                srcset(tag["imagesrcset"], tag, "imagesrcset")
        if tag.name == "script" and tag.get("src"):
            add(tag["src"], "script", tag, "src")
        if tag.name == "meta" and tag.get("property", tag.get("name", "")).lower() in {"og:image", "og:image:url", "og:image:secure_url", "twitter:image", "twitter:image:src"} and tag.get("content"):
            add(tag["content"], "image", tag, "content")
        if tag.name == "input" and tag.get("type", "").lower() == "image" and tag.get("src"):
            add(tag["src"], "image", tag, "src")
        if tag.name in {"iframe", "embed", "video", "audio"} and tag.get("src"):
            add(tag["src"], "embed" if tag.name in {"iframe", "embed"} else "media", tag, "src")
        if tag.name == "video" and tag.get("poster"):
            add(tag["poster"], "image", tag, "poster")
        if tag.name == "object" and tag.get("data"):
            add(tag["data"], "document", tag, "data")
        if tag.get("background"):
            add(tag["background"], "image", tag, "background")
        for attribute, value in list(tag.attrs.items()):
            if not attribute.startswith("data-") or attribute in {"data-src", "data-original", "data-lazy-src", "data-srcset"}:
                continue
            if not any(word in attribute for word in ("image", "img", "photo", "gallery", "src", "full", "thumb", "original", "background", "href", "url")):
                continue
            try:
                parsed = json.loads(value) if isinstance(value, str) else value
            except (ValueError, TypeError):
                parsed = value
            for candidate in _source_values(parsed):
                add(candidate, "image", tag, attribute)
        css = (tag.get("style", "") or "") + (tag.get_text() if tag.name == "style" else "")
        for match in _CSS_URL.finditer(css):
            add(match.group(2), "image", tag, "css:url")
    return list(found.values())


def _allowed_origin(url):
    try:
        parsed = urlparse(url)
        return (parsed.scheme == "https" and parsed.hostname == OFFICIAL_HOST
            and parsed.port in {None, 443} and not parsed.username and not parsed.password)
    except ValueError:
        return False


def _path_allowed(url, kind):
    path = urlparse(url).path
    for _ in range(3):
        path = unquote(path)
    if "\\" in path or ".." in path.split("/"):
        return False
    return path.startswith("/uploads/") or (kind == "image" and path.startswith(("/theme/assets/", "/assets/")))


def _signature_matches(body, extension):
    if extension == ".jpg":
        return body.startswith(b"\xff\xd8\xff")
    if extension == ".png":
        return body.startswith(b"\x89PNG\r\n\x1a\n")
    if extension == ".gif":
        return body.startswith((b"GIF87a", b"GIF89a"))
    if extension == ".webp":
        return body.startswith(b"RIFF") and body[8:12] == b"WEBP"
    if extension == ".avif":
        return body[4:8] == b"ftyp" and any(brand in body[8:40] for brand in (b"avif", b"avis"))
    if extension == ".pdf":
        return body.startswith(b"%PDF-")
    if extension in {".doc", ".xls", ".ppt"}:
        return body.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    return extension in {".docx", ".xlsx", ".pptx"} and body.startswith(b"PK\x03\x04")


def _download(session, url, kind, budget):
    current = url
    for redirect in range(MAX_REDIRECTS + 1):
        if not _allowed_origin(current) or not _path_allowed(current, kind):
            raise ValueError("redirect_target_not_allowed")
        response = session.get(current, stream=True, allow_redirects=False,
            timeout=(3.0, 12.0), headers={"User-Agent": "EfooWebAcademicCrawler/1.0 (public restaurant assets)"})
        try:
            final = response.url or current
            if not _allowed_origin(final) or not _path_allowed(final, kind):
                raise ValueError("response_origin_not_allowed")
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location or redirect == MAX_REDIRECTS:
                    raise ValueError("redirect_limit_or_missing_location")
                current = urljoin(current, location)
                continue
            response.raise_for_status()
            mime = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            extension = _MIME_EXTENSIONS.get(mime)
            if not extension:
                raise ValueError("content_type_not_allowed")
            if (kind == "image") != mime.startswith("image/"):
                raise ValueError("content_type_mismatch")
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared = int(content_length)
                except ValueError:
                    declared = 0
                if declared > min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - budget["used"]):
                    raise ValueError("asset_size_limit")
            body = bytearray()
            for piece in response.iter_content(chunk_size=64 * 1024):
                if not piece:
                    continue
                budget["used"] += len(piece)
                if len(body) + len(piece) > MAX_FILE_BYTES or budget["used"] > MAX_TOTAL_BYTES:
                    raise ValueError("asset_size_limit")
                body.extend(piece)
            if not _signature_matches(body, extension):
                raise ValueError("file_signature_mismatch")
            return bytes(body), extension, mime, current
        finally:
            response.close()
    raise ValueError("redirect_limit")


def reset_asset_budget(session):
    """Begin a new bounded crawl when reusing an HTTP session across runs."""
    session._efoo_asset_budget = {"used": 0, "attempted": 0}


def archive_assets(html: str, source_url: str, data_dir: Path, session=None) -> list[dict]:
    """Return a resource audit and local paths for safe downloaded files.

    Downloads are nonfatal and bounded to 150 files, 10 MiB per file and
    100 MiB of new content per shared HTTP session/run. Pass the same session
    for multiple facility pages. SVG and executable content are never served.
    """
    manifest = _discover(html, source_url)
    data_dir = Path(data_dir)
    media = data_dir / "media"
    cache_path = data_dir / _CACHE_FILE
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(cache, dict):
            cache = {}
    except (OSError, ValueError):
        cache = {}
    owned = session is None
    session = session or requests.Session()
    budget = getattr(session, "_efoo_asset_budget", None)
    if not isinstance(budget, dict):
        reset_asset_budget(session)
        budget = session._efoo_asset_budget
    try:
        for item in manifest:
            url, kind = item["url"], item["kind"]
            try:
                scheme = urlparse(url).scheme
            except ValueError:
                item["status"] = "invalid_url"
                continue
            if scheme not in {"https", "http"}:
                item["status"] = "rejected_scheme"
                continue
            if not _allowed_origin(source_url) or not _allowed_origin(url):
                item["status"] = "external_or_disallowed_origin"
                continue
            if kind not in {"image", "document"}:
                continue
            if _extension(url) in {".svg", ".ico"}:
                item["status"] = "unsupported_image_format"
                continue
            if not _path_allowed(url, kind):
                item["status"] = "path_not_allowed"
                continue
            previous = cache.get(url, {})
            filename = previous.get("filename", "") if isinstance(previous, dict) else ""
            if re.fullmatch(r"[0-9a-f]{64}\.(jpg|png|gif|webp|avif|pdf|doc|docx|xls|xlsx|ppt|pptx)", filename):
                existing = media / filename
                if existing.is_file() and not existing.is_symlink() and existing.stat().st_size <= MAX_FILE_BYTES:
                    body = existing.read_bytes()
                    if sha256(body).hexdigest() == filename.split(".")[0] and _signature_matches(body, existing.suffix):
                        item.update(downloaded=True, local_path="/media/" + filename,
                            sha256=filename.split(".")[0], content_type=previous.get("content_type"),
                            size=len(body), status="cached")
                        continue
            if budget["attempted"] >= MAX_FILES or budget["used"] >= MAX_TOTAL_BYTES:
                item["status"] = "run_limit"
                continue
            budget["attempted"] += 1
            try:
                body, extension, mime, final_url = _download(session, url, kind, budget)
                digest = sha256(body).hexdigest()
                filename = digest + extension
                media.mkdir(parents=True, exist_ok=True)
                destination = media / filename
                if destination.is_symlink():
                    raise OSError("asset_destination_is_symlink")
                if not destination.exists() or sha256(destination.read_bytes()).hexdigest() != digest:
                    temporary = media / (filename + ".tmp")
                    temporary.write_bytes(body)
                    temporary.replace(destination)
                item.update(downloaded=True, local_path="/media/" + filename,
                    sha256=digest, content_type=mime, size=len(body), status="downloaded")
                if final_url != url:
                    item["final_url"] = final_url
                cache[url] = {"filename": filename, "content_type": mime, "size": len(body)}
            except (requests.RequestException, ValueError, OSError) as exc:
                item.update(status="download_failed", error=str(exc)[:300])
    finally:
        if owned:
            session.close()
    if cache:
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(cache_path)
        except OSError:
            # The archive manifest remains useful even on a read-only disk.
            pass
    return manifest
