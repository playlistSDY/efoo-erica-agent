from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from app.assets import archive_assets, reset_asset_budget


SOURCE = "https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id=1&date=2026-11-13"
PNG = b"\x89PNG\r\n\x1a\n" + b"test-image-content"
JPEG = b"\xff\xd8\xff" + b"test-jpeg-content"


def response(body=PNG, mime="image/png", status=200, url=None, headers=None):
    result = Mock(spec=requests.Response)
    result.status_code = status
    result.url = url
    result.headers = {"Content-Type": mime, **(headers or {})}
    result.iter_content.return_value = [body]
    result.raise_for_status.return_value = None
    return result


class AssetArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.session = Mock()

    def archive(self, html):
        return archive_assets(html, SOURCE, self.folder, self.session)

    def test_empty_document_does_not_fetch_or_create_files(self):
        self.assertEqual(self.archive("<html><body>no resources</body></html>"), [])
        self.session.get.assert_not_called()
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_deduplicates_occurrences_and_reuses_content_hash_and_url_cache(self):
        self.session.get.return_value = response()
        html = '<img src="/uploads/gallery.png" alt="식당 사진"><a href="/uploads/gallery.png">원본</a><img src="/uploads/gallery.png">'
        manifest = self.archive(html)
        self.assertEqual(len(manifest), 1)
        item = manifest[0]
        self.assertTrue(item["downloaded"])
        self.assertEqual(item["kind"], "image")
        self.assertEqual(item["alt"], "식당 사진")
        self.assertEqual(item["sha256"], sha256(PNG).hexdigest())
        self.assertEqual((self.folder / item["local_path"].lstrip("/")).read_bytes(), PNG)
        self.assertEqual(item["content_type"], "image/png")
        self.assertEqual(item["size"], len(PNG))
        self.assertEqual(len(item["occurrences"]), 3)
        cached = self.archive(html)[0]
        self.assertEqual(cached["status"], "cached")
        self.assertEqual(self.session.get.call_count, 1)

    def test_audits_scripts_styles_navigation_external_and_unsafe_links_without_following(self):
        html = '''<link rel="stylesheet" href="/theme/assets/site.css">
        <script src="/theme/assets/app.js"></script>
        <a href="/theme/pages/auth/login.php">로그인</a>
        <a href="https://elsewhere.example/menu.pdf">외부 문서</a>
        <img src="https://elsewhere.example/image.png">
        <img src="/theme/assets/img/logo.svg" alt="로고">
        <img src="data:image/png;base64,FAKE">
        <a href="javascript:alert(1)">잘못된 링크</a>'''
        manifest = self.archive(html)
        self.assertEqual(len(manifest), 8)
        self.assertEqual({x["kind"] for x in manifest}, {"stylesheet", "script", "link", "document", "image"})
        self.assertFalse(any(x["downloaded"] for x in manifest))
        self.session.get.assert_not_called()
        self.assertEqual(next(x for x in manifest if x["url"].endswith(".svg"))["status"], "unsupported_image_format")
        self.assertEqual(sum(x["status"] == "rejected_scheme" for x in manifest), 2)

    def test_srcset_gallery_data_and_css_background_are_discovered(self):
        self.session.get.return_value = response()
        manifest = self.archive('''<picture><source srcset="/uploads/a.png 1x, /uploads/b.png 2x"><img data-src="/uploads/c.png"></picture>
            <div data-gallery='["/uploads/d.png", {"src":"/uploads/e.png"}]' style="background-image:url('/uploads/f.png')"></div>
            <style>.cover { background: url(/uploads/g.png); }</style>''')
        self.assertEqual(len(manifest), 7)
        self.assertTrue(all(item["downloaded"] for item in manifest))
        self.assertEqual(len(list((self.folder / "media").glob("*.png"))), 1)

    def test_descriptorless_srcset_and_social_metadata_images(self):
        self.session.get.return_value = response()
        manifest = self.archive('''<img srcset="/uploads/a.png, /uploads/b.png">
            <meta property="og:image" content="/uploads/c.png">
            <meta name="twitter:image" content="/uploads/c.png">''')
        self.assertEqual(len(manifest), 3)
        self.assertTrue(all(item["downloaded"] for item in manifest))

    def test_external_redirect_is_rejected_before_second_request(self):
        self.session.get.return_value = response(status=302, headers={"Location": "https://elsewhere.example/private.png"})
        item = self.archive('<img src="/uploads/a.png">')[0]
        self.assertFalse(item["downloaded"])
        self.assertIn("redirect_target_not_allowed", item["error"])
        self.assertEqual(self.session.get.call_count, 1)
        self.assertFalse(self.session.get.call_args.kwargs["allow_redirects"])
        self.assertEqual(self.session.get.call_args.kwargs["timeout"], (3.0, 12.0))

    def test_allowed_redirect_revalidates_and_archives_target(self):
        self.session.get.side_effect = [response(status=302, headers={"Location": "/uploads/b.jpg"}), response(JPEG, "image/jpeg")]
        item = self.archive('<img src="/uploads/a.png">')[0]
        self.assertTrue(item["downloaded"])
        self.assertEqual(item["final_url"], "https://life.hanyang.ac.kr/uploads/b.jpg")
        self.assertTrue(item["local_path"].endswith(".jpg"))

    def test_mime_and_magic_reject_svg_html_and_disguised_html(self):
        for mime, body in (("image/svg+xml", b"<svg>"), ("text/html", b"<html>"), ("image/png", b"<html>fake image")):
            with self.subTest(mime=mime):
                self.session.get.return_value = response(body, mime)
                item = self.archive('<img src="/uploads/a.png">')[0]
                self.assertFalse(item["downloaded"])
                self.assertEqual(item["status"], "download_failed")
        self.assertFalse((self.folder / "media").exists())

    def test_document_download_and_file_size_limit(self):
        self.session.get.return_value = response(b"%PDF-1.7\nexample", "application/pdf")
        item = self.archive('<a href="/uploads/notice.pdf">공지</a>')[0]
        self.assertEqual(item["kind"], "document")
        self.assertTrue(item["downloaded"])
        self.assertTrue(item["local_path"].endswith(".pdf"))
        with patch("app.assets.MAX_FILE_BYTES", 12):
            self.session.get.return_value = response(PNG, headers={"Content-Length": "999"})
            item = self.archive('<img src="/uploads/oversize.png">')[0]
            self.assertFalse(item["downloaded"])
            self.assertEqual(item["error"], "asset_size_limit")
            self.session.get.return_value = response(PNG)
            item = self.archive('<img src="/uploads/stream-oversize.png">')[0]
            self.assertEqual(item["error"], "asset_size_limit")

    def test_total_bytes_and_file_count_are_bounded(self):
        self.session.get.return_value = response()
        html = '<img src="/uploads/a.png"><img src="/uploads/b.png"><img src="/uploads/c.png">'
        with patch("app.assets.MAX_FILES", 1):
            manifest = self.archive(html)
            self.assertEqual([x["status"] for x in manifest], ["downloaded", "run_limit", "run_limit"])
        self.session.get.reset_mock()
        reset_asset_budget(self.session)
        with patch("app.assets.MAX_TOTAL_BYTES", len(PNG)):
            manifest = self.archive('<img src="/uploads/new-a.png"><img src="/uploads/new-b.png">')
            self.assertEqual([x["status"] for x in manifest], ["downloaded", "run_limit"])
            self.assertEqual(self.session.get.call_count, 1)

    def test_shared_session_budget_spans_pages_but_cached_files_do_not_count(self):
        self.session.get.return_value = response()
        with patch("app.assets.MAX_FILES", 1):
            self.assertEqual(self.archive('<img src="/uploads/shared.png">')[0]["status"], "downloaded")
            self.assertEqual(self.archive('<img src="/uploads/shared.png">')[0]["status"], "cached")
            self.assertEqual(self.archive('<img src="/uploads/second-page.png">')[0]["status"], "run_limit")
            reset_asset_budget(self.session)
            self.assertEqual(self.archive('<img src="/uploads/second-page.png">')[0]["status"], "downloaded")

    def test_bad_final_response_origin_and_timeout_are_nonfatal(self):
        self.session.get.return_value = response(url="https://elsewhere.example/a.png")
        item = self.archive('<img src="/uploads/a.png">')[0]
        self.assertEqual(item["error"], "response_origin_not_allowed")
        self.session.get.side_effect = requests.Timeout("test timeout")
        item = self.archive('<img src="/uploads/b.png">')[0]
        self.assertFalse(item["downloaded"])
        self.assertIn("test timeout", item["error"])

    def test_path_traversal_and_unrelated_paths_are_manifest_only(self):
        items = self.archive('<img src="/uploads/%2e%2e/auth/x.png"><img src="/theme/pages/auth/x.png">')
        self.assertTrue(all(x["status"] == "path_not_allowed" for x in items))
        self.session.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
