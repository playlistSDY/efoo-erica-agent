import io
import importlib.util
import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("efoo_week2_probe", ROOT / "app.py")
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"DOCUMENT_BUCKET": "efoo-test", "APP_VERSION": "test-sha"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.manifest = (ROOT / "data/manifest.json").read_bytes()

    def client(self, raw=None):
        client = Mock()
        client.get_object.return_value = {"Body": io.BytesIO(self.manifest if raw is None else raw), "ETag": '"test-etag"', "VersionId": "test-version"}
        return client

    def test_success_requires_actual_get_and_no_rag_claim(self):
        client = self.client()
        code, result = app.environment_result(client)
        self.assertEqual(code, 200)
        client.get_object.assert_called_once_with(Bucket="efoo-test", Key="samples/manifest.json")
        self.assertTrue(result["storage"]["read_ok"])
        self.assertFalse(result["rag_enabled"])
        self.assertEqual(result["source_count"], 2)
        self.assertEqual(result["version"], "test-sha")

    def test_access_denied_is_503_not_fake_success(self):
        client = self.client()
        client.get_object.side_effect = ClientError({"Error": {"Code": "AccessDenied", "Message": "SECRET DETAIL"}}, "GetObject")
        code, result = app.environment_result(client)
        self.assertEqual(code, 503)
        self.assertFalse(result["storage"]["read_ok"])
        self.assertEqual(result["error_code"], "AccessDenied")
        self.assertNotIn("SECRET DETAIL", json.dumps(result))

    def test_missing_bucket_does_not_create_client(self):
        with patch.dict(os.environ, {"DOCUMENT_BUCKET": ""}), patch.object(app.boto3, "client") as factory:
            self.assertEqual(app.environment_result()[0], 503)
            factory.assert_not_called()

    def test_other_project_and_nonofficial_urls_are_rejected(self):
        for change in ({"project": "Other project"}, {"kind": "indexed-documents"}, {"documents": []}, {"documents": [{"id": "x", "title": "x", "category": "x", "source_url": "https://example.com/"}]}):
            data = json.loads(self.manifest)
            data.update(change)
            self.assertEqual(app.environment_result(self.client(json.dumps(data).encode()))[0], 503)

    def test_invalid_or_large_s3_data_is_rejected(self):
        for raw in (b"not json", b"[]", b"x" * (app.MAX_MANIFEST_BYTES + 1)):
            self.assertEqual(app.environment_result(self.client(raw))[0], 503)

    def test_http_routes_reflect_actual_status(self):
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/healthz") as response:
            self.assertEqual(json.load(response)["project"], "EFOO")
        with urllib.request.urlopen(base + "/") as response:
            self.assertIn("RAG 미연결", response.read().decode())
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        with patch.object(app.boto3, "client", return_value=self.client()):
            with urllib.request.urlopen(base + "/api/environment") as response:
                self.assertTrue(json.load(response)["storage"]["read_ok"])
        client = self.client()
        client.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        with patch.object(app.boto3, "client", return_value=client):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(base + "/api/environment")
            self.assertEqual(caught.exception.code, 503)
            caught.exception.close()
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(base + "/missing")
        self.assertEqual(caught.exception.code, 404)
        caught.exception.close()


if __name__ == "__main__":
    unittest.main()
