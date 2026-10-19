"""Offline checks for downloading and safely extracting research assets."""
import contextlib
import functools
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("download_assets", REPO / "scripts/download_assets.py")
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def package(path, members):
    with tarfile.open(path, "w:gz") as stream:
        for name, data in members:
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            stream.addfile(entry, io.BytesIO(data))


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class DownloadTests(unittest.TestCase):
    def test_cli_download_verify_extract_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            public.mkdir()
            data = b'{"text":"A short fixture.","label":0}\n'
            archive = public / "tiny.tar.gz"
            package(archive, [("dataset/tiny.jsonl", data), ("metadata/tiny.json", b"{}")])
            server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(public)))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                output = root / "data"
                cache = output / ".downloads/v0.1.0"
                cache.mkdir(parents=True)
                # A server ignoring Range should still allow a clean verified retry.
                (cache / "tiny.tar.gz.partial").write_bytes(archive.read_bytes()[:7])
                manifest = {"release_tag": "v0.1.0", "assets": [{
                    "id": "temp-hc3", "filename": archive.name,
                    "bytes": archive.stat().st_size, "sha256": downloader.sha256(archive),
                    "url": f"http://127.0.0.1:{server.server_port}/{archive.name}",
                    "files": [{"path": "dataset/tiny.jsonl", "bytes": len(data),
                               "sha256": hashlib.sha256(data).hexdigest()}]}]}
                path = root / "manifest.json"
                path.write_text(json.dumps(manifest))
                command = [sys.executable, str(REPO / "scripts/download_assets.py"),
                           "--assets", "temp-hc3", "--output", str(output), "--manifest", str(path)]
                first = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(first.returncode, 0, first.stderr)
                self.assertEqual((output / "dataset/tiny.jsonl").read_bytes(), data)
                second = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(second.returncode, 0, second.stderr)
                self.assertIn("Verified cached download", second.stdout)
                (output / "dataset/tiny.jsonl").write_bytes(b"local modification")
                third = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(third.returncode, 0)
                self.assertEqual((output / "dataset/tiny.jsonl").read_bytes(), b"local modification")
            finally:
                server.shutdown()
                server.server_close()
                worker.join()

    def test_corrupted_download_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.tar.gz"
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Existing download differs"):
                downloader.download("https://example.invalid/unused", path, "0" * 64, 7)

    def test_unsafe_archive_paths_and_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["../escape.txt", "/absolute.txt", "metadata/../../escape.txt", "metadata\\escape.txt"]:
                with self.subTest(name=name):
                    path = root / "unsafe.tar.gz"
                    package(path, [(name, b"bad")])
                    with self.assertRaises(ValueError):
                        downloader.extract(path, root / "output", [])
            path = root / "link.tar.gz"
            with tarfile.open(path, "w:gz") as stream:
                entry = tarfile.TarInfo("metadata/link")
                entry.type = tarfile.SYMTYPE
                entry.linkname = "/tmp"
                stream.addfile(entry)
            with self.assertRaises(ValueError):
                downloader.extract(path, root / "output", [])
            self.assertFalse((root / "escape.txt").exists())

    def test_unlisted_missing_and_mismatched_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "data.tar.gz"
            package(archive, [("dataset/data.txt", b"value")])
            with self.assertRaisesRegex(ValueError, "absent from"):
                downloader.extract(archive, root / "output", [])
            expected = [{"path": "dataset/data.txt", "bytes": 5, "sha256": "0" * 64}]
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                downloader.extract(archive, root / "output", expected)
            empty = root / "empty.tar.gz"
            package(empty, [])
            with self.assertRaisesRegex(ValueError, "missing files"):
                downloader.extract(empty, root / "other", expected)


if __name__ == "__main__":
    unittest.main()
