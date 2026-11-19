import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.memory import router
from app.services import memory_service
from app.services.note_paths import resolve_note_path


class NotePathSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.notes = self.base / "notes"
        self.notes.mkdir()
        self.outside = self.base / "outside.md"
        self.outside.write_text("private", encoding="utf-8")
        self.patcher = patch.object(memory_service, "REFINED_DIR", self.notes)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.indexer = patch.object(memory_service, "index_refined_note").start()
        self.addCleanup(patch.stopall)
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_nested_markdown_roundtrip(self):
        memory_service.save_note_content("ideas/example.md", "hello")
        self.assertEqual(memory_service.read_note_content("ideas/example.md"), "hello")
        self.indexer.assert_called_once_with(self.notes / "ideas/example.md")

    def test_read_rejects_parent_and_absolute_paths(self):
        for value in ["../outside.md", str(self.outside), "ideas/../../outside.md"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                memory_service.read_note_content(value)

    def test_write_rejects_escape_before_modifying_files(self):
        for value in ["../outside.md", str(self.outside), "../created/file.md"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                memory_service.save_note_content(value, "overwritten")
        self.assertEqual(self.outside.read_text(encoding="utf-8"), "private")
        self.assertFalse((self.base / "created").exists())
        self.indexer.assert_not_called()

    def test_rejects_windows_paths_on_every_platform(self):
        for value in [r"C:\outside.md", "C:/outside.md", "C:outside.md", r"\\server\share\a.md", r"..\outside.md", "note.md:stream", "//server/share/a.md"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_note_path(self.notes, value)

    def test_rejects_empty_non_markdown_and_nul(self):
        for value in ["", ".", "note.txt", "server.py", "note.md\x00", None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_note_path(self.notes, value)

    def test_sibling_prefix_does_not_count_as_containment(self):
        with self.assertRaises(ValueError):
            resolve_note_path(self.notes, "../notes-other/note.md")

    def make_symlink(self, link, target, is_directory=False):
        try:
            link.symlink_to(target, target_is_directory=is_directory)
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")

    def test_symlink_file_cannot_escape(self):
        self.make_symlink(self.notes / "link.md", self.outside)
        with self.assertRaises(ValueError):
            memory_service.read_note_content("link.md")
        with self.assertRaises(ValueError):
            memory_service.save_note_content("link.md", "changed")
        self.assertEqual(self.outside.read_text(encoding="utf-8"), "private")

    def test_symlink_directory_cannot_escape(self):
        self.make_symlink(self.notes / "linked", self.base, is_directory=True)
        with self.assertRaises(ValueError):
            memory_service.save_note_content("linked/new.md", "changed")
        self.assertFalse((self.base / "new.md").exists())

    def test_api_returns_400_for_invalid_paths(self):
        r = self.client.get("/api/memory/notes/content", params={"path": "../outside.md"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json(), {"detail": "Invalid note path"})
        r = self.client.post("/api/memory/notes/save", json={"path": "../outside.md", "content": "changed"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.outside.read_text(encoding="utf-8"), "private")

    def test_api_preserves_valid_save_read_and_missing_note(self):
        r = self.client.post("/api/memory/notes/save", json={"path": "new/example.md", "content": "valid"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get("/api/memory/notes/content", params={"path": "new/example.md"}).json(), {"content": "valid"})
        self.assertEqual(self.client.get("/api/memory/notes/content", params={"path": "missing.md"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
