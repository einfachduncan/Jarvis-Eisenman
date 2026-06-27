"""Tests für memory.py (PHASE 5 – Langzeitgedächtnis)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import memory as mem


def _tmp_db() -> Path:
    """Gibt den Pfad zu einer frischen temporären Datenbank zurück."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    p = Path(tmp.name)
    mem.init_db(p)
    return p


class TestInitDb(unittest.TestCase):
    def test_tables_created(self):
        db = _tmp_db()
        with mem._connect(db) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        expected = {"conversations", "memories", "settings", "people", "notes"}
        self.assertTrue(expected.issubset(tables))


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------


class TestConversations(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()

    def _msgs(self, text: str = "Hallo") -> str:
        return json.dumps([{"role": "user", "content": text}])

    def test_save_and_load(self):
        cid = mem.save_conversation(self._msgs(), title="Test", db_path=self.db)
        conv = mem.load_conversation(cid, db_path=self.db)
        self.assertIsNotNone(conv)
        self.assertEqual(conv.title, "Test")

    def test_load_nonexistent_returns_none(self):
        self.assertIsNone(mem.load_conversation(9999, db_path=self.db))

    def test_list_conversations(self):
        mem.save_conversation(self._msgs("a"), db_path=self.db)
        mem.save_conversation(self._msgs("b"), db_path=self.db)
        convs = mem.list_conversations(db_path=self.db)
        self.assertEqual(len(convs), 2)

    def test_search_conversations(self):
        mem.save_conversation(self._msgs("Wetter heute"), title="Wetter", db_path=self.db)
        mem.save_conversation(self._msgs("Python Tipps"), title="Python", db_path=self.db)
        results = mem.search_conversations("Wetter", db_path=self.db)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Wetter")

    def test_update_conversation(self):
        cid = mem.save_conversation(self._msgs(), title="Alt", db_path=self.db)
        ok = mem.update_conversation(cid, title="Neu", db_path=self.db)
        self.assertTrue(ok)
        conv = mem.load_conversation(cid, db_path=self.db)
        self.assertEqual(conv.title, "Neu")

    def test_update_nonexistent_returns_false(self):
        self.assertFalse(mem.update_conversation(9999, title="X", db_path=self.db))

    def test_delete_conversation(self):
        cid = mem.save_conversation(self._msgs(), db_path=self.db)
        self.assertTrue(mem.delete_conversation(cid, db_path=self.db))
        self.assertIsNone(mem.load_conversation(cid, db_path=self.db))

    def test_delete_nonexistent_returns_false(self):
        self.assertFalse(mem.delete_conversation(9999, db_path=self.db))


# ---------------------------------------------------------------------------
# memories
# ---------------------------------------------------------------------------


class TestMemories(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()

    def test_save_and_load(self):
        mem.save_memory("name", "Duncan", db_path=self.db)
        m = mem.load_memory("name", db_path=self.db)
        self.assertIsNotNone(m)
        self.assertEqual(m.value, "Duncan")

    def test_upsert_updates_value(self):
        mem.save_memory("name", "Duncan", db_path=self.db)
        mem.save_memory("name", "Alex", db_path=self.db)
        m = mem.load_memory("name", db_path=self.db)
        self.assertEqual(m.value, "Alex")

    def test_load_nonexistent_returns_none(self):
        self.assertIsNone(mem.load_memory("nonexistent", db_path=self.db))

    def test_list_memories(self):
        mem.save_memory("a", "1", db_path=self.db)
        mem.save_memory("b", "2", db_path=self.db)
        self.assertEqual(len(mem.list_memories(db_path=self.db)), 2)

    def test_list_by_category(self):
        mem.save_memory("x", "1", category="persönlich", db_path=self.db)
        mem.save_memory("y", "2", category="arbeit", db_path=self.db)
        results = mem.list_memories(category="persönlich", db_path=self.db)
        self.assertEqual(len(results), 1)

    def test_search_memories(self):
        mem.save_memory("lieblingsfarbe", "Blau", db_path=self.db)
        mem.save_memory("hobby", "Schach", db_path=self.db)
        results = mem.search_memories("Blau", db_path=self.db)
        self.assertEqual(len(results), 1)

    def test_update_memory(self):
        mem.save_memory("lang", "Python", db_path=self.db)
        ok = mem.update_memory("lang", value="Rust", db_path=self.db)
        self.assertTrue(ok)
        self.assertEqual(mem.load_memory("lang", db_path=self.db).value, "Rust")

    def test_update_nonexistent_returns_false(self):
        self.assertFalse(mem.update_memory("ghost", value="X", db_path=self.db))

    def test_delete_memory(self):
        mem.save_memory("temp", "X", db_path=self.db)
        self.assertTrue(mem.delete_memory("temp", db_path=self.db))
        self.assertIsNone(mem.load_memory("temp", db_path=self.db))

    def test_delete_nonexistent_returns_false(self):
        self.assertFalse(mem.delete_memory("ghost", db_path=self.db))


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


class TestSettings(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()

    def test_set_and_get(self):
        mem.set_setting("theme", "dark", db_path=self.db)
        self.assertEqual(mem.get_setting("theme", db_path=self.db), "dark")

    def test_default_returned_when_missing(self):
        self.assertEqual(mem.get_setting("missing", default="fallback", db_path=self.db), "fallback")

    def test_overwrite(self):
        mem.set_setting("theme", "dark", db_path=self.db)
        mem.set_setting("theme", "light", db_path=self.db)
        self.assertEqual(mem.get_setting("theme", db_path=self.db), "light")

    def test_list_settings(self):
        mem.set_setting("a", "1", db_path=self.db)
        mem.set_setting("b", "2", db_path=self.db)
        self.assertEqual(len(mem.list_settings(db_path=self.db)), 2)

    def test_delete_setting(self):
        mem.set_setting("del_me", "x", db_path=self.db)
        self.assertTrue(mem.delete_setting("del_me", db_path=self.db))
        self.assertEqual(mem.get_setting("del_me", db_path=self.db), "")

    def test_delete_nonexistent_returns_false(self):
        self.assertFalse(mem.delete_setting("ghost", db_path=self.db))


# ---------------------------------------------------------------------------
# people
# ---------------------------------------------------------------------------


class TestPeople(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()

    def test_save_and_load(self):
        mem.save_person("Anna", "Freundin", db_path=self.db)
        p = mem.load_person("Anna", db_path=self.db)
        self.assertIsNotNone(p)
        self.assertEqual(p.notes, "Freundin")

    def test_upsert(self):
        mem.save_person("Anna", "Freundin", db_path=self.db)
        mem.save_person("Anna", "Beste Freundin", db_path=self.db)
        p = mem.load_person("Anna", db_path=self.db)
        self.assertEqual(p.notes, "Beste Freundin")

    def test_load_nonexistent_returns_none(self):
        self.assertIsNone(mem.load_person("Niemand", db_path=self.db))

    def test_list_people(self):
        mem.save_person("Anna", db_path=self.db)
        mem.save_person("Ben", db_path=self.db)
        self.assertEqual(len(mem.list_people(db_path=self.db)), 2)

    def test_search_people(self):
        mem.save_person("Anna", "Ärztin", db_path=self.db)
        mem.save_person("Ben", "Ingenieur", db_path=self.db)
        results = mem.search_people("Ärztin", db_path=self.db)
        self.assertEqual(len(results), 1)

    def test_update_person(self):
        mem.save_person("Anna", "Freundin", db_path=self.db)
        ok = mem.update_person("Anna", "Kollegin", db_path=self.db)
        self.assertTrue(ok)
        self.assertEqual(mem.load_person("Anna", db_path=self.db).notes, "Kollegin")

    def test_update_nonexistent_returns_false(self):
        self.assertFalse(mem.update_person("Niemand", "X", db_path=self.db))

    def test_delete_person(self):
        mem.save_person("Anna", db_path=self.db)
        self.assertTrue(mem.delete_person("Anna", db_path=self.db))
        self.assertIsNone(mem.load_person("Anna", db_path=self.db))


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------


class TestNotes(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()

    def test_save_and_load(self):
        nid = mem.save_note("Idee", "JARVIS mit Gedächtnis", tags="projekt", db_path=self.db)
        n = mem.load_note(nid, db_path=self.db)
        self.assertIsNotNone(n)
        self.assertEqual(n.title, "Idee")

    def test_load_nonexistent_returns_none(self):
        self.assertIsNone(mem.load_note(9999, db_path=self.db))

    def test_list_notes(self):
        mem.save_note("A", db_path=self.db)
        mem.save_note("B", db_path=self.db)
        self.assertEqual(len(mem.list_notes(db_path=self.db)), 2)

    def test_list_by_tag(self):
        mem.save_note("X", tags="arbeit", db_path=self.db)
        mem.save_note("Y", tags="privat", db_path=self.db)
        self.assertEqual(len(mem.list_notes(tag="arbeit", db_path=self.db)), 1)

    def test_search_notes(self):
        mem.save_note("Einkauf", "Milch, Eier", db_path=self.db)
        mem.save_note("Meeting", "9 Uhr", db_path=self.db)
        results = mem.search_notes("Milch", db_path=self.db)
        self.assertEqual(len(results), 1)

    def test_update_note(self):
        nid = mem.save_note("Alt", "Inhalt", db_path=self.db)
        ok = mem.update_note(nid, title="Neu", db_path=self.db)
        self.assertTrue(ok)
        self.assertEqual(mem.load_note(nid, db_path=self.db).title, "Neu")

    def test_update_nonexistent_returns_false(self):
        self.assertFalse(mem.update_note(9999, title="X", db_path=self.db))

    def test_delete_note(self):
        nid = mem.save_note("Temp", db_path=self.db)
        self.assertTrue(mem.delete_note(nid, db_path=self.db))
        self.assertIsNone(mem.load_note(nid, db_path=self.db))


# ---------------------------------------------------------------------------
# Embedding-Vorbereitung
# ---------------------------------------------------------------------------


class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.db = _tmp_db()

    def test_set_embedding_on_memory(self):
        mem.save_memory("test_key", "Wert", db_path=self.db)
        dummy = b"\x00\x01\x02\x03"
        ok = mem.set_embedding("memories", "test_key", dummy, db_path=self.db)
        self.assertTrue(ok)
        with mem._connect(self.db) as conn:
            row = conn.execute(
                "SELECT embedding FROM memories WHERE key='test_key'"
            ).fetchone()
        self.assertEqual(row["embedding"], dummy)

    def test_set_embedding_invalid_table_raises(self):
        with self.assertRaises(mem.MemoryError):
            mem.set_embedding("invalid_table", 1, b"x", db_path=self.db)

    def test_set_embedding_on_note(self):
        nid = mem.save_note("Note", db_path=self.db)
        ok = mem.set_embedding("notes", nid, b"\xff", db_path=self.db)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
