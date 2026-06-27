"""Langzeitgedächtnis für JARVIS über SQLite.

Tabellen:
- conversations  Gespeicherte Gesprächsverläufe
- memories       Einzelne Erinnerungen / Fakten
- settings       Persistente Einstellungen
- people         Personen, die JARVIS kennt
- notes          Notizen

Alle CRUD-Operationen sowie eine Vorbereitung für semantische Suche
(Embedding-Spalte bereits vorhanden, noch nicht befüllt).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional


DB_PATH = Path(__file__).with_name("jarvis.db")

# ---------------------------------------------------------------------------
# Fehlerklasse
# ---------------------------------------------------------------------------


class MemoryError(RuntimeError):
    """Wird bei Datenbankfehlern ausgelöst."""


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class Conversation:
    id: int
    title: str
    messages: str  # JSON-kodierte Nachrichtenliste
    created_at: str
    updated_at: str


@dataclass
class Memory:
    id: int
    key: str
    value: str
    category: str
    embedding: Optional[bytes]  # Vorbereitung für semantische Suche
    created_at: str
    updated_at: str


@dataclass
class Setting:
    key: str
    value: str
    updated_at: str


@dataclass
class Person:
    id: int
    name: str
    notes: str
    embedding: Optional[bytes]  # Vorbereitung für semantische Suche
    created_at: str
    updated_at: str


@dataclass
class Note:
    id: int
    title: str
    content: str
    tags: str  # kommagetrennte Tags
    embedding: Optional[bytes]  # Vorbereitung für semantische Suche
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Datenbank-Verbindung
# ---------------------------------------------------------------------------


@contextmanager
def _connect(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema erstellen
# ---------------------------------------------------------------------------


def init_db(db_path: Path = DB_PATH) -> None:
    """Erstellt alle Tabellen, falls sie noch nicht existieren."""
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT    NOT NULL DEFAULT '',
                messages    TEXT    NOT NULL DEFAULT '[]',
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                key         TEXT    NOT NULL,
                value       TEXT    NOT NULL,
                category    TEXT    NOT NULL DEFAULT 'general',
                embedding   BLOB,
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL,
                UNIQUE(key)
            );

            CREATE TABLE IF NOT EXISTS settings (
                key         TEXT    PRIMARY KEY,
                value       TEXT    NOT NULL DEFAULT '',
                updated_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS people (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                notes       TEXT    NOT NULL DEFAULT '',
                embedding   BLOB,
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL,
                UNIQUE(name)
            );

            CREATE TABLE IF NOT EXISTS notes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT    NOT NULL,
                content     TEXT    NOT NULL DEFAULT '',
                tags        TEXT    NOT NULL DEFAULT '',
                embedding   BLOB,
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );
            """
        )


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------


def save_conversation(
    messages_json: str,
    title: str = "",
    db_path: Path = DB_PATH,
) -> int:
    """Speichert einen Gesprächsverlauf und gibt die neue ID zurück."""
    now = _now()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title, messages, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (title, messages_json, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def load_conversation(conv_id: int, db_path: Path = DB_PATH) -> Optional[Conversation]:
    """Lädt einen Gesprächsverlauf anhand der ID."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
    if row is None:
        return None
    return Conversation(**dict(row))


def list_conversations(db_path: Path = DB_PATH) -> list[Conversation]:
    """Gibt alle Gesprächsverläufe zurück (neueste zuerst)."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [Conversation(**dict(r)) for r in rows]


def search_conversations(query: str, db_path: Path = DB_PATH) -> list[Conversation]:
    """Sucht in Titel und Nachrichten (einfache Volltextsuche)."""
    pattern = f"%{query}%"
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM conversations "
            "WHERE title LIKE ? OR messages LIKE ? "
            "ORDER BY updated_at DESC",
            (pattern, pattern),
        ).fetchall()
    return [Conversation(**dict(r)) for r in rows]


def update_conversation(
    conv_id: int,
    messages_json: str | None = None,
    title: str | None = None,
    db_path: Path = DB_PATH,
) -> bool:
    """Aktualisiert Titel und/oder Nachrichten eines Gesprächs."""
    now = _now()
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if row is None:
            return False
        new_messages = messages_json if messages_json is not None else row["messages"]
        new_title = title if title is not None else row["title"]
        conn.execute(
            "UPDATE conversations SET messages=?, title=?, updated_at=? WHERE id=?",
            (new_messages, new_title, now, conv_id),
        )
    return True


def delete_conversation(conv_id: int, db_path: Path = DB_PATH) -> bool:
    """Löscht einen Gesprächsverlauf."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# memories
# ---------------------------------------------------------------------------


def save_memory(
    key: str,
    value: str,
    category: str = "general",
    db_path: Path = DB_PATH,
) -> int:
    """Speichert eine Erinnerung (upsert über key)."""
    now = _now()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO memories (key, value, category, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value      = excluded.value,
                category   = excluded.category,
                updated_at = excluded.updated_at
            """,
            (key, value, category, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def load_memory(key: str, db_path: Path = DB_PATH) -> Optional[Memory]:
    """Lädt eine Erinnerung anhand des Schlüssels."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM memories WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return None
    return Memory(**dict(row))


def list_memories(
    category: str | None = None, db_path: Path = DB_PATH
) -> list[Memory]:
    """Listet alle Erinnerungen, optional gefiltert nach Kategorie."""
    with _connect(db_path) as conn:
        if category:
            rows = conn.execute(
                "SELECT * FROM memories WHERE category = ? ORDER BY updated_at DESC",
                (category,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY updated_at DESC"
            ).fetchall()
    return [Memory(**dict(r)) for r in rows]


def search_memories(query: str, db_path: Path = DB_PATH) -> list[Memory]:
    """Sucht in Schlüssel und Wert."""
    pattern = f"%{query}%"
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM memories WHERE key LIKE ? OR value LIKE ? "
            "ORDER BY updated_at DESC",
            (pattern, pattern),
        ).fetchall()
    return [Memory(**dict(r)) for r in rows]


def update_memory(
    key: str,
    value: str | None = None,
    category: str | None = None,
    db_path: Path = DB_PATH,
) -> bool:
    """Aktualisiert eine bestehende Erinnerung."""
    now = _now()
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM memories WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return False
        new_value = value if value is not None else row["value"]
        new_cat = category if category is not None else row["category"]
        conn.execute(
            "UPDATE memories SET value=?, category=?, updated_at=? WHERE key=?",
            (new_value, new_cat, now, key),
        )
    return True


def delete_memory(key: str, db_path: Path = DB_PATH) -> bool:
    """Löscht eine Erinnerung anhand des Schlüssels."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


def set_setting(key: str, value: str, db_path: Path = DB_PATH) -> None:
    """Speichert oder überschreibt eine Einstellung."""
    now = _now()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, value, now),
        )


def get_setting(key: str, default: str = "", db_path: Path = DB_PATH) -> str:
    """Liest eine Einstellung."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else default


def list_settings(db_path: Path = DB_PATH) -> list[Setting]:
    """Gibt alle Einstellungen zurück."""
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM settings ORDER BY key").fetchall()
    return [Setting(**dict(r)) for r in rows]


def delete_setting(key: str, db_path: Path = DB_PATH) -> bool:
    """Löscht eine Einstellung."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM settings WHERE key = ?", (key,))
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# people
# ---------------------------------------------------------------------------


def save_person(name: str, notes: str = "", db_path: Path = DB_PATH) -> int:
    """Speichert eine Person (upsert über name)."""
    now = _now()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO people (name, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                notes      = excluded.notes,
                updated_at = excluded.updated_at
            """,
            (name, notes, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def load_person(name: str, db_path: Path = DB_PATH) -> Optional[Person]:
    """Lädt eine Person anhand des Namens."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM people WHERE name = ?", (name,)
        ).fetchone()
    if row is None:
        return None
    return Person(**dict(row))


def list_people(db_path: Path = DB_PATH) -> list[Person]:
    """Listet alle bekannten Personen."""
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM people ORDER BY name").fetchall()
    return [Person(**dict(r)) for r in rows]


def search_people(query: str, db_path: Path = DB_PATH) -> list[Person]:
    """Sucht in Name und Notizen."""
    pattern = f"%{query}%"
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM people WHERE name LIKE ? OR notes LIKE ? ORDER BY name",
            (pattern, pattern),
        ).fetchall()
    return [Person(**dict(r)) for r in rows]


def update_person(
    name: str, notes: str, db_path: Path = DB_PATH
) -> bool:
    """Aktualisiert die Notizen zu einer Person."""
    now = _now()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE people SET notes=?, updated_at=? WHERE name=?",
            (notes, now, name),
        )
    return cur.rowcount > 0


def delete_person(name: str, db_path: Path = DB_PATH) -> bool:
    """Löscht eine Person."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM people WHERE name = ?", (name,))
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------


def save_note(
    title: str,
    content: str = "",
    tags: str = "",
    db_path: Path = DB_PATH,
) -> int:
    """Speichert eine neue Notiz und gibt die ID zurück."""
    now = _now()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO notes (title, content, tags, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (title, content, tags, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def load_note(note_id: int, db_path: Path = DB_PATH) -> Optional[Note]:
    """Lädt eine Notiz anhand der ID."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
    if row is None:
        return None
    return Note(**dict(row))


def list_notes(tag: str | None = None, db_path: Path = DB_PATH) -> list[Note]:
    """Listet alle Notizen, optional gefiltert nach Tag."""
    with _connect(db_path) as conn:
        if tag:
            rows = conn.execute(
                "SELECT * FROM notes WHERE tags LIKE ? ORDER BY updated_at DESC",
                (f"%{tag}%",),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM notes ORDER BY updated_at DESC"
            ).fetchall()
    return [Note(**dict(r)) for r in rows]


def search_notes(query: str, db_path: Path = DB_PATH) -> list[Note]:
    """Sucht in Titel, Inhalt und Tags."""
    pattern = f"%{query}%"
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM notes "
            "WHERE title LIKE ? OR content LIKE ? OR tags LIKE ? "
            "ORDER BY updated_at DESC",
            (pattern, pattern, pattern),
        ).fetchall()
    return [Note(**dict(r)) for r in rows]


def update_note(
    note_id: int,
    title: str | None = None,
    content: str | None = None,
    tags: str | None = None,
    db_path: Path = DB_PATH,
) -> bool:
    """Aktualisiert eine Notiz."""
    now = _now()
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE notes SET title=?, content=?, tags=?, updated_at=? WHERE id=?",
            (
                title if title is not None else row["title"],
                content if content is not None else row["content"],
                tags if tags is not None else row["tags"],
                now,
                note_id,
            ),
        )
    return True


def delete_note(note_id: int, db_path: Path = DB_PATH) -> bool:
    """Löscht eine Notiz."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Vorbereitung semantische Suche
# ---------------------------------------------------------------------------


def set_embedding(
    table: str,
    record_id: int | str,
    embedding: bytes,
    db_path: Path = DB_PATH,
) -> bool:
    """
    Speichert ein vorberechnetes Embedding in der jeweiligen Tabelle.

    Diese Funktion ist eine Vorbereitung für Phase 6+ (semantische Suche).
    Das Embedding-Blob wird gespeichert, aber noch nicht für die Suche genutzt.

    table: 'memories' | 'people' | 'notes'
    record_id: id (int) für memories/people/notes, key (str) für memories
    """
    allowed = {"memories", "people", "notes"}
    if table not in allowed:
        raise MemoryError(f"Ungültige Tabelle: {table}. Erlaubt: {allowed}")

    id_column = "key" if (table == "memories" and isinstance(record_id, str)) else "id"
    now = _now()

    with _connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE {table} SET embedding=?, updated_at=? WHERE {id_column}=?",
            (embedding, now, record_id),
        )
    return cur.rowcount > 0
