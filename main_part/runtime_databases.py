"""Released seed databases -> independent writable inference workspace."""

from __future__ import annotations

import hashlib
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile


def prepare_runtime_databases(
    *, seed_dir: Path | None = None, runtime_root: Path | None = None
) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    seeds = (seed_dir or root / "data" / "demo" / "db").resolve()
    sources = {name: seeds / f"{name}.sqlite" for name in ("rule", "knowledge")}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(f"released seed database is missing: {path}")
    destination_root = (runtime_root or root / "runtime").expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="inference_", dir=destination_root))
    manifest = {}
    for name, source in sources.items():
        target = destination / source.name
        # A read-only source plus SQLite backup also handles a valid WAL snapshot.
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
            if original.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError(f"invalid seed database: {source}")
            with closing(sqlite3.connect(target)) as copied:
                original.backup(copied)
        manifest[name] = {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "runtime_file": target.name,
        }
    (destination / "seed_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {name: str(destination / f"{name}.sqlite") for name in ("rule", "knowledge", "memory")}


def default_db_paths() -> dict[str, str]:
    """MAIN_PART_DB_DIR means an explicitly managed writable runtime, not seeds."""
    configured = os.getenv("MAIN_PART_DB_DIR", "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        bundled = Path(__file__).resolve().parents[1] / "data" / "demo" / "db"
        if path == bundled.resolve():
            raise ValueError("MAIN_PART_DB_DIR must not point at the released seed databases")
        for name in ("rule", "knowledge"):
            if not (path / f"{name}.sqlite").is_file():
                raise FileNotFoundError(f"configured runtime database is missing: {path / (name + '.sqlite')}")
        return {name: str(path / f"{name}.sqlite") for name in ("rule", "knowledge", "memory")}
    configured_root = os.getenv("MAIN_PART_RUNTIME_DIR", "").strip()
    return prepare_runtime_databases(runtime_root=Path(configured_root) if configured_root else None)
