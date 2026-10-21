"""The released inference registry is independent of private author databases."""
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from main_part.demo_seed import restore_demo_seed
from main_part.persistence import rule_db, knowledge_db

PACK = Path(__file__).resolve().parents[1] / "data/demo_v2"


def test_inference_seed_roundtrip_and_isolation(tmp_path):
    before = (PACK / "seed.json.gz").read_bytes()
    manifest = json.loads((PACK / "manifest.json").read_text())
    payload = json.loads(gzip.decompress(before))
    assert hashlib.sha256(before).hexdigest() == manifest["seed_sha256"]
    assert payload["learning_enabled"] is False
    assert not any(r["rule_layer"] == "auto_learning" for r in payload["rules"])
    paths = restore_demo_seed(PACK, tmp_path / "runtime")
    assert len(rule_db.load_rule_records(paths["rule"])) == manifest["rules"]
    assert len(knowledge_db.load_knowledge_records(paths["knowledge"])) == manifest["knowledge"]
    with sqlite3.connect(paths["memory"]) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
    assert (PACK / "seed.json.gz").read_bytes() == before
    with pytest.raises(FileExistsError):
        restore_demo_seed(PACK, tmp_path / "runtime")


def test_checksum_failure_does_not_create_runtime(tmp_path):
    damaged = tmp_path / "damaged"
    damaged.mkdir()
    (damaged / "manifest.json").write_bytes((PACK / "manifest.json").read_bytes())
    (damaged / "seed.json.gz").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        restore_demo_seed(damaged, tmp_path / "runtime")
    assert not (tmp_path / "runtime").exists()


def test_learning_implementation_is_not_published():
    from main_part.model_launcher import MainPartModelLauncher
    from main_part.scheduler_new import SchedulerNew
    root = PACK.parents[1]
    assert not (root / "main_part/auto_learn").exists()
    assert not hasattr(SchedulerNew, "submit_supervised_learning_feedback")
    assert not hasattr(MainPartModelLauncher, "install_learning_rules")
