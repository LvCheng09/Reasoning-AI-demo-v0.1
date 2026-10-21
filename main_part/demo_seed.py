"""Restore the portable, integrity-checked demo into an empty writable directory."""
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

from .persistence import rule_db, knowledge_db
from .rule_execution_contract import validate_rule_contract


def restore_demo_seed(pack: Path, destination: Path) -> dict[str, str]:
    manifest = json.loads((pack / "manifest.json").read_text())
    compressed = (pack / "seed.json.gz").read_bytes()
    if hashlib.sha256(compressed).hexdigest() != manifest["seed_sha256"]:
        raise ValueError("Demo seed checksum mismatch")
    payload = json.loads(gzip.decompress(compressed))
    if payload.get("schema") != "rule_network_demo.v2" or payload.get("learning_enabled") is not False:
        raise ValueError("Unsupported demo seed contract")
    rules, knowledge = payload["rules"], payload["knowledge"]
    if len(rules) != manifest["rules"] or len(knowledge) != manifest["knowledge"]:
        raise ValueError("Demo seed counts do not match the manifest")
    if len({r["rule_id"] for r in rules}) != len(rules):
        raise ValueError("Duplicate demo rule IDs")
    if len({k["knowledge_id"] for k in knowledge}) != len(knowledge):
        raise ValueError("Duplicate demo Knowledge IDs")
    for rule in rules:
        if rule.get("rule_layer") == "auto_learning" or rule.get("rule_kind") == "auto_learning":
            raise ValueError("Learning Rules cannot be loaded from an inference release seed")
        validate_rule_contract(rule)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {kind: str(destination / f"{kind}.sqlite") for kind in ("rule", "knowledge", "memory")}
    if any(Path(path).exists() for path in paths.values()):
        raise FileExistsError("Demo restore requires an empty destination")
    try:
        for kind in {r["rule_kind"] for r in rules}:
            rule_db.upsert_rule_records(paths["rule"], [r for r in rules if r["rule_kind"] == kind], rule_kind=kind)
        knowledge_db.upsert_knowledge_records(paths["knowledge"], knowledge)
        with sqlite3.connect(paths["memory"]) as memory:
            memory.execute("PRAGMA user_version=0")
        (destination / "seed_manifest.json").write_text(json.dumps({
            "source": "portable_demo_v2", "sha256": manifest["seed_sha256"]}, indent=2) + "\n")
        (destination / "activation_manifest.json").write_text(json.dumps(manifest["installed"], indent=2) + "\n")
    except Exception:
        # Failed restores must not leave a directory that looks usable.
        for path in paths.values():
            for suffix in ("", "-wal", "-shm"):
                Path(path + suffix).unlink(missing_ok=True)
        raise
    return paths
