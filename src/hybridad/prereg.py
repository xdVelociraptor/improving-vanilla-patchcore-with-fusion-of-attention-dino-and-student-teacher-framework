"""Pre-registration of the experiment suite and an append-only log of test-set access.

`preregister` freezes, for every experiment, the effective branch variants, fusion rule and
calibration settings, plus a hash of the source code. Confirmatory evaluation requires the
current configuration to match the frozen one; anything else must be run with
`--exploratory` and is labelled as exploratory in every output. Every test evaluation
(confirmatory or exploratory) is appended to TEST_ACCESS_LOG.jsonl so the number of test
inspections is auditable.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from .config import project_path, stable_hash, PROJECT_ROOT
from .utils import write_json, read_json


class PreregistrationError(RuntimeError):
    pass


# Modules that cannot change any score, calibration, prediction or metric (presentation / tooling only).
PRESENTATION_ONLY = {"report.py", "visualize.py", "cli.py", "synthetic.py"}


def code_hash() -> str:
    """Hash of every module that can change scores, calibration, predictions or metrics."""
    h = hashlib.sha256()
    for p in sorted((PROJECT_ROOT / "src" / "hybridad").rglob("*.py")):
        if p.name in PRESENTATION_ONLY and p.parent.name == "hybridad":
            continue
        h.update(p.relative_to(PROJECT_ROOT).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def frozen_spec(cfg: dict, dataset: str, category: str) -> dict:
    from .pipeline import variant_id, branch_enabled
    return {
        "experiment": cfg["experiment"]["name"],
        "fusion": cfg["experiment"]["fusion"],
        "calibration": cfg["calibration"],
        "protocol": cfg["protocol"],
        "branches": {slot: (variant_id(bc, dataset, category) if branch_enabled(bc, cfg, dataset, category)[0]
                            else "disabled") for slot, bc in cfg["experiment"]["branch_cfgs"].items()},
    }


def prereg_path(cfg: dict) -> Path:
    return project_path(cfg["paths"]["runs"]) / "PREREGISTRATION.json"


def preregister(cfgs: list[dict], datasets: dict[str, list[str]], note: str) -> dict:
    path = prereg_path(cfgs[0])
    if path.exists():
        raise PreregistrationError(f"{path} already exists. Pre-registration is write-once; create a new "
                                   "version deliberately (rename the old file and state why) if the plan changed.")
    entries = {}
    for cfg in cfgs:
        name = cfg["experiment"]["name"]
        entries[name] = {"source": cfg["_source"], "specs": {
            f"{d}/{c}": stable_hash(frozen_spec(cfg, d, c), n=24) for d, cats in datasets.items() for c in cats}}
    rec = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "code_hash": code_hash(), "note": note,
           "experiments": entries,
           "rule": "confirmatory evaluation only for configurations whose spec hash matches this file"}
    write_json(path, rec)
    return rec


def check(cfg: dict, dataset: str, category: str, exploratory: bool) -> dict:
    path = prereg_path(cfg)
    status = {"confirmatory": False, "reasons": []}
    if not path.exists():
        status["reasons"].append("no pre-registration file")
    else:
        rec = read_json(path)
        e = rec["experiments"].get(cfg["experiment"]["name"])
        key = f"{dataset}/{category}"
        if e is None:
            status["reasons"].append("experiment not pre-registered")
        elif e["specs"].get(key) != stable_hash(frozen_spec(cfg, dataset, category), n=24):
            status["reasons"].append(f"configuration for {key} differs from the pre-registered one")
        if rec["code_hash"] != code_hash():
            status["reasons"].append("source code changed since pre-registration (hash differs)")
    status["confirmatory"] = not status["reasons"]
    if not status["confirmatory"] and not exploratory:
        raise PreregistrationError("confirmatory evaluation refused: " + "; ".join(status["reasons"]) +
                                   ". Re-run with --exploratory to label the results as exploratory.")
    return status


def log_test_access(cfg: dict, dataset: str, categories: list[str], status: dict) -> None:
    path = project_path(cfg["paths"]["runs"]) / "TEST_ACCESS_LOG.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "experiment": cfg["experiment"]["name"], "dataset": dataset,
                            "categories": categories, "confirmatory": status["confirmatory"],
                            "reasons": status["reasons"]}) + "\n")
