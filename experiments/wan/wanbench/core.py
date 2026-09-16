"""No GPU or external packages required for preparation and integrity checks."""
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing run, plan or artifact accidentally.
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", value):
        raise ValueError(f"invalid identifier: {value!r}")
    return value


def positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def validate_device_selection(selection, visible_devices, logical_index):
    expected = selection.get("uuid")
    if not expected or not expected.startswith("GPU-"):
        raise ValueError("GPU UUID must be verified before running this campaign")
    if visible_devices != expected or logical_index != 0:
        raise ValueError(f"use CUDA_VISIBLE_DEVICES={expected} and --device 0; the user selected physical GPU 3")


def validate_campaign(campaign, frameworks, tasks):
    safe_id(campaign["id"])
    if campaign["condition"] not in {"native", "controlled"}:
        raise ValueError("condition must be native or controlled")
    if campaign["status"] != "draft":
        raise ValueError("v0 prepares draft campaigns only; formal runner is not implemented")
    positive_int(campaign["repeats"], "repeats")
    positive_int(campaign["wall_seconds"], "wall_seconds")
    if type(campaign["order_seed"]) is not int:
        raise ValueError("order_seed must be an integer")
    for field, known in [("frameworks", frameworks), ("tasks", tasks)]:
        values = campaign[field]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{field} must be a nonempty unique list")
        for value in values:
            safe_id(value)
            if value not in known:
                raise ValueError(f"unknown {field}: {value}")
    for field in ["token_limit", "candidate_limit"]:
        if campaign.get(field) is not None:
            positive_int(campaign[field], field)


def build_plan(campaign, frameworks, tasks):
    validate_campaign(campaign, frameworks, tasks)
    rng = random.Random(campaign["order_seed"])
    trials = []
    # Block randomization: one replicate contains every framework/task pair.
    for repeat in range(campaign["repeats"]):
        block = [(f, t) for f in campaign["frameworks"] for t in campaign["tasks"]]
        rng.shuffle(block)
        for framework, task in block:
            trials.append({
                "id": f"{framework}__{task}__r{repeat + 1:02}",
                "framework": framework, "task": task, "repeat": repeat + 1,
                "framework_commit": frameworks[framework]["commit"],
                "task_hash": object_digest(tasks[task]),
                "status": "planned", "latency_ms": None, "speedup": None,
            })
    return {"schema_version": 1, "campaign": campaign,
            "campaign_hash": object_digest(campaign), "trials": trials,
            "trial_wall_budget_hours": len(trials) * campaign["wall_seconds"] / 3600,
            "execution_enabled": False,
            "pending": ["B300 preflight", "framework adapters and model configuration",
                        "independent evaluator isolation", "real fixtures and frozen tolerances",
                        "token and candidate budgets", "recoverable remote backup"]}


def verify_sources(root, manifest):
    root = Path(root).resolve()
    checked = []
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("source path escapes repository")
        state = "missing" if not path.is_file() else (
            "ok" if digest(path) == row["sha256"] else "changed")
        checked.append({"path": row["path"], "status": state})
    return {"ok": bool(checked) and all(x["status"] == "ok" for x in checked),
            "files": checked}


def compare_smoke(baseline, candidate):
    """Reject invalid comparisons; this never emits a formal framework ranking."""
    for row in [baseline, candidate]:
        if row.get("kind") != "exploratory-gpu" or row.get("profile") != "none":
            raise ValueError("comparison requires unprofiled exploratory GPU measurements")
        if row.get("status") != "pass" or row.get("correctness", {}).get("passed") is not True:
            raise ValueError("both implementations must pass correctness")
        samples = row.get("samples_ms")
        if not isinstance(samples, list) or len(samples) < 2:
            raise ValueError("at least two timing samples required")
        if any(type(x) not in {float, int} or not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError("timings must be finite positive numbers")
    for key in ["task_hash", "fixture_hash", "environment_hash", "measurement_hash"]:
        if not baseline.get(key) or baseline[key] != candidate.get(key):
            raise ValueError(f"comparison mismatch: {key}")
    if baseline.get("variant") != "compile-default":
        raise ValueError("baseline must be native compile-default")
    if candidate.get("variant") != "candidate":
        raise ValueError("candidate must be a generated candidate, not another baseline")
    source_hash = candidate.get("candidate_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("candidate source hash required")
    b = statistics.median(baseline["samples_ms"])
    c = statistics.median(candidate["samples_ms"])
    return {"status": "exploratory-only", "baseline_p50_ms": b, "candidate_p50_ms": c,
            "speedup": b / c, "whole_block_speedup": None,
            "limitation": "synthetic inputs, no independent rounds or held-out validation; not a framework score"}
