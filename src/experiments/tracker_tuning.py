"""Tune BoT-SORT for existing, frozen detector checkpoints.

Run this module with --dry-run first. The default wall-time budget
is 168 active hours, including inference and final test evaluation. Search uses
balanced rounds, seeded random exploration and local refinement. No training or
checkpoint selection is performed. Selection uses validation by default;
--tuning-split train searches all training sequences, excluding validation;
--tuning-split test enables a separate diagnostic search on both test sequences.
"""

from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import importlib.metadata
import json
import multiprocessing
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.experiments.tracker_tuning_runtime import (
    METRICS,
    TEST_SEQUENCES,
    VAL_SEQUENCE,
    atomic_json,
    check_disk,
    digest,
    evaluate,
    export_cache,
    file_hash,
    file_identity,
    initialize_worker,
    read_json,
    split_sequences,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = "results/dsec_mot_tracker_tuning"
DEFAULT_TRAIN_OUTPUT = "results/dsec_mot_tracker_tuning_train"
DEFAULT_TEST_OUTPUT = "results/dsec_mot_tracker_tuning_test"
BASE_PARAMS = {
    "score": 0.9,
    "high": 0.6,
    "low": 0.1,
    "buffer": 2,
    "match": 0.5,
    "appearance": 0.25,
    "proximity": 0.5,
}
GRIDS = {
    "score": [round(i / 20, 2) for i in range(1, 20)] + [0.97, 0.99],
    "high": [round(i / 20, 2) for i in range(6, 20)] + [0.97, 0.99],
    "low": [round(i / 20, 2) for i in range(1, 10)],
    "buffer": [2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120],
    "match": [round(i / 20, 2) for i in range(6, 20)],
    "appearance": [round(i / 20, 2) for i in range(1, 20)],
    "proximity": [round(i / 20, 2) for i in range(1, 20)],
}
PLAIN_ROOTS = {
    "simple_detector": "representations",
    "simple_detector_sweep": "fusion_sweep",
    "simple_detector_car_only": "representations_car",
    "simple_detector_large": "larger_models",
    "simple_detector_model_series": "model_architectures",
}
SELECTION_ROOTS = {
    "event_reid_embedding_frozen": "roi_frozen",
    "dense_event_reid_embedding_frozen": "dense_frozen",
}


def discover(repo: Path) -> list[dict]:
    """Keep every previously selected variant, never rank using test metrics."""
    entries: dict[tuple[str, str], dict] = {}

    def add(
        checkpoint: Path,
        family: str,
        label: str,
        source: Path,
        normalisation: str | None = None,
        selection: dict | None = None,
    ) -> None:
        checkpoint = checkpoint if checkpoint.is_absolute() else repo / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Selected checkpoint missing: {checkpoint} (source {source})")
        config_path = checkpoint.parent / "config.json"
        config = read_json(config_path)
        model = config.get("model", config.get("model_config", {}))
        num_classes = int(model["num_classes"])
        if num_classes not in (1, 7):
            raise ValueError(f"Unsupported DSEC-MOT class scope in {config_path}")
        if config.get("val_sequences") != [VAL_SEQUENCE]:
            raise ValueError(f"Unexpected validation sequence in {config_path}")
        if set(config.get("train_sequences", ())) & {VAL_SEQUENCE, *TEST_SEQUENCES}:
            raise ValueError(f"Training/validation/test overlap in {config_path}")
        embedding_dim = int(model.get("embedding_dim", 0))
        normalisation = normalisation or ("component" if embedding_dim else "whole")
        relative = str(checkpoint.relative_to(repo))
        key = (relative, normalisation)
        scope = "car_only" if num_classes == 1 else "all_classes"
        alias = {"family": family, "label": label, "source": str(source.relative_to(repo))}
        if key not in entries:
            identifier = f"{scope}-{family}-{digest(key)[:12]}"
            entries[key] = {
                "id": identifier,
                "checkpoint": str(checkpoint),
                "checkpoint_relative": relative,
                "checkpoint_identity": file_identity(checkpoint),
                "config_identity": file_identity(config_path),
                "scope": scope,
                "num_classes": num_classes,
                "embedding_dim": embedding_dim,
                "representation": config["representation"],
                "num_bins": config["num_bins"],
                "time_window_us": config["time_window_us"],
                "normalisation": normalisation,
                "model": model,
                "aliases": [],
                "previous_selections": [],
            }
        if alias not in entries[key]["aliases"]:
            entries[key]["aliases"].append(alias)
        if selection:
            # Deliberately whitelist validation/configuration fields; never copy test metrics.
            allowed = (
                "selected_epoch",
                "epoch",
                "score_threshold",
                "appearance_threshold",
                "proximity_threshold",
                "selection_policy",
                "validation_metrics",
                "checkpoint_validation_metrics",
                "association_validation_metrics",
            )
            previous = {k: selection[k] for k in allowed if selection.get(k) is not None}
            previous["source"] = str(source.relative_to(repo))
            if previous not in entries[key]["previous_selections"]:
                entries[key]["previous_selections"].append(previous)

    for directory, family in PLAIN_ROOTS.items():
        for checkpoint in sorted((repo / "runs" / directory).rglob("best.pt")):
            add(
                checkpoint,
                family,
                str(checkpoint.parent.relative_to(repo / "runs" / directory)),
                checkpoint.parent / "config.json",
            )
    joint = repo / "results/dsec_mot_event_reid_embedding/benchmark_summary.json"
    if joint.exists():
        for record in read_json(joint)["records"]:
            add(Path(record["checkpoint"]), "roi_joint", record["variant"], joint, selection=record)
    for directory, family in SELECTION_ROOTS.items():
        for source in sorted((repo / "runs" / directory).rglob("best_hota_selection.json")):
            add(
                source.parent / "best_hota.pt",
                family,
                str(source.parent.relative_to(repo / "runs" / directory)),
                source,
                normalisation="component",
                selection=read_json(source),
            )
    for directory in (
        "dsec_mot_recurrent_placement_ablation",
        "dsec_mot_novelty_contribution_benchmark",
    ):
        for source in sorted((repo / "results" / directory).rglob("variant_summary.json")):
            record = read_json(source)
            variant = record["variant"]
            label = variant["name"] if isinstance(variant, dict) else variant
            family = (
                variant.get("suite", "recurrence") if isinstance(variant, dict) else "recurrence"
            )
            add(
                Path(record["selected_checkpoint"]),
                family,
                label,
                source,
                normalisation="component",
                selection=record,
            )
    return sorted(entries.values(), key=lambda entry: entry["id"])


def modes(entry: dict) -> list[str]:
    return ["motion", "reid"] if entry["embedding_dim"] else ["motion"]


def canonical(params: dict, mode: str) -> dict:
    result = dict(params)
    if mode == "motion":
        result.update(appearance=0.25, proximity=0.5)
    return result


def import_reference_params(entries: list[dict], args: dict) -> dict | None:
    """Import only parameters from a matching frozen run; re-evaluate every anchor."""
    reference = args.get("reference_selection")
    if not reference:
        return None
    path = Path(reference).resolve()
    if Path(args["output_root"]).resolve() == path.parent:
        raise ValueError("Reference results require a different --output-root")
    manifest_path = path.parent / "manifest.json"
    manifest = read_json(manifest_path)
    selection = read_json(path)
    if selection["manifest_sha256"] != digest(manifest):
        raise RuntimeError("Reference selection belongs to a different manifest")
    protocols = {
        (entry["checkpoint_relative"], entry["normalisation"]): entry
        for entry in manifest["entries"]
    }
    for entry in entries:
        source = protocols.get((entry["checkpoint_relative"], entry["normalisation"]))
        fields = (
            "checkpoint_identity",
            "config_identity",
            "scope",
            "num_classes",
            "embedding_dim",
            "representation",
            "num_bins",
            "time_window_us",
            "model",
        )
        if source is None or any(source[key] != entry[key] for key in fields):
            raise ValueError(f"Reference checkpoint/protocol mismatch: {entry['id']}")
        imported = {}
        for mode in modes(entry):
            params = dict(selection["selected"][source["id"]][mode]["params"])
            if (
                set(params) != set(BASE_PARAMS)
                or any(params[key] not in GRIDS[key] for key in GRIDS)
                or params["low"] >= params["high"]
            ):
                raise ValueError(f"Invalid reference parameters: {entry['id']}/{mode}")
            imported[mode] = canonical(params, mode)
        entry["reference_params"] = imported
    return {**file_identity(path), "sha256": file_hash(path), "manifest_sha256": digest(manifest)}


def anchors(entry: dict, mode: str) -> list[dict]:
    candidates = [dict(BASE_PARAMS)]
    for score in (0.05, 0.9, 0.95):
        candidates += [
            dict(BASE_PARAMS, score=score, high=0.95, low=0.25, appearance=0.4, proximity=0.7),
            dict(
                BASE_PARAMS,
                score=score,
                high=0.95,
                low=0.25,
                buffer=10,
                appearance=0.25,
                proximity=0.8,
            ),
        ]
    for buffer in (10, 30, 60):
        candidates.append(dict(BASE_PARAMS, buffer=buffer))
        candidates.append(dict(BASE_PARAMS, score=0.05, buffer=buffer))
    for previous in entry.get("previous_selections", []):
        candidate = dict(BASE_PARAMS)
        for source, target in (
            ("score_threshold", "score"),
            ("appearance_threshold", "appearance"),
            ("proximity_threshold", "proximity"),
        ):
            if source in previous:
                candidate[target] = previous[source]
        candidates.append(candidate)
    if mode in entry.get("reference_params", {}):
        candidates.append(entry["reference_params"][mode])
    unique = {}
    for candidate in candidates:
        candidate = canonical(candidate, mode)
        unique.setdefault(digest(candidate), candidate)
    return list(unique.values())


def metric_key(record: dict) -> tuple:
    metrics = record["metrics"]
    return (metrics["HOTA"], metrics["AssA"], metrics["IDF1"])


def best_trial(records: list[dict]) -> dict:
    # Rank completed trial metrics; the candidate hash is the deterministic tie-break.
    return sorted(
        records,
        key=lambda record: tuple(-v for v in metric_key(record)) + (digest(record["params"]),),
    )[0]


def propose(entry: dict, mode: str, history: list[dict], seed: int) -> dict:
    seen = {digest(record["params"]) for record in history}
    for candidate in anchors(entry, mode):
        if digest(candidate) not in seen:
            return candidate
    # Shared global geometry stream for every checkpoint and both association modes.
    rng = random.Random(f"{seed}:{len(history)}")
    fields = list(GRIDS) if mode == "reid" else list(GRIDS)[:5]
    for attempt in range(10000):
        candidate = {key: rng.choice(values) for key, values in GRIDS.items()}
        if len(history) % 4 == 3 and attempt < 100:
            elite = sorted(history, key=metric_key, reverse=True)[:5]
            candidate = dict(rng.choice(elite)["params"])
            for key in rng.sample(fields, k=min(2, len(fields))):
                grid = GRIDS[key]
                index = min(range(len(grid)), key=lambda i: abs(grid[i] - candidate[key]))
                candidate[key] = grid[max(0, min(len(grid) - 1, index + rng.choice([-1, 1])))]
        candidate = canonical(candidate, mode)
        if candidate["low"] >= candidate["high"]:
            continue
        if digest(candidate) not in seen:
            return candidate
    raise RuntimeError("Could not generate an untested tracker configuration")


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_bytes()
    # A killed append may leave one partial trailing line. Only complete lines count.
    return [json.loads(line) for line in data[: data.rfind(b"\n") + 1].splitlines()]


def append_history(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("rb+") as stream:
            data = stream.read()
            if data and not data.endswith(b"\n"):
                stream.truncate(data.rfind(b"\n") + 1)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def search_batch(entry: dict, mode: str, target: int, args: dict) -> dict:
    path = Path(args["output_root"]) / "trials" / entry["id"] / f"{mode}.jsonl"
    history = read_history(path)
    while len(history) < target:
        if (Path(args["output_root"]) / "STOP.request").exists():
            return {"interrupted": True}
        params = propose(entry, mode, history, args["seed"])
        try:
            record = evaluate(entry, mode, params, args.get("tuning_split", "val"), args)
        except Exception:
            atomic_json(
                path.parent / f"{mode}_failure.json",
                {
                    "entry_id": entry["id"],
                    "mode": mode,
                    "params": params,
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        record.update(entry_id=entry["id"], trial_index=len(history), candidate_id=digest(params))
        append_history(path, record)
        history.append(record)
    return {
        "entry_id": entry["id"],
        "mode": mode,
        "trials": len(history),
        "best": best_trial(history),
        "total_trial_s": sum(r["elapsed_s"] for r in history),
    }


def runtime_source_files(repo: Path) -> list[Path]:
    """Follow local imports, including function imports and package initializers."""
    pending: set[Path] = set()
    visited: set[Path] = set()

    def add_module(module: str) -> None:
        if module != "src" and not module.startswith("src."):
            return
        parts = module.split(".")
        candidates = [repo.joinpath(*parts).with_suffix(".py")]
        candidates.extend(
            repo.joinpath(*parts[:depth], "__init__.py") for depth in range(1, len(parts) + 1)
        )
        pending.update(path for path in candidates if path.is_file() and path not in visited)

    add_module("src.experiments.tracker_tuning")
    add_module("src.experiments.tracker_tuning_runtime")
    while pending:
        path = pending.pop()
        visited.add(path)
        package = list(path.relative_to(repo).parent.parts)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[: len(package) - node.level + 1] if node.level else []
                module = ".".join(prefix + (node.module.split(".") if node.module else []))
                add_module(module)
                for alias in node.names:
                    if alias.name != "*":
                        add_module(f"{module}.{alias.name}")
    return sorted(visited)


def runtime_identity(repo: Path, trackeval_root: Path) -> dict:
    files = runtime_source_files(repo)
    files += sorted((trackeval_root / "trackeval").rglob("*.py"))
    code = {
        str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path): file_hash(path)
        for path in files
    }
    packages = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "torch", "torchvision", "boxmot")
    }
    # Local package edits matter even if the installed version number did not change.
    distribution = importlib.metadata.distribution("boxmot")
    for relative in distribution.files or ():
        if str(relative).endswith(".py"):
            path = Path(distribution.locate_file(relative))
            code[f"installed/{relative}"] = file_hash(path)
    return {
        "packages": packages,
        "code_scope": "tracker_tuning_dependencies",
        "code_sha256": digest(code),
        "code_files": code,
    }


def build_manifest(entries: list[dict], args: dict) -> dict:
    tuning_split = args.get("tuning_split", "val")
    tuning_sequences = split_sequences(tuning_split, args)
    eros_identity = {}
    if any("eros" in entry["representation"] for entry in entries):
        train_sequences = tuning_sequences if tuning_split == "train" else [VAL_SEQUENCE]
        for split, sequences in (("train", train_sequences), ("test", TEST_SEQUENCES)):
            for sequence in sequences:
                for name in ("metadata.json", "snapshots.npy"):
                    path = Path(args["eros_cache_root"]) / split / sequence / name
                    eros_identity[str(path)] = file_identity(path)
    manifest = {
        "version": 1,
        "eros_identity": eros_identity,
        "validation_sequence": VAL_SEQUENCE,
        "test_sequences": list(TEST_SEQUENCES),
        "selection_policy": [
            "validation HOTA",
            "validation AssA",
            "validation IDF1",
            "canonical parameter hash",
        ],
        "search_space": GRIDS,
        "seed": args["seed"],
        "entries": entries,
        "protocol": {
            key: args[key]
            for key in (
                "root",
                "eros_cache_root",
                "trackeval_root",
                "runtime_identity",
                "dataset_identity",
            )
        },
        "notes": [
            "Existing checkpoint selection is frozen; no retraining.",
            "Supervisor settings are exploratory anchors informed by prior test results.",
            "The test split has been used in earlier research; it is not a new holdout.",
            "Normalisation stays fixed within each checkpoint protocol.",
            "NMS uses stable cell order for equal scores; "
            "historical default tie order is unchanged.",
            "Global search is approximate; best means best completed validation trial.",
        ],
    }
    if tuning_split == "train":
        manifest.update(
            tuning_split="train",
            tuning_sequences=tuning_sequences,
            selection_policy=[
                "combined training HOTA",
                "combined training AssA",
                "combined training IDF1",
                "canonical parameter hash",
            ],
        )
        manifest["notes"][-1] = (
            "Global search is approximate; best means best completed combined-training trial."
        )
        manifest["notes"].append(
            "Tracker parameters are selected on all training sequences, excluding validation "
            "and test sequences; final test evaluation starts only after selection is frozen."
        )
    if tuning_split == "test":
        manifest.update(
            tuning_split="test",
            tuning_sequences=list(TEST_SEQUENCES),
            evaluation_is_independent=False,
            selection_policy=[
                "combined test HOTA",
                "combined test AssA",
                "combined test IDF1",
                "canonical parameter hash",
            ],
        )
        manifest["notes"][-1] = (
            "Global search is approximate; best means best completed combined-test trial."
        )
        manifest["notes"].append(
            "Diagnostic: parameters are selected and evaluated on the same two test "
            "sequences; these results are not an independent test."
        )
    if args.get("reference_identity") is not None:
        manifest["protocol"]["reference_selection"] = args["reference_identity"]
    return manifest


def final_cases(entry: dict, selections: dict) -> dict:
    result = {
        "baseline_motion": {"mode": "motion", "params": dict(BASE_PARAMS)},
        "tuned_motion": {"mode": "motion", "params": selections["motion"]["params"]},
    }
    if entry["embedding_dim"]:
        reid = selections["reid"]["params"]
        result.update(
            baseline_reid={"mode": "reid", "params": dict(BASE_PARAMS)},
            tuned_reid={"mode": "reid", "params": reid},
            reid_settings_without_embeddings={
                "mode": "motion",
                "params": canonical(reid, "motion"),
            },
        )
    return result


def freeze_selection(entries: list[dict], args: dict, manifest: dict) -> dict:
    output = Path(args["output_root"])
    path = output / "selection.json"
    if path.exists():
        selection = read_json(path)
        if selection["manifest_sha256"] != digest(manifest):
            raise RuntimeError("Frozen selection belongs to a different manifest")
        return selection
    selected = {}
    tuning_split = args.get("tuning_split", "val")
    for entry in entries:
        selected[entry["id"]] = {}
        for mode in modes(entry):
            history = read_history(output / "trials" / entry["id"] / f"{mode}.jsonl")
            if not history:
                raise RuntimeError(f"No completed {tuning_split} trials for {entry['id']}/{mode}")
            if any(record["split"] != tuning_split for record in history):
                raise RuntimeError(f"Mixed tuning splits for {entry['id']}/{mode}")
            if tuning_split in ("train", "test"):
                seen = {digest(record["params"]) for record in history}
                if any(digest(params) not in seen for params in anchors(entry, mode)):
                    raise RuntimeError(
                        f"Initial candidates are incomplete for {entry['id']}/{mode}; resume search"
                    )
            best = dict(best_trial(history))
            best["completed_trials"] = len(history)
            selected[entry["id"]][mode] = best
    counts = {record["completed_trials"] for item in selected.values() for record in item.values()}
    if len(counts) != 1:
        raise RuntimeError(
            "Cannot freeze an incomplete round; resume search to equalize trial counts"
        )
    selection = {
        "manifest_sha256": digest(manifest),
        "created_unix_s": time.time(),
        "policy": manifest["selection_policy"],
        "selected": selected,
    }
    if tuning_split == "train":
        selection.update(tuning_split="train", tuning_sequences=manifest["tuning_sequences"])
    if tuning_split == "test":
        selection.update(
            tuning_split="test",
            tuning_sequences=list(TEST_SEQUENCES),
            evaluation_is_independent=False,
        )
    atomic_json(path, selection)
    return selection


def write_report(manifest: dict, output: Path) -> dict:
    records, csv_rows = [], []
    test_tuning = manifest.get("tuning_split", "val") == "test"
    train_tuning = manifest.get("tuning_split", "val") == "train"
    search_key = "training" if train_tuning else "tuning" if test_tuning else "validation"
    hota_key = (
        "training_HOTA" if train_tuning else "tuning_HOTA" if test_tuning else "validation_HOTA"
    )
    selection = (
        read_json(output / "selection.json") if (output / "selection.json").exists() else None
    )
    for entry in manifest["entries"]:
        final_path = output / "final" / entry["id"] / "results.json"
        final = read_json(final_path) if final_path.exists() else {}
        tuning = {}
        for mode in modes(entry):
            history = read_history(output / "trials" / entry["id"] / f"{mode}.jsonl")
            if history:
                tuning[mode] = dict(best_trial(history), completed_trials=len(history))
        records.append({"entry": entry, search_key: tuning, "test": final})
        for case, result in final.items():
            row = {
                "id": entry["id"],
                "scope": entry["scope"],
                "case": case,
                "families": ";".join(sorted({a["family"] for a in entry["aliases"]})),
                "labels": ";".join(a["label"] for a in entry["aliases"]),
                "checkpoint": entry["checkpoint_relative"],
                "normalisation": entry["normalisation"],
                **result["params"],
                **result["metrics"],
            }
            mode = result["mode"]
            matching = [
                r
                for r in read_history(output / "trials" / entry["id"] / f"{mode}.jsonl")
                if r["params"] == result["params"]
            ]
            row[hota_key] = matching[0]["metrics"]["HOTA"] if matching else None
            row["completed_trials"] = tuning.get(mode, {}).get("completed_trials", 0)
            if train_tuning:
                row.update(tuning_split="train")
            if test_tuning:
                row.update(tuning_split="test", evaluation_is_independent=False)
            baseline = final.get(f"baseline_{mode}", {}).get("metrics", {})
            row["delta_HOTA_vs_fixed_baseline"] = (
                result["metrics"]["HOTA"] - baseline["HOTA"] if baseline else None
            )
            csv_rows.append(row)
    complete = bool(selection) and all(record["test"] for record in records)
    summary = {
        "complete": complete,
        "test_selection": (
            "training only"
            if train_tuning
            else "test diagnostic (same sequences)"
            if test_tuning
            else "validation only"
        ),
        "notes": manifest["notes"],
        "records": records,
    }
    if train_tuning:
        summary.update(tuning_split="train", tuning_sequences=manifest["tuning_sequences"])
    if test_tuning:
        summary.update(
            tuning_split="test",
            tuning_sequences=list(TEST_SEQUENCES),
            evaluation_is_independent=False,
        )
    atomic_json(output / "benchmark_summary.json", summary)
    if csv_rows:
        with (output / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    lines = [
        "# Fixed-checkpoint tracker tuning",
        "",
        f"Status: {'complete' if complete else 'partial; missing test results are not zero'}.",
        "",
        (
            f"Parameters selected on all {len(manifest['tuning_sequences'])} training sequences, "
            "excluding validation and test sequences. The search objective is TrackEval's "
            "detection-averaged aggregate of the training sequences. Final test metrics combine "
            "both test sequences; test evaluation starts only after selection is frozen."
            if train_tuning
            else "Diagnostic: parameters selected on interlaken_00_d and zurich_city_00_b. "
            "Selection and evaluation use the same sequences; this is not an independent test. "
            "The objective is TrackEval's detection-averaged aggregate of both sequences."
            if test_tuning
            else "Parameters selected on zurich_city_01_d only. Test metrics combine both test "
            "sequences with TrackEval's detection-averaged aggregation."
        ),
        "",
        "Baseline: score=0.90, high=0.60, low=0.10, buffer=2, match=0.50, "
        "appearance=0.25, proximity=0.50. This is a new fixed control, not an exact "
        "reproduction of every historical thesis row.",
        "",
        f"| Configuration | Mode | Trials | "
        f"{'Train' if train_tuning else 'Tuning' if test_tuning else 'Val'} HOTA | "
        "Test HOTA | AssA | DetA | IDF1 | "
        "MOTA | IDS | FP | FN | Delta HOTA vs control |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in csv_rows:
        if not row["case"].startswith("tuned_"):
            continue
        lines.append(
            f"| {row['id']} | {row['case'][6:]} | {row['completed_trials']} | "
            f"{row[hota_key]:.6f} | "
            + " | ".join(
                f"{row[key]:.6f}" if key not in ("IDS", "FP", "FN") else str(row[key])
                for key in METRICS
            )
            + f" | {row['delta_HOTA_vs_fixed_baseline']:+.6f} |"
        )
    lines += [
        "",
        "## Same-checkpoint embedding effect",
        "",
        "Both rows use the ReID-selected geometry and detection thresholds. Only "
        "embedding use changes; this separates the association effect from tuning.",
        "",
        "| Configuration | HOTA without embeddings | HOTA with embeddings | Delta |",
        "|---|---:|---:|---:|",
    ]
    for record in records:
        final = record["test"]
        if "tuned_reid" in final:
            off = final["reid_settings_without_embeddings"]["metrics"]["HOTA"]
            on = final["tuned_reid"]["metrics"]["HOTA"]
            lines.append(f"| {record['entry']['id']} | {off:.6f} | {on:.6f} | {on - off:+.6f} |")
    lines += ["", "## Configuration key", ""]
    for entry in manifest["entries"]:
        aliases = "; ".join(f"{a['family']}/{a['label']}" for a in entry["aliases"])
        lines.append(
            f"- `{entry['id']}`: {aliases}; `{entry['checkpoint_relative']}`; "
            f"normalisation `{entry['normalisation']}`."
        )
    lines += ["", "## Interpretation limits", "", *[f"- {note}" for note in manifest["notes"]]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_contribution_comparisons(records, output, manifest)
    return summary


def write_contribution_comparisons(
    records: list[dict], output: Path, manifest: dict | None = None
) -> None:
    rows = []
    for scope in ("all_classes", "car_only"):
        variants = {}
        for record in records:
            if record["entry"]["scope"] != scope:
                continue
            for alias in record["entry"]["aliases"]:
                if alias["family"] == "contribution":
                    variants[alias["label"]] = record
        baseline = variants.get("baseline", {}).get("test", {}).get("tuned_motion")
        if baseline is None:
            continue
        for label, record in variants.items():
            case = "tuned_reid" if record["entry"]["embedding_dim"] else "tuned_motion"
            if case not in record["test"]:
                continue
            metrics = record["test"][case]["metrics"]
            rows.append(
                {
                    "scope": scope,
                    "variant": label,
                    "case": case,
                    **metrics,
                    **{
                        f"delta_{key}_vs_baseline": metrics[key] - baseline["metrics"][key]
                        for key in METRICS
                    },
                }
            )
    comparison = {"records": rows}
    if manifest and manifest.get("tuning_split", "val") == "train":
        comparison.update(tuning_split="train", tuning_sequences=manifest["tuning_sequences"])
        for row in rows:
            row.update(tuning_split="train")
    if manifest and manifest.get("tuning_split", "val") == "test":
        comparison.update(
            tuning_split="test",
            tuning_sequences=list(TEST_SEQUENCES),
            evaluation_is_independent=False,
        )
        for row in rows:
            row.update(tuning_split="test", evaluation_is_independent=False)
    atomic_json(output / "contribution_comparisons.json", comparison)
    if rows:
        with (output / "contribution_comparisons.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/datasets/dsec_mot")
    parser.add_argument("--output-root")
    parser.add_argument(
        "--tuning-split",
        choices=("val", "train", "test"),
        default="val",
        help="Select on validation (default), all training sequences, or both test sequences.",
    )
    parser.add_argument(
        "--reference-selection",
        help="Frozen selection.json whose parameters are re-evaluated as initial candidates.",
    )
    parser.add_argument("--eros-cache-root", default="data/cache/dsec_mot_eros")
    parser.add_argument("--trackeval-root", default="external/TrackEval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budget-hours", type=float, default=168)
    parser.add_argument(
        "--final-reserve-hours",
        type=float,
        default=2,
        help="Minimum final reserve; measured export times may increase it.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--round-size", type=int, default=4)
    parser.add_argument(
        "--max-trials",
        type=int,
        default=100000,
        help="Per checkpoint/mode cap, normally limited by the time budget.",
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--min-free-gb", type=float, default=3)
    parser.add_argument(
        "--families", nargs="+", help="Optional subset; default includes all families."
    )
    parser.add_argument("--scopes", nargs="+", choices=("all_classes", "car_only"))
    parser.add_argument(
        "--phase", choices=("all", "prepare", "search", "finalize", "report"), default="all"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/discover the manifest without inference or evaluation.",
    )
    args = parser.parse_args(argv)
    default_outputs = {
        "val": DEFAULT_OUTPUT,
        "train": DEFAULT_TRAIN_OUTPUT,
        "test": DEFAULT_TEST_OUTPUT,
    }
    if args.output_root is None:
        args.output_root = default_outputs[args.tuning_split]
    if any(
        split != args.tuning_split
        and Path(args.output_root).resolve() == (REPO_ROOT / path).resolve()
        for split, path in default_outputs.items()
    ):
        parser.error(f"{args.tuning_split} tuning requires a separate --output-root")
    if min(args.workers, args.threads, args.round_size, args.max_trials) < 1:
        parser.error("workers, threads, round-size and max-trials must be positive")
    if args.budget_hours <= 0 or args.final_reserve_hours < 0 or args.min_free_gb < 0:
        parser.error("Invalid time or disk budget")
    return args


def run(args: dict) -> int:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(args["threads"])
    output = Path(args["output_root"])
    manifest_path = output / "manifest.json"
    if args["phase"] == "report":
        manifest = read_json(manifest_path)
        summary = write_report(manifest, output)
        print(f"Report: {output / 'report.md'} (complete={summary['complete']})")
        return 0
    args["runtime_identity"] = runtime_identity(REPO_ROOT, Path(args["trackeval_root"]))
    tuning_split = args.get("tuning_split", "val")
    args["dataset_identity"] = dataset_identity(Path(args["root"]), tuning_split)
    search_label = {"val": "Validation", "train": "Training", "test": "Test tuning"}[tuning_split]
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        entries = manifest["entries"]
        args["reference_identity"] = import_reference_params(entries, args)
        expected = build_manifest(entries, args)
        if expected != manifest:
            saved_files = manifest["protocol"]["runtime_identity"].get("code_files", {})
            current_files = args["runtime_identity"].get("code_files", {})
            changed_files = (
                sorted(
                    name
                    for name in saved_files.keys() | current_files.keys()
                    if saved_files.get(name) != current_files.get(name)
                )
                if saved_files
                else []
            )
            detail = (
                f" Changed code files: {', '.join(changed_files[:10])}." if changed_files else ""
            )
            raise RuntimeError(f"Manifest/code/protocol changed. Use a new --output-root.{detail}")
        for entry in entries:
            if file_identity(Path(entry["checkpoint"])) != entry["checkpoint_identity"]:
                raise RuntimeError(f"Checkpoint changed: {entry['checkpoint']}")
            if (
                file_identity(Path(entry["checkpoint"]).parent / "config.json")
                != entry["config_identity"]
            ):
                raise RuntimeError(f"Training config changed: {entry['checkpoint']}")
        if args["families"] or args["scopes"]:
            filtered = filter_entries(entries, args)
            if filtered != entries:
                raise RuntimeError("Cannot change the cohort on resume; use a new output root")
    else:
        entries = filter_entries(discover(REPO_ROOT), args)
        if not entries:
            raise RuntimeError("No selected checkpoints found for this cohort")
        args["reference_identity"] = import_reference_params(entries, args)
        manifest = build_manifest(entries, args)
        atomic_json(manifest_path, manifest)
    counts = {}
    for entry in entries:
        for family in {alias["family"] for alias in entry["aliases"]}:
            counts[family] = counts.get(family, 0) + 1
    total_modes = sum(len(modes(entry)) for entry in entries)
    required_trials = (
        max(len(anchors(entry, mode)) for entry in entries for mode in modes(entry))
        if tuning_split in ("train", "test")
        else 1
    )
    if args["max_trials"] < required_trials:
        raise ValueError(f"--max-trials must cover all {required_trials} initial candidates")
    print(f"Frozen cohort: {len(entries)} checkpoint protocols, {total_modes} searches.")
    print(json.dumps(counts, indent=2))
    print(f"Manifest: {manifest_path}")
    if tuning_split == "train":
        print(
            f"Training tuning: selection uses all {len(manifest['tuning_sequences'])} "
            "training sequences."
        )
    if tuning_split == "test":
        print("Diagnostic test tuning: selection and evaluation use the same two sequences.")
    if args["dry_run"]:
        print("Dry run complete: no inference, tracking, or test evaluation was run.")
        return 0
    (output / "STOP.request").unlink(missing_ok=True)
    state_path = output / "state.json"
    state = read_json(state_path) if state_path.exists() else {"spent_s": 0, "phase": "prepare"}
    started = time.monotonic()
    prior_spent = state["spent_s"]

    def save_state(phase: str) -> None:
        state.update(
            spent_s=prior_spent + time.monotonic() - started,
            phase=phase,
            budget_hours=args["budget_hours"],
        )
        atomic_json(state_path, state)

    def remaining() -> float:
        return args["budget_hours"] * 3600 - prior_spent - (time.monotonic() - started)

    try:
        selection_path = output / "selection.json"
        if not selection_path.exists() and args["phase"] != "finalize":
            for entry in entries:
                if remaining() <= 0:
                    raise RuntimeError(
                        "Time budget exhausted preparing caches. Increase --budget-hours."
                    )
                export_cache(entry, tuning_split, args)
                save_state("prepare")
            if args["phase"] == "prepare":
                return 0
            export_seconds = 0.0
            for entry in entries:
                if tuning_split == "test":
                    # Both test caches already exist and are reused for final evaluation.
                    continue
                metadata = [
                    read_json(output / "cache" / entry["id"] / tuning_split / f"{sequence}.json")
                    for sequence in split_sequences(tuning_split, args)
                ]
                # Use measured per-frame inference with a margin, not a promised runtime.
                test_frames = sum(
                    len(
                        (
                            Path(args["root"])
                            / "test"
                            / sequence
                            / f"{sequence}_image_timestamps.txt"
                        )
                        .read_text()
                        .splitlines()
                    )
                    for sequence in TEST_SEQUENCES
                )
                export_seconds += (
                    sum(item["elapsed_s"] for item in metadata)
                    * test_frames
                    / sum(item["frames"] for item in metadata)
                    * 1.25
                )
            state["estimated_test_export_s"] = export_seconds
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=args["workers"],
                mp_context=context,
                initializer=initialize_worker,
                initargs=(args["threads"],),
            ) as pool:
                while True:
                    histories = [
                        (
                            entry,
                            mode,
                            read_history(output / "trials" / entry["id"] / f"{mode}.jsonl"),
                        )
                        for entry in entries
                        for mode in modes(entry)
                    ]
                    counts = [len(history) for _, _, history in histories]
                    minimum, maximum = min(counts), max(counts)
                    if minimum >= args["max_trials"]:
                        break
                    # Finish interrupted rounds before advancing any better-progressed mode.
                    target = (
                        maximum
                        if minimum < maximum
                        else min(minimum + args["round_size"], args["max_trials"])
                    )
                    durations = [
                        record["elapsed_s"]
                        for _, _, history in histories
                        for record in history[-8:]
                    ]
                    mean_s = sum(durations) / len(durations) if durations else 0
                    final_s = export_seconds + mean_s * len(entries) * 5 * 1.5
                    reserve = max(args["final_reserve_hours"] * 3600, final_s)
                    if mean_s:
                        estimated_extra = max(0, remaining() - reserve) * args["workers"] / mean_s
                        state["estimated_trials_per_mode"] = min(
                            args["max_trials"], minimum + int(estimated_extra / total_modes)
                        )
                        state["estimated_final_reserve_s"] = reserve
                    round_s = (
                        mean_s * sum(max(0, target - n) for n in counts) / args["workers"] * 1.4
                    )
                    if (
                        minimum == maximum
                        and minimum >= required_trials
                        and remaining() < reserve + round_s
                    ):
                        print("Search budget reached; freezing the last complete balanced round.")
                        break
                    if remaining() <= 0:
                        raise RuntimeError(
                            "Time budget exhausted; increase --budget-hours and resume."
                        )
                    print(
                        f"{search_label} round -> {target} trials/mode; "
                        f"remaining {remaining() / 3600:.2f} h; "
                        f"estimated final reserve {reserve / 3600:.2f} h",
                        flush=True,
                    )
                    futures = {
                        pool.submit(search_batch, entry, mode, target, args): (entry["id"], mode)
                        for entry, mode, history in histories
                        if len(history) < target
                    }
                    try:
                        for future in as_completed(futures):
                            result = future.result()
                            print(
                                f"{result['entry_id']}/{result['mode']}: "
                                f"{result['trials']} trials, "
                                f"{tuning_split} HOTA={result['best']['metrics']['HOTA']:.6f}",
                                flush=True,
                            )
                            save_state("search")
                    except BaseException:
                        (output / "STOP.request").touch()
                        for future in futures:
                            future.cancel()
                        raise
                    write_report(manifest, output)
            save_state("search_complete")
            if args["phase"] == "search":
                return 0
        selection = freeze_selection(entries, args, manifest)
        selection_hash = file_hash(selection_path)
        save_state("finalize")
        for entry in entries:
            cases = final_cases(entry, selection["selected"][entry["id"]])
            final_root = output / "final" / entry["id"]
            result_path = final_root / "results.json"
            if result_path.exists():
                finished = read_json(result_path)
                if set(finished) != set(cases) or any(
                    r["selection_sha256"] != selection_hash for r in finished.values()
                ):
                    raise RuntimeError(f"Invalid final results: {result_path}")
                continue
            if remaining() <= 0:
                raise RuntimeError(
                    "Time budget exhausted during final evaluation; increase --budget-hours."
                )
            if tuning_split != "test":
                export_cache(entry, "test", args)
            finished = {}
            reuse = {}
            for case, config in cases.items():
                fingerprint = digest(config)
                case_path = final_root / case / "result.json"
                if case_path.exists():
                    record = read_json(case_path)
                    if record["selection_sha256"] != selection_hash:
                        raise RuntimeError(f"Final result uses a different selection: {case_path}")
                elif fingerprint in reuse:
                    record = dict(reuse[fingerprint], reused_case=True)
                    atomic_json(case_path, record)
                else:
                    print(f"Final test {entry['id']}/{case}", flush=True)
                    record = evaluate(
                        entry,
                        config["mode"],
                        config["params"],
                        "test",
                        args,
                        destination=case_path.parent,
                    )
                    record["selection_sha256"] = selection_hash
                    atomic_json(case_path, record)
                if tuning_split == "test" and case.startswith("tuned_"):
                    selected_metrics = selection["selected"][entry["id"]][config["mode"]]["metrics"]
                    if record["metrics"] != selected_metrics:
                        raise RuntimeError(f"Final metrics differ from the selected trial: {case}")
                finished[case] = record
                reuse[fingerprint] = record
                save_state("finalize")
            atomic_json(result_path, finished)
            # Keep metadata/checksums and final tracks; bound disk use to one test cache.
            from src.experiments.tracker_tuning_runtime import load_cache

            load_cache.cache_clear()
            for sequence in TEST_SEQUENCES:
                (output / "cache" / entry["id"] / "test" / f"{sequence}.npz").unlink()
            write_report(manifest, output)
        save_state("complete")
        write_report(manifest, output)
        print(f"Complete. Results: {output / 'report.md'}")
        return 0
    finally:
        save_state(state.get("phase", "interrupted"))


def filter_entries(entries: list[dict], args: dict) -> list[dict]:
    return [
        entry
        for entry in entries
        if (
            not args["families"]
            or any(alias["family"] in args["families"] for alias in entry["aliases"])
        )
        and (not args["scopes"] or entry["scope"] in args["scopes"])
    ]


def dataset_identity(root: Path, tuning_split: str = "val") -> dict:
    result = {}
    train_sequences = (
        split_sequences("train", {"root": str(root)}) if tuning_split == "train" else [VAL_SEQUENCE]
    )
    for split, sequences in (("train", train_sequences), ("test", TEST_SEQUENCES)):
        for sequence in sequences:
            for path in (
                root / split / sequence / f"{sequence}_image_timestamps.txt",
                root / "annotations" / split / f"{sequence}.txt",
                root / split / sequence / "events_left/events.h5",
            ):
                identity = file_identity(path)
                if path.suffix != ".h5":
                    identity["sha256"] = file_hash(path)
                result[str(path.relative_to(root))] = identity
    return result


def main(argv: list[str] | None = None) -> int:
    args = vars(parse_args(argv))
    os.chdir(REPO_ROOT)
    for key in ("root", "output_root", "eros_cache_root", "trackeval_root"):
        args[key] = str(Path(args[key]).resolve())
    output = Path(args["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another tracker search owns {output}") from exc
        check_disk(output, args["min_free_gb"])
        return run(args)


if __name__ == "__main__":
    sys.exit(main())
