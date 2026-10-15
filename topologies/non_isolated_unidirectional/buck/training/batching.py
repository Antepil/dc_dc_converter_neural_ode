"""Resolve and persist per-stage training batches without changing config hashes."""
import argparse


def parse_stage_batch(value):
    try:
        name, size = value.split("=", 1)
        size = int(size)
        if not name or size < 1:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected STAGE=POSITIVE_INTEGER") from exc
    return name, size


def resolve_training_batch_sizes(stages, checkpoint=None, global_batch=None, stage_batches=()):
    checkpoint = checkpoint or {}
    reports = checkpoint.get("stage_report", {})
    saved = checkpoint.get("training_batch_sizes", {})
    batches = {
        stage["name"]: saved.get(stage["name"], reports.get(stage["name"], {}).get(
            "training_batch_size", stage["batch_size"]))
        for stage in stages if stage["window_cycles"]
    }
    if global_batch is not None:
        if global_batch < 1:
            raise ValueError("Training batch size must be positive")
        batches = dict.fromkeys(batches, global_batch)
    seen = set()
    for name, size in stage_batches:
        if name not in batches:
            raise ValueError(f"Unknown rollout stage: {name}")
        if name in seen:
            raise ValueError(f"Duplicate stage batch override: {name}")
        if size < 1:
            raise ValueError("Training batch size must be positive")
        seen.add(name)
        batches[name] = size
    return batches
