"""Portable CPU helpers for restart-safe HLVid measurements.

This module does not modify inference or infer missing historical measurements.
Raw decoder actions must be measured before resolution adaptation; retained
patches must be measured after it. Each processor call contributes separate
observations, including calls for the same source frame in different tiles.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize a portable JSON value deterministically, rejecting NaN/Inf."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_example_key(question_id: str | int, split: str) -> str:
    return fingerprint({"split": split, "question_id": question_id})


def stable_video_key(video_path: str | Path, split: str) -> str:
    """Use a manifest-relative path to retain identity across mount points."""
    return fingerprint({"split": split, "video_path": str(video_path).replace("\\", "/")})


def resume_compatibility_key(identity: Mapping[str, Any]) -> str:
    """Hash inference identity, including policy/checkpoint, data and protocol.

    Callers must exclude measurement timestamps and instrumentation-only code
    revisions. They must include all inputs that can affect selections/answers.
    """
    if not identity:
        raise ValueError("resume identity must not be empty")
    return fingerprint(dict(identity))


def legacy_decode_map(
    intended_indices: Sequence[int], successfully_decoded_indices: Sequence[int]
) -> dict[str, Any]:
    """Describe legacy compaction and last-frame padding without decoding pixels.

    Failed requested indices disappear before the remaining chronological list
    is padded. Therefore a failure can shift many output slots, not just replace
    the failed slot. A successful unique frame is reusable at repeated requests.
    """
    intended = list(intended_indices)
    successful_indices = list(successfully_decoded_indices)
    successful = set(successful_indices)
    if not intended:
        raise ValueError("intended frame sequence must not be empty")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in intended):
        raise ValueError("frame indices must be nonnegative integers")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in successful_indices):
        raise ValueError("successful frame indices must be nonnegative integers")
    if not successful.issubset(set(intended)):
        raise ValueError("successful indices contain an unrequested frame")
    effective = [index for index in intended if index in successful]
    successful_slot_count = len(effective)
    if effective:
        effective.extend([effective[-1]] * (len(intended) - len(effective)))
    return {
        "intended_indices": intended,
        "failed_unique_indices": sorted(set(intended) - successful),
        "effective_indices": effective,
        "successful_requested_slots": successful_slot_count,
        "tail_padding_count": len(intended) - successful_slot_count if effective else 0,
        "decode_usable": bool(effective),
        "substitutions": [
            {"slot": slot, "intended_index": requested, "effective_index": actual,
             "tail_padding": slot >= successful_slot_count}
            for slot, (requested, actual) in enumerate(zip(intended, effective))
            if requested != actual or slot >= successful_slot_count
        ],
    }


def counter_summary(
    values: Sequence[int | float], *, expected_observations: int | None = None,
    unit: str = "count", source: str = "observed"
) -> dict[str, Any]:
    numbers = list(values)
    if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) for n in numbers):
        raise ValueError("counter observations must be finite numbers")
    if expected_observations is not None:
        if isinstance(expected_observations, bool) or not isinstance(expected_observations, int) or expected_observations < 0:
            raise ValueError("expected observations must be a nonnegative integer")
        if expected_observations < len(numbers):
            raise ValueError("observed count exceeds expected observations")
    complete = expected_observations is None or expected_observations == len(numbers)
    return {
        "availability": "complete" if complete else "partial",
        "observed_count": len(numbers), "expected_observations": expected_observations,
        "observed_sum": sum(numbers), "mean": sum(numbers) / len(numbers) if numbers else None,
        "min": min(numbers) if numbers else None, "max": max(numbers) if numbers else None,
        "unit": unit, "source": source,
    }


def unavailable_counter(
    reason: str, *, expected_observations: int | None = None, unit: str = "count"
) -> dict[str, Any]:
    if not reason:
        raise ValueError("unavailable counters require a reason")
    if expected_observations is not None and (isinstance(expected_observations, bool) or not isinstance(expected_observations, int) or expected_observations < 0):
        raise ValueError("expected observations must be a nonnegative integer")
    return {
        "availability": "unavailable", "observed_count": 0,
        "expected_observations": expected_observations, "observed_sum": None,
        "mean": None, "min": None, "max": None, "unit": unit,
        "source": "unavailable", "reason": reason,
    }


def merge_counter_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge observations without presenting a resumed tail as a full run."""
    if not summaries:
        return unavailable_counter("No counter records supplied")
    units = {entry["unit"] for entry in summaries}
    if len(units) != 1:
        raise ValueError("cannot merge counters with different units")
    expected_values = [entry.get("expected_observations") for entry in summaries]
    expected = None if None in expected_values else sum(expected_values)
    available = [entry for entry in summaries if entry["availability"] != "unavailable"]
    if not available:
        return unavailable_counter("All constituent counters unavailable", expected_observations=expected, unit=next(iter(units)))
    count = sum(entry["observed_count"] for entry in available)
    total = sum(entry["observed_sum"] for entry in available)
    extrema = [entry for entry in available if entry["observed_count"]]
    return {
        "availability": "complete" if all(entry["availability"] == "complete" for entry in summaries) else "partial",
        "observed_count": count, "expected_observations": expected, "observed_sum": total,
        "mean": total / count if count else None,
        "min": min(entry["min"] for entry in extrema) if extrema else None,
        "max": max(entry["max"] for entry in extrema) if extrema else None,
        "unit": next(iter(units)), "source": "merged",
        "unavailable_parts": len(summaries) - len(available),
    }


def _validate_record(record: Mapping[str, Any], compatibility_key: str) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("evidence record must be a JSON object")
    for name in ("example_key", "video_key", "attempt_id", "state", "compatibility_key"):
        if not isinstance(record.get(name), str) or not record[name]:
            raise ValueError(f"record requires nonempty {name}")
    if record.get("schema_version") != 1:
        raise ValueError("unsupported evidence schema_version")
    if record["compatibility_key"] != compatibility_key:
        raise ValueError("incompatible evidence resume identity")
    if record["state"] not in {"attempt_started", "completed_answer", "failed"}:
        raise ValueError("unknown evidence state")
    if record["state"] == "completed_answer" and not isinstance(record.get("answer"), dict):
        raise ValueError("completed_answer must embed its answer payload")
    canonical_json(record)


def load_resume_state(path: str | Path, expected_compatibility_key: str) -> dict[str, Any]:
    """Read immutable per-example records, or their derived JSONL export.

    Only completed_answer records authorize skipping QA. Started/failed attempts
    remain evidence of work performed and never count as completed answers.
    Duplicate completions are rejected even when their payloads are identical.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []
    if path.is_dir():
        for record_path in sorted(path.glob("*.json")):
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
    elif path.exists():
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Malformed JSONL evidence at line {line_number}") from error
    completed, attempts, events, terminal_attempts = {}, {}, set(), set()
    for record in records:
        _validate_record(record, expected_compatibility_key)
        event = (record["example_key"], record["attempt_id"], record["state"])
        if event in events:
            raise ValueError("duplicate evidence event")
        events.add(event)
        if record["state"] in {"completed_answer", "failed"}:
            attempt_key = (record["example_key"], record["attempt_id"])
            if attempt_key in terminal_attempts:
                raise ValueError("attempt has conflicting terminal states")
            terminal_attempts.add(attempt_key)
        attempts.setdefault(record["example_key"], []).append(record)
        if record["state"] == "completed_answer":
            if record["example_key"] in completed:
                raise ValueError("duplicate completed answer")
            completed[record["example_key"]] = record
    return {"completed": completed, "attempts": attempts, "records": records}


def write_evidence_record(directory: str | Path, record: Mapping[str, Any]) -> Path:
    """Atomically publish an immutable record; one run/arm owns one directory.

    A completed record embeds answer and measurements in the same durable write.
    Derive the QA JSONL from these records, or reconcile a separate existing QA
    log by identity; there is no atomic transaction across two independent files.
    """
    _validate_record(record, record.get("compatibility_key", ""))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = load_resume_state(directory, record["compatibility_key"])
    if record["example_key"] in state["completed"]:
        raise ValueError("example already has a completed answer")
    previous = state["attempts"].get(record["example_key"], [])
    if any(row["attempt_id"] == record["attempt_id"] and row["state"] in {"failed", "completed_answer"} for row in previous):
        raise ValueError("attempt already has a terminal state; use a new attempt_id")
    name = fingerprint({key: record[key] for key in ("example_key", "attempt_id", "state")}) + ".json"
    target = directory / name
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(dict(record)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Linking publishes the fully written inode atomically without replacing
        # an existing event. Both paths are in the same filesystem.
        os.link(temporary, target)
        if os.name == "posix":
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def export_evidence_jsonl(
    directory: str | Path, output_path: str | Path, expected_compatibility_key: str,
    *, completed_only: bool = False,
) -> Path:
    """Rebuild a derived export from the authoritative immutable records."""
    state = load_resume_state(directory, expected_compatibility_key)
    records = list(state["completed"].values()) if completed_only else state["records"]
    records.sort(key=lambda row: (row["example_key"], row["attempt_id"], row["state"]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-export-", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output_path


def _tolist(value: Any) -> Any:
    # Tensor conversion is deliberately optional: the module imports no torch.
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else value


def summarize_gaze_slots(
    num_gazing_each_frame: Any,
    if_padded_gazing: Any,
    *,
    gazing_pos: Any | None = None,
    actions_per_frame: int | None = None,
    fine_action_offset: int = 0,
) -> dict[str, Any]:
    """Count valid entries per batch/frame, not the shared padded maxima.

    ``num_gazing_each_frame`` gives T shared segment lengths. The padding mask
    has shape B x sum(lengths). When positions are supplied, subtract each
    frame's vocabulary offset and count only IDs in
    [fine_action_offset, actions_per_frame); this excludes EOS even when the
    first EOS is marked valid by the generation contract. Pass the appropriate
    vocabulary size before/after adaptation, never a decoder K in its place.
    """
    lengths = _tolist(num_gazing_each_frame)
    padding = _tolist(if_padded_gazing)
    positions = _tolist(gazing_pos) if gazing_pos is not None else None
    if not isinstance(lengths, (list, tuple)) or not lengths:
        raise ValueError("num_gazing_each_frame must be a nonempty vector")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in lengths):
        raise ValueError("frame slot counts must be nonnegative integers")
    if not isinstance(padding, (list, tuple)) or not padding:
        raise ValueError("if_padded_gazing must have shape B x sum(frame slots)")
    if positions is not None:
        if actions_per_frame is None or not 0 <= fine_action_offset < actions_per_frame:
            raise ValueError("positions require a valid action vocabulary")
        if len(positions) != len(padding):
            raise ValueError("position and padding batch sizes differ")
    total_slots = sum(lengths)
    valid_counts: list[list[int]] = []
    spatial_counts: list[list[int]] = []
    nonspatial_counts: list[list[int]] = []
    for batch_index, row in enumerate(padding):
        if not isinstance(row, (list, tuple)) or len(row) != total_slots:
            raise ValueError("padding length differs from sum(frame slots)")
        if any(not isinstance(flag, bool) for flag in row):
            raise ValueError("padding entries must be booleans")
        pos_row = positions[batch_index] if positions is not None else None
        if pos_row is not None and len(pos_row) != total_slots:
            raise ValueError("position length differs from sum(frame slots)")
        valid_row, spatial_row, nonspatial_row = [], [], []
        offset = 0
        for frame_index, length in enumerate(lengths):
            indices = [i for i in range(offset, offset + length) if not row[i]]
            valid_row.append(len(indices))
            if pos_row is not None:
                local_positions = [pos_row[i] - frame_index * actions_per_frame for i in indices]
                if any(isinstance(p, bool) or not isinstance(p, int) for p in local_positions):
                    raise ValueError("positions must be integers")
                spatial = sum(fine_action_offset <= p < actions_per_frame for p in local_positions)
                spatial_row.append(spatial)
                nonspatial_row.append(len(indices) - spatial)
            offset += length
        valid_counts.append(valid_row)
        if positions is not None:
            spatial_counts.append(spatial_row)
            nonspatial_counts.append(nonspatial_row)
    result = {
        "batch_size": len(padding),
        "frames_per_sample": len(lengths),
        "frame_observations": len(padding) * len(lengths),
        "padded_slots_per_frame": list(lengths),
        "padded_slot_sum": len(padding) * total_slots,
        "valid_counts_per_sample_frame": valid_counts,
        "valid_entry_sum": sum(map(sum, valid_counts)),
    }
    if positions is not None:
        result.update(
            spatial_counts_per_sample_frame=spatial_counts,
            spatial_entry_sum=sum(map(sum, spatial_counts)),
            excluded_valid_counts_per_sample_frame=nonspatial_counts,
        )
    return result
