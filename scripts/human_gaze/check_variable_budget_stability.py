#!/usr/bin/env python3
"""Fail a dependent launch when the short variable-budget pilot is unstable."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--reference-fixed-k16", type=float, required=True)
    parser.add_argument("--min-raw-mean-k", type=float, default=12.0)
    parser.add_argument("--max-raw-mean-k", type=float, default=20.0)
    parser.add_argument("--max-token-cost", type=float, default=0.0195)
    parser.add_argument("--max-forced-k16-drop", type=float, default=0.02)
    parser.add_argument("--max-single-boundary-rate", type=float, default=0.5)
    parser.add_argument("--max-total-boundary-rate", type=float, default=0.6)
    parser.add_argument("--min-calibrated-mean-k", type=float, default=15.5)
    parser.add_argument("--max-calibrated-mean-k", type=float, default=16.5)
    args = parser.parse_args()

    history = read_jsonl(args.history)
    if not history:
        raise SystemExit("stability gate failed: validation history is empty")
    endpoint = history[-1]
    with args.diagnostics.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    raw_mean_k = float(endpoint["mean_tokens_macro_source"])
    token_cost = float(endpoint["token_cost"])
    calibrated_mean_k = float(report["length"]["variable"]["macro_source_mean"])
    forced_k16 = float(
        report["coverage"]["forced_k16"]["16"]["macro_source_mean"]
    )
    min_rate = float(report["diagnostics"]["min_rate"])
    cap_rate = float(report["diagnostics"]["cap_rate"])
    values = (
        raw_mean_k,
        token_cost,
        calibrated_mean_k,
        forced_k16,
        min_rate,
        cap_rate,
    )
    if not all(math.isfinite(value) for value in values):
        raise SystemExit("stability gate failed: endpoint contains a non-finite value")

    failures = []
    if not args.min_raw_mean_k <= raw_mean_k <= args.max_raw_mean_k:
        failures.append(
            f"raw mean K {raw_mean_k:.4f} outside "
            f"[{args.min_raw_mean_k:.1f}, {args.max_raw_mean_k:.1f}]"
        )
    if token_cost > args.max_token_cost:
        failures.append(
            f"token price {token_cost:.6f} exceeds {args.max_token_cost:.6f}"
        )
    if forced_k16 < args.reference_fixed_k16 - args.max_forced_k16_drop:
        failures.append(
            f"forced-K16 {forced_k16:.4f} dropped more than "
            f"{args.max_forced_k16_drop:.3f} from {args.reference_fixed_k16:.4f}"
        )
    if not args.min_calibrated_mean_k <= calibrated_mean_k <= args.max_calibrated_mean_k:
        failures.append(
            f"calibrated mean K {calibrated_mean_k:.4f} outside "
            f"[{args.min_calibrated_mean_k:.1f}, {args.max_calibrated_mean_k:.1f}]"
        )
    if max(min_rate, cap_rate) > args.max_single_boundary_rate:
        failures.append(
            f"single-boundary rate {max(min_rate, cap_rate):.4f} exceeds "
            f"{args.max_single_boundary_rate:.4f}"
        )
    if min_rate + cap_rate > args.max_total_boundary_rate:
        failures.append(
            f"combined boundary rate {min_rate + cap_rate:.4f} exceeds "
            f"{args.max_total_boundary_rate:.4f}"
        )

    summary = {
        "calibrated_mean_k": calibrated_mean_k,
        "forced_k16_macro": forced_k16,
        "cap_rate": cap_rate,
        "min_rate": min_rate,
        "raw_mean_k": raw_mean_k,
        "reference_fixed_k16": args.reference_fixed_k16,
        "token_cost": token_cost,
    }
    print(json.dumps(summary, sort_keys=True))
    if failures:
        raise SystemExit("stability gate failed: " + "; ".join(failures))
    print("stability gate passed")


if __name__ == "__main__":
    main()
