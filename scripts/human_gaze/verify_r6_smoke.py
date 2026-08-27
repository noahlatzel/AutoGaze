# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the real-data R6 two-update gradient and finite-state smoke."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.training_metrics.read_text().splitlines()]
    if len(rows) != 2 or [row["train_step"] for row in rows] != [0, 1]:
        raise AssertionError("R6 smoke must contain exactly updates 0 and 1")
    checks = {
        "first_update_gate_gradient_nonzero": rows[0]["recurrent_state_gate_grad_norm"] > 0,
        "second_update_gate_nonzero": abs(rows[1]["recurrent_state_logit_gate"]) > 0,
        "second_update_cell_gradient_nonzero": rows[1]["recurrent_state_cell_grad_norm"] > 0,
        "second_update_readout_gradient_nonzero": rows[1]["recurrent_state_readout_grad_norm"] > 0,
        "all_state_checks_finite": all(row["recurrent_state_all_finite"] == 1 for row in rows),
        "state_rms_bounded": all(row["recurrent_state_rms_max"] <= 1.000001 for row in rows),
    }
    if not all(checks.values()):
        raise AssertionError(f"R6 smoke failed: {checks}")
    report = {
        "schema_version": 1,
        "status": "passed",
        "updates": 2,
        "checks": checks,
        "rows": rows,
        "protected_test_opened": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
