# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render or submit proxy-sampling SLURM jobs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    tool = Path(__file__).resolve().parents[2] / "tools" / "submit_slurm.py"
    raise SystemExit(subprocess.call([sys.executable, str(tool), *sys.argv[1:]]))


if __name__ == "__main__":
    main()
