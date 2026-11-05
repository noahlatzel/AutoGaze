#!/usr/bin/env bash
set -euo pipefail
exec uv run --no-project --python /home/stud/latn/miniconda3/envs/autogaze/bin/python python "$@"
