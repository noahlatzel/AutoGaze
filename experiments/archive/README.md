# Experiment Archive

This directory keeps legacy launch wrappers for compatibility and provenance.
They may be adjusted when repository paths move, so they are not byte-for-byte
historical snapshots. They are not the preferred way to launch new work.

Use current experiment configs and tools instead:

```bash
python experiments/tools/render_command.py <config.yaml> --run-id <run>
python experiments/tools/submit_slurm.py <config.yaml> --run-id <run> --submit
```
