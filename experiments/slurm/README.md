# SLURM

New cluster launches should go through:

```bash
python experiments/tools/submit_slurm.py <config.yaml> --run-id <run> --submit
```

The `legacy/` directory contains old hand-written sbatch wrappers. Keep them for
provenance, but prefer config-driven submission for new experiments.
