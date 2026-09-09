# Reproduce the fixed-budget publication figures

Run from the AutoGaze repository root. The renderer reads only preserved compact
R2e tables and metadata. It validates the compact addendum against the original
tables and writes publication assets inside this editable WP5 package. It does
not read videos, prediction streams, model weights, or protected gaze-test data;
does not invoke the heavy result aggregator; and does not overwrite the original
completed experimental bundle.

## Optional plotting environment

The reviewed rendering used Python 3.12.14, Matplotlib 3.11.1, and NumPy 2.5.2.
The renderer requires Matplotlib, NumPy, and the Python standard library. Its
dependencies are recorded separately in `requirements-figures.txt`; no model
evaluation or training environment is changed by this package.

For an isolated reproduction environment:

```text
python3.12 -m venv .venv-wp5-figures
.venv-wp5-figures/bin/python -m pip install -r results/wp5_three_strand_evidence/requirements-figures.txt
.venv-wp5-figures/bin/python results/wp5_three_strand_evidence/scripts/render_hlvid_fixed.py --input-dir experiments/human_gaze/results/r2e_hlvid_nvila_fixed_budget_all_seeds --output-dir results/wp5_three_strand_evidence/figures/hlvid_fixed --cross-check-dir results/wp5_three_strand_evidence/addenda/r2e_fixed_budget
```

On Windows, the environment interpreter is
`.venv-wp5-figures/Scripts/python.exe`. The generic `python ...` commands recorded
in `figures.csv` assume this plotting environment is active. Rendering needs no
GPU or network access once the plotting dependencies are available.

The `--input-dir` and `--output-dir` options support other source and staging
locations. The renderer rejects an output directory inside its immutable input
bundle. After WP5 publication, regenerate into a fresh destination and give any
changed evidence bundle a new identity rather than editing published assets.

## Assets and validation

Both figure sets are written as PDF, SVG, and 300-dpi PNG from the same figure
objects. PDF embeds TrueType fonts; SVG retains editable text. PDF/SVG timestamps
are omitted and SVG IDs use a fixed salt. The reviewed two-panel layouts are
11.8 inches wide, with explicit typography (10-point body/axis text, 11-point panel
titles, 14-point figure headings, and 9-point notes), DejaVu Sans, and redundant
seed markers. They share the package's blue, vermillion, green, and neutral
palette. These declared layout sizes preserve the reviewed figure typography
rather than silently inheriting a different physical-size default.

- `hlvid_fixed_common_seeds`: primary question-micro accuracy for exactly the
  three common base seeds, plus the published paired video-cluster intervals.
- `hlvid_fixed_aggregation_sensitivity`: secondary aggregation sensitivity on
  the same nine checkpoints, showing point estimates only.
- `figure_captions.md`: complete captions and claim boundaries.
- Per-figure JSON: input hashes, renderer hash, versions, exact comparison values,
  seeds, metric identity, and output file hashes.
- `data_checks.json`: exact common-seed membership; correct-count/micro agreement;
  published mean/difference checks; and addendum table-hash checks.

Input hashes identify the actual bytes read. The additional
`source_text_lf_sha256` map normalizes CRLF to LF for comparison with Linux or Git
text blobs; a Windows checkout can have different line endings without changed
scientific content. The renderer and newly generated text assets are pinned to
LF in the package's scoped `.gitattributes`. Original source files are not
renormalized or modified.

No bootstrap was recomputed. The source's video-resampling intervals condition
on the observed three-seed set and do not resample training seeds. All intervals
include zero; the figure makes no reliable ordering or equivalence claim. The
additional K16 endpoints and pending pretrained/Center16 controls are excluded.

Both PNGs were visually reviewed for legibility, clipping, overlapping labels,
and visible endpoint markers. No immutable source table or experimental artifact
was changed during integration.
