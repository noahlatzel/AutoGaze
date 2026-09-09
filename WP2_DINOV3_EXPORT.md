# Frozen AutoGaze to DINOv3 panel export

This exports the approved validation panel and seven existing AutoGaze policies.
It does not train, load NVILA, evaluate saliency outcomes, or choose clips using
selector output. The older manuscript is irrelevant to this protocol.

## Immutable input contract

The checked-in run configuration is
`experiments/human_gaze/configs/wp2_dinov3_export.json`. It contains explicit Linux
paths and full SHA-256 values for the population and each checkpoint's
`model.safetensors`, `config.json` and `preprocessor_config.json`. All six human
policies are the existing 20k K16 endpoints: base seeds 440826 through 440831,
with continuation seeds 540826 through 540831. The pretrained policy uses the
verified `nvidia--AutoGaze` directory. Missing hashes, missing files, changed
bytes, missing seeds and incomplete model loads block inference. If a checkpoint
contains `generation_config.json`, add its verified hash to that policy's `files`
map as well. That file can be loaded implicitly by Transformers.

Panel selection is fixed before decoding any selected RGB:

- Population SHA-256:
  `0ef9fa17881517ea69d9edbaf57f8571c5c58d1da517370746fbe264fe33ff10`.
- Use only `val`, salt `wp2-dinov3-bridge-v1`.
- Within each source, rank clips by
  `sha256((salt + "\0" + clip_id).encode("utf-8"))`, then clip ID as tie-breaker.
- Take the first two distinct videos per source, one clip per video. There is no
  replacement from the same video or from another split if a file is missing.
- Canonically sort selected records by source, video ID, clip ID and frames.
- Twelve unique videos from six sources, 16 original frame numbers each:
  192 frames. Coutrot_db2 and ETMD_av have only two validation videos each; the
  superseded 24-clip proposal is not used.

The existing STAViS `load_aligned_clip` decodes RGB only, converts to RGB, and
resizes the complete field of view to 224 by 224 with PIL bilinear interpolation.
No heatmap is loaded. `panel.json` is written before this decoding;
`rgb_manifest.json` follows only after all twelve C-contiguous uint8
`[16,3,224,224]` NPY files and hashes are complete. Its `panel_payload_sha256`
covers clip identities and RGB metadata, before any policy is loaded.

## Linux execution

Use the fixed owner's admitted sequential A2000 lane, from a clean detached
checkout of this exporter commit, using the existing AutoGaze environment. Do
not overlap the decode audit/curation process. The script forces imports from
this checkout so an editable installation cannot silently select another copy.
The dataset-root path comes from the existing STAViS evaluation configuration;
the `freeze` stage verifies the required files by decoding the selected RGB.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --no-sync python scripts/human_gaze/export_dinov3_panel.py validate \
  --config experiments/human_gaze/configs/wp2_dinov3_export.json

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --no-sync python scripts/human_gaze/export_dinov3_panel.py run \
  --config experiments/human_gaze/configs/wp2_dinov3_export.json \
  --output outputs/human_gaze/wp2_dinov3_bridge_v1 \
  --device cuda:0 --batch-size 1 --cpu-threads 2 --num-workers 0
```

`run` validates every checkpoint before freezing RGB, then exports every policy.
For independent CPU preparation, use `freeze` with the same config/output, then
`export` against that existing output directory. Freeze performs no checkpoint
inference. Output directories for `freeze`/`run` must be new. A failed run
preserves its partial artifacts and blocks accidental overwrite; it never
substitutes a different clip or silently resumes partial policy inference.

The default batch is one complete 16-frame clip. The interface admits batches
1 through 4, zero workers and at most two CPU threads. All policies use the same
chosen batch size, which is recorded. Allocator reservation is capped at 9 GiB
to leave context/library headroom under the admitted 10 GiB GPU ceiling; the
Linux owner still monitors total process VRAM and host RSS (target at most
8 GiB). A batch does not contain more than 64 frames. The RGB cache is about
27.6 MiB and selection totals 1,344 frame-policy inputs, with only one model
resident at a time. These are workload counts, not measured runtime evidence.

## Policy input and output

Every selector consumes normalized values derived directly from the cached
uint8 bytes. Its verified `AutoGazeImageProcessor` retains its own rescaling,
offset and normalization. Resize and center crop are explicitly disabled for
this call because the shared full-field geometry is already fixed. All seven
processors must have identical normalization settings. This is not DINO
normalization; the downstream benchmark separately applies DINO normalization
once to the same RGB cache.

Models are loaded locally with safetensors and float32 parameters, set to eval,
and called under inference mode. The existing model chooses greedy generation
in eval mode. The exporter forces 16 fine actions, IDs 69 through 264, no EOS,
and no cross-clip cache. Parameter/input dtype and model attention mode are
recorded separately: existing `flash_attention_2` generation internally uses
BF16 autocast. No attention backend is changed by the exporter.

Native output must declare scales `[32,64,112,224]`, temporal stride one and
265 actions/frame. Subtract `frame_index * 265`, then 69, to obtain the ordered
14 by 14 fine-cell indices. The existing conversion function checks range and
uniqueness. Additional checks require exactly 16 actions/frame, no padding/EOS,
no coarse selection, and agreement with the model's native binary fine mask.
Rows are the original 16 frames; within each row, greedy action order is retained.

Per-batch progress is flushed as JSONL. A policy JSON is written only after its
12 clips succeed. The final `manifest.json` is assembled only after all seven
policies validate; its schema is the schema-version-1 interface accepted by
video-tokenization's `scripts/benchmark_gaze_dinov3.py` at commit
`9b5ac79cf6f49fb90905602f3b9bbd2cded011e1`. Each clip contains source-qualified
`video_id`, original ID, clip ID, frames, timestamps, relative RGB path/hash and
`policies: {name: [16 rows of 16 unique cells]}`.

**Publication requires `complete.json` and its matching final-manifest SHA-256.**
Do not launch the benchmark based on the presence of a partial JSON file alone.
`export_run.json` records the clean exporter commit, packages, hardware, hashes,
normalization and generation arguments. Batch wall times exclude model loading,
RGB preprocessing and transfer; the first batch is cold. They are single-run
operational measurements, not a repeated latency benchmark. CUDA allocated and
reserved peaks also exclude driver/context memory. Preserve the original final
manifest and completion hash with the DINO results.

## CPU evidence and remaining checks

```bash
uv run --no-sync python -m pytest tests/test_dinov3_panel.py tests/test_dinov3_export.py -q
```

The tests cover outcome-independent hash selection, unique-video/source
coverage, frozen identity-before-decode ordering, missing RGB without replacement,
byte hashes, full-field geometry, the real processor's per-call overrides,
normalization batch independence, the real native mask constructor, global
frame offsets, retained greedy order, no EOS/duplicates, and final schema/identity
alignment. Synthetic masks and RGB are test fixtures only.

Local CPU validation: **36 tests passed in 8.43 seconds**. A separate schema
round trip passed through the actual DINO benchmark loader using twelve
synthetic RGB clips and all seven policy names; no neural model was loaded.

Real Linux checkpoint loading and all 1,344 selector frame inputs remain to be
run. DINO's actual-weight dense/all-true/r0 gate and the encoder measurements
are a separate next step; passing these CPU checks supplies no GPU evidence.
