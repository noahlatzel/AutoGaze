# External Frame Selectors

This directory is for upstream frame-selection projects used as baselines.
Keep cloned repositories here instead of vendoring their code into `autogaze/`.

The `InfoShot` checkout at commit
`ab20b22fcb48fe865af72b98d3688f24b15052ba` contains only a placeholder
README. AutoGaze therefore uses a clearly named paper reimplementation in
`scripts/frame_selectors/generate_infoshot_indices.py`.

Expected output interface for all adapters:

```json
{"question_id": "0", "video_path": "clip.mp4", "method": "aks", "num_frames": 128, "selected_source_indices": [0, 10, 20]}
```

Adapters should write JSONL files under:

```text
outputs/frame_indices/<method>_128.jsonl
```

The HLVid/NVILA evaluator consumes them with:

```bash
python scripts/runners/evaluate_hlvid_nvila.py \
  --frame-source selected_indices \
  --selected-indices outputs/frame_indices/<method>_128.jsonl \
  --num-video-frames 128
```
