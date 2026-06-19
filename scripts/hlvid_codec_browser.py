# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve a small HLVid codec/keyframe inspection website."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import statistics
import subprocess
import time
import urllib.parse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq


DEFAULT_DATASET_ROOT = "/storage/slurm/latn/data/AutoGaze/HLVid"
DEFAULT_CACHE = "outputs/hlvid/codec_browser/keyframe_index.json"
DEFAULT_THUMB_DIR = "outputs/hlvid/codec_browser/keyframes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--thumb-dir", default=DEFAULT_THUMB_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the keyframe cache before serving.")
    parser.add_argument("--build-only", action="store_true", help="Build the keyframe cache and exit.")
    parser.add_argument("--limit", type=int, default=None, help="Only index the first N videos for a smoke run.")
    parser.add_argument("--jpeg-width", type=int, default=360, help="Rendered keyframe thumbnail width.")
    parser.add_argument("--jpeg-quality", type=int, default=82)
    return parser.parse_args()


def run_json(command: List[str]) -> Dict[str, Any]:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def fps_from_rate(rate: str) -> Optional[float]:
    if not rate or rate == "0/0":
        return None
    if "/" not in rate:
        return float(rate)
    numerator, denominator = rate.split("/", 1)
    denominator_float = float(denominator)
    if denominator_float == 0:
        return None
    return float(numerator) / denominator_float


def probe_stream(video_path: Path) -> Dict[str, Any]:
    data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,avg_frame_rate,duration",
            "-of",
            "json",
            str(video_path),
        ]
    )
    stream = data["streams"][0]
    fps = fps_from_rate(stream.get("avg_frame_rate", ""))
    duration = float(stream["duration"]) if stream.get("duration") else None
    frame_count = int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None
    return {
        "codec": stream.get("codec_name", ""),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
        "duration_sec": duration,
        "frame_count": frame_count,
    }


def probe_keyframes(video_path: Path, fps: Optional[float]) -> List[Dict[str, Any]]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-skip_frame",
            "nokey",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts_time,pict_type",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    keyframes = []
    for ordinal, line in enumerate(result.stdout.splitlines()):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        try:
            time_sec = float(fields[0])
        except (IndexError, ValueError):
            continue
        keyframes.append(
            {
                "ordinal": ordinal,
                "time_sec": time_sec,
                "frame_index": int(round(time_sec * fps)) if fps else None,
                "pict_type": fields[1] if len(fields) > 1 else "",
            }
        )
    return keyframes


def probe_gops(video_path: Path) -> List[Dict[str, Any]]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,size,flags",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gops: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        try:
            time_sec = float(fields[0])
            packet_size = int(fields[1])
        except (IndexError, ValueError):
            continue
        flags = fields[2] if len(fields) > 2 else ""
        starts_gop = "K" in flags or current is None
        if starts_gop:
            if current is not None:
                current["end_time_sec"] = time_sec
                current["duration_sec"] = round(time_sec - current["start_time_sec"], 6)
                current["avg_packet_size_bytes"] = current["total_packet_size_bytes"] / current["packet_count"]
                gops.append(current)
            current = {
                "ordinal": len(gops),
                "start_time_sec": time_sec,
                "end_time_sec": None,
                "duration_sec": None,
                "packet_count": 0,
                "total_packet_size_bytes": 0,
                "avg_packet_size_bytes": None,
            }
        if current is None:
            continue
        current["packet_count"] += 1
        current["total_packet_size_bytes"] += packet_size

    if current is not None:
        current["avg_packet_size_bytes"] = current["total_packet_size_bytes"] / current["packet_count"]
        gops.append(current)
    return gops


def load_hlvid_metadata(dataset_root: Path) -> Dict[str, Any]:
    parquet_path = dataset_root / "data" / "test-00000-of-00001.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing metadata parquet: {parquet_path}")

    rows = pq.read_table(parquet_path).to_pylist()
    videos: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        video_name = Path(row["video_path"]).name
        entry = videos.setdefault(
            video_name,
            {
                "question_count": 0,
                "categories": set(),
                "question_ids": [],
            },
        )
        entry["question_count"] += 1
        entry["question_ids"].append(row.get("question_id"))
        if row.get("category"):
            entry["categories"].add(row["category"])

    return {
        "row_count": len(rows),
        "unique_video_count": len(videos),
        "videos": {
            name: {
                "question_count": entry["question_count"],
                "categories": sorted(entry["categories"]),
                "question_ids": entry["question_ids"],
            }
            for name, entry in videos.items()
        },
    }


def intervals(values: List[float]) -> List[float]:
    return [round(values[index] - values[index - 1], 6) for index in range(1, len(values))]


def summarize_video(video: Dict[str, Any]) -> None:
    keyframes = video["keyframes"]
    times = [frame["time_sec"] for frame in keyframes]
    gaps = intervals(times)
    frame_count = video.get("frame_count")
    keyframe_count = len(keyframes)
    video["keyframe_count"] = keyframe_count
    video["keyframe_ratio"] = keyframe_count / frame_count if frame_count else None
    video["mean_keyframe_interval_sec"] = statistics.fmean(gaps) if gaps else None
    video["median_keyframe_interval_sec"] = statistics.median(gaps) if gaps else None
    video["min_keyframe_interval_sec"] = min(gaps) if gaps else None
    video["max_keyframe_interval_sec"] = max(gaps) if gaps else None
    gop_sizes = [gop["avg_packet_size_bytes"] for gop in video.get("gops", []) if gop.get("avg_packet_size_bytes") is not None]
    video["gop_count"] = len(video.get("gops", []))
    video["mean_gop_avg_packet_size_bytes"] = statistics.fmean(gop_sizes) if gop_sizes else None
    video["median_gop_avg_packet_size_bytes"] = statistics.median(gop_sizes) if gop_sizes else None
    video["min_gop_avg_packet_size_bytes"] = min(gop_sizes) if gop_sizes else None
    video["max_gop_avg_packet_size_bytes"] = max(gop_sizes) if gop_sizes else None


def build_index(dataset_root: Path, cache_path: Path, limit: Optional[int] = None) -> Dict[str, Any]:
    videos_dir = dataset_root / "videos"
    video_paths = sorted(videos_dir.glob("*.mp4"))
    metadata = load_hlvid_metadata(dataset_root)
    if limit is not None:
        video_paths = video_paths[:limit]
    if not video_paths:
        raise FileNotFoundError(f"No MP4 videos found under {videos_dir}")

    indexed = []
    started = time.perf_counter()
    for index, video_path in enumerate(video_paths, 1):
        print(f"[{index}/{len(video_paths)}] indexing {video_path.name}", flush=True)
        stream = probe_stream(video_path)
        keyframes = probe_keyframes(video_path, stream["fps"])
        gops = probe_gops(video_path)
        for keyframe, gop in zip(keyframes, gops):
            keyframe["gop_avg_packet_size_bytes"] = gop["avg_packet_size_bytes"]
            keyframe["gop_packet_count"] = gop["packet_count"]
        video = {
            "name": video_path.name,
            "path": str(video_path),
            "size_bytes": video_path.stat().st_size,
            "hlvid_questions": metadata["videos"].get(video_path.name, {}),
            **stream,
            "keyframes": keyframes,
            "gops": gops,
        }
        summarize_video(video)
        indexed.append(video)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "videos_dir": str(videos_dir),
        "metadata": {
            "row_count": metadata["row_count"],
            "unique_video_count": metadata["unique_video_count"],
            "missing_extracted_videos": sorted(set(metadata["videos"]) - {path.name for path in video_paths}),
            "videos_without_metadata": sorted({path.name for path in video_paths} - set(metadata["videos"])),
        },
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "videos": indexed,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_or_build_index(dataset_root: Path, cache_path: Path, rebuild: bool, limit: Optional[int]) -> Dict[str, Any]:
    if rebuild or not cache_path.exists():
        return build_index(dataset_root, cache_path, limit)
    with cache_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def histogram(values: List[float], bin_size: float = 0.25, max_value: float = 10.0) -> List[Dict[str, Any]]:
    bins: Dict[str, int] = {}
    overflow = 0
    for value in values:
        if value > max_value:
            overflow += 1
            continue
        lower = math.floor(value / bin_size) * bin_size
        label = f"{lower:.2f}-{lower + bin_size:.2f}"
        bins[label] = bins.get(label, 0) + 1
    rows = [{"range": key, "count": bins[key]} for key in sorted(bins, key=lambda item: float(item.split("-")[0]))]
    if overflow:
        rows.append({"range": f">{max_value:.2f}", "count": overflow})
    return rows


def scaled_histogram(
    values: List[float],
    scale: float,
    bin_size: float,
    max_value: float,
) -> List[Dict[str, Any]]:
    return histogram([value / scale for value in values], bin_size=bin_size, max_value=max_value)


def make_summary(index: Dict[str, Any]) -> Dict[str, Any]:
    videos = index["videos"]
    intervals_sec: List[float] = []
    gop_avg_packet_sizes = []
    keyframe_counts = []
    gop_counts = []
    total_frames = 0
    total_duration = 0.0
    total_size = 0
    for video in videos:
        keyframe_counts.append(video["keyframe_count"])
        gop_counts.append(video.get("gop_count", len(video.get("gops", []))))
        total_frames += video.get("frame_count") or 0
        total_duration += video.get("duration_sec") or 0.0
        total_size += video.get("size_bytes") or 0
        intervals_sec.extend(intervals([frame["time_sec"] for frame in video["keyframes"]]))
        gop_avg_packet_sizes.extend(
            gop["avg_packet_size_bytes"]
            for gop in video.get("gops", [])
            if gop.get("avg_packet_size_bytes") is not None
        )

    total_keyframes = sum(keyframe_counts)
    total_gops = sum(gop_counts)
    return {
        "generated_at": index.get("generated_at"),
        "dataset_root": index.get("dataset_root"),
        "physical_video_count": len(videos),
        "metadata_row_count": index.get("metadata", {}).get("row_count"),
        "metadata_unique_video_count": index.get("metadata", {}).get("unique_video_count"),
        "total_keyframes": total_keyframes,
        "total_gops": total_gops,
        "total_frames": total_frames,
        "total_duration_sec": total_duration,
        "total_size_bytes": total_size,
        "overall_keyframe_ratio": total_keyframes / total_frames if total_frames else None,
        "keyframes_per_video": {
            "min": min(keyframe_counts) if keyframe_counts else 0,
            "mean": statistics.fmean(keyframe_counts) if keyframe_counts else 0,
            "median": statistics.median(keyframe_counts) if keyframe_counts else 0,
            "max": max(keyframe_counts) if keyframe_counts else 0,
        },
        "interval_sec": {
            "min": min(intervals_sec) if intervals_sec else None,
            "p25": percentile(intervals_sec, 0.25),
            "mean": statistics.fmean(intervals_sec) if intervals_sec else None,
            "median": statistics.median(intervals_sec) if intervals_sec else None,
            "p75": percentile(intervals_sec, 0.75),
            "max": max(intervals_sec) if intervals_sec else None,
            "histogram": histogram(intervals_sec),
        },
        "gop_avg_packet_size_bytes": {
            "min": min(gop_avg_packet_sizes) if gop_avg_packet_sizes else None,
            "p25": percentile(gop_avg_packet_sizes, 0.25),
            "mean": statistics.fmean(gop_avg_packet_sizes) if gop_avg_packet_sizes else None,
            "median": statistics.median(gop_avg_packet_sizes) if gop_avg_packet_sizes else None,
            "p75": percentile(gop_avg_packet_sizes, 0.75),
            "max": max(gop_avg_packet_sizes) if gop_avg_packet_sizes else None,
            "histogram_kb": scaled_histogram(gop_avg_packet_sizes, scale=1024.0, bin_size=32.0, max_value=1024.0),
        },
    }


def public_video(video: Dict[str, Any], include_keyframes: bool = False) -> Dict[str, Any]:
    keys = [
        "name",
        "size_bytes",
        "codec",
        "width",
        "height",
        "fps",
        "duration_sec",
        "frame_count",
        "keyframe_count",
        "keyframe_ratio",
        "mean_keyframe_interval_sec",
        "median_keyframe_interval_sec",
        "min_keyframe_interval_sec",
        "max_keyframe_interval_sec",
        "gop_count",
        "mean_gop_avg_packet_size_bytes",
        "median_gop_avg_packet_size_bytes",
        "min_gop_avg_packet_size_bytes",
        "max_gop_avg_packet_size_bytes",
        "hlvid_questions",
    ]
    payload = {key: video.get(key) for key in keys}
    if include_keyframes:
        payload["keyframes"] = video["keyframes"]
        payload["gops"] = video.get("gops", [])
    return payload


def safe_thumb_stem(video_name: str, ordinal: int) -> str:
    stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in video_name)
    return f"{stem}_{ordinal:05d}.jpg"


def render_keyframe(video: Dict[str, Any], ordinal: int, thumb_dir: Path, width: int, quality: int) -> bytes:
    keyframes = video["keyframes"]
    if ordinal < 0 or ordinal >= len(keyframes):
        raise IndexError(f"Keyframe ordinal {ordinal} out of range")
    thumb_dir.mkdir(parents=True, exist_ok=True)
    target = thumb_dir / safe_thumb_stem(video["name"], ordinal)
    if target.exists():
        return target.read_bytes()

    keyframe = keyframes[ordinal]
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{float(keyframe['time_sec']):.6f}",
        "-i",
        video["path"],
        "-frames:v",
        "1",
        "-q:v",
        str(max(2, min(31, round((100 - quality) / 3) + 2))),
    ]
    if width > 0:
        command.extend(["-vf", f"scale={width}:-2"])
    command.extend(["-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"])
    result = subprocess.run(command, check=True, capture_output=True)
    payload = result.stdout
    if not payload:
        raise ValueError(f"Failed to render {video['name']} keyframe {ordinal}")
    target.write_bytes(payload)
    return payload


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HLVid Codec Browser</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1e2329;
      --muted: #657181;
      --line: #d8dde5;
      --panel: #f4f6f8;
      --accent: #0c6b58;
      --accent-2: #77532b;
      --bg: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      border-bottom: 1px solid var(--line);
      padding: 18px 24px 14px;
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      flex-wrap: wrap;
    }
    h1 { margin: 0; font-size: 22px; font-weight: 700; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 16px; letter-spacing: 0; }
    main { display: grid; grid-template-columns: 380px 1fr; min-height: calc(100vh - 70px); }
    aside { border-right: 1px solid var(--line); padding: 18px; overflow: auto; }
    section { padding: 18px 22px; overflow: auto; }
    .stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 18px; }
    .stat { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 10px; min-width: 0; }
    .stat strong { display: block; font-size: 18px; margin-bottom: 2px; overflow-wrap: anywhere; }
    .stat span { color: var(--muted); font-size: 12px; }
    .controls { display: grid; gap: 8px; margin-bottom: 12px; }
    input, select {
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 8px;
      background: white;
      color: var(--ink);
      font: inherit;
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 7px 6px; text-align: left; }
    th { color: var(--muted); font-weight: 600; position: sticky; top: 0; background: white; z-index: 1; }
    tr { cursor: pointer; }
    tr:hover, tr.selected { background: #edf5f2; }
    .meta { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-bottom: 14px; }
    .video-row { display: grid; grid-template-columns: minmax(260px, 520px) 1fr; gap: 18px; align-items: start; margin-bottom: 18px; }
    video { width: 100%; max-height: 340px; background: #111; border: 1px solid var(--line); border-radius: 6px; }
    canvas { width: 100%; height: 160px; border: 1px solid var(--line); border-radius: 6px; background: white; }
    .keyframes { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }
    .frame { border: 1px solid var(--line); border-radius: 6px; overflow: hidden; background: var(--panel); }
    .frame img { width: 100%; display: block; aspect-ratio: 16 / 9; object-fit: cover; background: #d8dde5; }
    .caption { display: flex; justify-content: space-between; gap: 8px; padding: 7px 8px; font-size: 12px; color: var(--muted); }
    .caption b { color: var(--ink); font-weight: 600; }
    .subtle { color: var(--muted); font-size: 13px; }
    .pill { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 2px 8px; color: var(--muted); font-size: 12px; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); max-height: 52vh; }
      .video-row, .meta { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<header>
  <div>
    <h1>HLVid Codec Browser</h1>
    <div class="subtle" id="dataset"></div>
  </div>
  <span class="pill" id="generated"></span>
</header>
<main>
  <aside>
    <h2>Aggregate Keyframes</h2>
    <div class="stats" id="stats"></div>
    <canvas id="histogram" width="640" height="240"></canvas>
    <h2 style="margin-top:18px">GOP Packet Density</h2>
    <canvas id="gopHistogram" width="640" height="240"></canvas>
    <h2 style="margin-top:18px">Videos</h2>
    <div class="controls">
      <input id="filter" placeholder="Filter videos">
      <select id="sort">
        <option value="name">Name</option>
        <option value="gop_density_desc">Highest GOP density</option>
        <option value="keyframes_desc">Most keyframes</option>
        <option value="interval_asc">Shortest interval</option>
        <option value="duration_desc">Longest duration</option>
      </select>
    </div>
    <table>
      <thead><tr><th>Video</th><th>KF</th><th>Gap</th><th>GOP KB</th></tr></thead>
      <tbody id="videos"></tbody>
    </table>
  </aside>
  <section>
    <div id="empty" class="subtle">Select a video to inspect rendered keyframes.</div>
    <div id="detail" hidden>
      <h2 id="videoTitle"></h2>
      <div class="meta" id="meta"></div>
      <div class="video-row">
        <video id="video" controls preload="metadata"></video>
        <div>
          <h2>Intervals</h2>
          <canvas id="videoHistogram" width="640" height="240"></canvas>
        </div>
      </div>
      <h2 id="keyframeTitle"></h2>
      <div class="keyframes" id="keyframes"></div>
    </div>
  </section>
</main>
<script>
const state = { summary: null, videos: [], selected: null };
const fmt = new Intl.NumberFormat();
const sec = value => value == null ? "n/a" : `${Number(value).toFixed(3)}s`;
const pct = value => value == null ? "n/a" : `${(Number(value) * 100).toFixed(2)}%`;
const mb = value => value == null ? "n/a" : `${(Number(value) / 1024 / 1024).toFixed(1)} MB`;
const kb = value => value == null ? "n/a" : `${(Number(value) / 1024).toFixed(1)} KB`;

function stat(label, value) {
  return `<div class="stat"><strong>${value}</strong><span>${label}</span></div>`;
}

function drawHistogram(canvas, rows, color, label) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  if (!rows.length) return;
  const pad = 34;
  const max = Math.max(...rows.map(row => row.count));
  const barW = (w - pad * 2) / rows.length;
  ctx.strokeStyle = "#d8dde5";
  ctx.beginPath();
  ctx.moveTo(pad, 12);
  ctx.lineTo(pad, h - pad);
  ctx.lineTo(w - 10, h - pad);
  ctx.stroke();
  rows.forEach((row, i) => {
    const x = pad + i * barW + 1;
    const barH = Math.max(1, (h - pad - 18) * row.count / max);
    ctx.fillStyle = color;
    ctx.fillRect(x, h - pad - barH, Math.max(1, barW - 2), barH);
  });
  ctx.fillStyle = "#657181";
  ctx.font = "12px system-ui";
  ctx.fillText(label, pad, h - 8);
  ctx.fillText(String(max), 6, 18);
}

function intervals(keyframes) {
  const rows = [];
  for (let i = 1; i < keyframes.length; i++) rows.push(keyframes[i].time_sec - keyframes[i - 1].time_sec);
  return rows;
}

function histogram(values, binSize = 0.25, maxValue = 10) {
  const bins = new Map();
  let overflow = 0;
  values.forEach(value => {
    if (value > maxValue) {
      overflow += 1;
      return;
    }
    const lower = Math.floor(value / binSize) * binSize;
    const key = `${lower.toFixed(2)}-${(lower + binSize).toFixed(2)}`;
    bins.set(key, (bins.get(key) || 0) + 1);
  });
  const rows = [...bins.entries()]
    .sort((a, b) => Number(a[0].split("-")[0]) - Number(b[0].split("-")[0]))
    .map(([range, count]) => ({ range, count }));
  if (overflow) rows.push({ range: `>${maxValue.toFixed(2)}`, count: overflow });
  return rows;
}

async function load() {
  const [summary, videos] = await Promise.all([
    fetch("/api/summary").then(r => r.json()),
    fetch("/api/videos").then(r => r.json()),
  ]);
  state.summary = summary;
  state.videos = videos.videos;
  document.getElementById("dataset").textContent = summary.dataset_root;
  document.getElementById("generated").textContent = `indexed ${summary.generated_at}`;
  document.getElementById("stats").innerHTML = [
    stat("physical videos", fmt.format(summary.physical_video_count)),
    stat("HLVid questions", summary.metadata_row_count == null ? "n/a" : fmt.format(summary.metadata_row_count)),
    stat("keyframes", fmt.format(summary.total_keyframes)),
    stat("GOPs", fmt.format(summary.total_gops)),
    stat("frames", fmt.format(summary.total_frames)),
    stat("duration", sec(summary.total_duration_sec)),
    stat("overall keyframe ratio", pct(summary.overall_keyframe_ratio)),
    stat("median interval", sec(summary.interval_sec.median)),
    stat("median GOP packet avg", kb(summary.gop_avg_packet_size_bytes.median)),
  ].join("");
  drawHistogram(document.getElementById("histogram"), summary.interval_sec.histogram, "#0c6b58", "keyframe interval seconds");
  drawHistogram(document.getElementById("gopHistogram"), summary.gop_avg_packet_size_bytes.histogram_kb, "#77532b", "GOP average packet size KB");
  renderVideoTable();
  if (state.videos.length) selectVideo(state.videos[0].name);
}

function sortedVideos() {
  const query = document.getElementById("filter").value.toLowerCase();
  const sort = document.getElementById("sort").value;
  const rows = state.videos.filter(video => video.name.toLowerCase().includes(query));
  rows.sort((a, b) => {
    if (sort === "gop_density_desc") return (b.mean_gop_avg_packet_size_bytes ?? 0) - (a.mean_gop_avg_packet_size_bytes ?? 0);
    if (sort === "keyframes_desc") return b.keyframe_count - a.keyframe_count;
    if (sort === "interval_asc") return (a.mean_keyframe_interval_sec ?? 9999) - (b.mean_keyframe_interval_sec ?? 9999);
    if (sort === "duration_desc") return (b.duration_sec ?? 0) - (a.duration_sec ?? 0);
    return a.name.localeCompare(b.name);
  });
  return rows;
}

function renderVideoTable() {
  const tbody = document.getElementById("videos");
  tbody.innerHTML = sortedVideos().map(video => `
    <tr data-name="${video.name}" class="${state.selected && state.selected.name === video.name ? "selected" : ""}">
      <td>${video.name}</td>
      <td>${fmt.format(video.keyframe_count)}</td>
      <td>${sec(video.mean_keyframe_interval_sec)}</td>
      <td>${kb(video.mean_gop_avg_packet_size_bytes)}</td>
    </tr>
  `).join("");
  tbody.querySelectorAll("tr").forEach(row => row.addEventListener("click", () => selectVideo(row.dataset.name)));
}

async function selectVideo(name) {
  const video = await fetch(`/api/video?name=${encodeURIComponent(name)}`).then(r => r.json());
  state.selected = video;
  document.getElementById("empty").hidden = true;
  document.getElementById("detail").hidden = false;
  document.getElementById("videoTitle").textContent = video.name;
  document.getElementById("video").src = `/media/video?name=${encodeURIComponent(video.name)}`;
  document.getElementById("meta").innerHTML = [
    stat("codec", video.codec),
    stat("HLVid questions", video.hlvid_questions && video.hlvid_questions.question_count ? fmt.format(video.hlvid_questions.question_count) : "0"),
    stat("resolution", `${video.width}x${video.height}`),
    stat("fps", video.fps == null ? "n/a" : Number(video.fps).toFixed(2)),
    stat("duration", sec(video.duration_sec)),
    stat("frames", fmt.format(video.frame_count || 0)),
    stat("keyframes", fmt.format(video.keyframe_count)),
    stat("GOP avg packet", kb(video.mean_gop_avg_packet_size_bytes)),
    stat("keyframe ratio", pct(video.keyframe_ratio)),
    stat("file size", mb(video.size_bytes)),
  ].join("");
  drawHistogram(document.getElementById("videoHistogram"), histogram(intervals(video.keyframes)), "#77532b", "keyframe interval seconds");
  document.getElementById("keyframeTitle").textContent = `${fmt.format(video.keyframes.length)} rendered keyframes`;
  document.getElementById("keyframes").innerHTML = video.keyframes.map(frame => `
    <div class="frame">
      <img loading="lazy" src="/media/keyframe?name=${encodeURIComponent(video.name)}&ordinal=${frame.ordinal}" alt="${video.name} keyframe ${frame.ordinal}">
      <div class="caption"><b>#${frame.ordinal}</b><span>${sec(frame.time_sec)} · ${kb(frame.gop_avg_packet_size_bytes)}</span></div>
    </div>
  `).join("");
  renderVideoTable();
}

document.getElementById("filter").addEventListener("input", renderVideoTable);
document.getElementById("sort").addEventListener("change", renderVideoTable);
load().catch(error => {
  document.body.innerHTML = `<pre style="padding:24px;color:#9b1c1c">${error.stack || error}</pre>`;
});
</script>
</body>
</html>
"""


class CodecBrowserHandler(BaseHTTPRequestHandler):
    index: Dict[str, Any] = {}
    videos_by_name: Dict[str, Dict[str, Any]] = {}
    thumb_dir = Path(DEFAULT_THUMB_DIR)
    jpeg_width = 360
    jpeg_quality = 82

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_bytes(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, payload: Any) -> None:
        self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json")

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        payload = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def query(self) -> Dict[str, List[str]]:
        return urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

    def selected_video(self) -> Dict[str, Any]:
        name = self.query().get("name", [""])[0]
        if name not in self.videos_by_name:
            raise KeyError(f"Unknown video: {name}")
        return self.videos_by_name[name]

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/summary":
                self.send_json(make_summary(self.index))
            elif parsed.path == "/api/videos":
                self.send_json({"videos": [public_video(video) for video in self.index["videos"]]})
            elif parsed.path == "/api/video":
                self.send_json(public_video(self.selected_video(), include_keyframes=True))
            elif parsed.path == "/media/keyframe":
                query = self.query()
                ordinal = int(query.get("ordinal", ["0"])[0])
                payload = render_keyframe(
                    self.selected_video(),
                    ordinal,
                    self.thumb_dir,
                    self.jpeg_width,
                    self.jpeg_quality,
                )
                self.send_bytes(payload, "image/jpeg")
            elif parsed.path == "/media/video":
                self.send_video(self.selected_video())
            else:
                self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")
        except BrokenPipeError:
            return
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def send_video(self, video: Dict[str, Any]) -> None:
        path = Path(video["path"])
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if range_header:
            unit, _, value = range_header.partition("=")
            if unit != "bytes":
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            start_text, _, end_text = value.partition("-")
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                self.wfile.write(handle.read(length))
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)


def serve(index: Dict[str, Any], args: argparse.Namespace) -> None:
    CodecBrowserHandler.index = index
    CodecBrowserHandler.videos_by_name = {video["name"]: video for video in index["videos"]}
    CodecBrowserHandler.thumb_dir = Path(args.thumb_dir)
    CodecBrowserHandler.jpeg_width = args.jpeg_width
    CodecBrowserHandler.jpeg_quality = args.jpeg_quality
    server = ThreadingHTTPServer((args.host, args.port), CodecBrowserHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving HLVid codec browser at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    cache_path = Path(args.cache)
    index = load_or_build_index(dataset_root, cache_path, args.rebuild, args.limit)
    if args.build_only:
        print(f"Wrote keyframe index to {cache_path}")
        return
    serve(index, args)


if __name__ == "__main__":
    main()
