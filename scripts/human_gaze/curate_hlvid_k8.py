"""Verify and curate the completed six-seed K16-trained/K8 HLVid evaluation.

CPU-only; reads completed artifacts without rewriting them. Historical reference
comparison is descriptive because its decode/runtime provenance is incomplete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from scripts.runners.hlvid_evidence import _validate_record, resume_compatibility_key


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def table(path, records):
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def cluster_interval(difference, video_ids, iterations=10000, seed=170926):
    """Paired question-micro difference; sample videos, keeping all six seeds."""
    videos = sorted(set(video_ids))
    totals = np.array([difference[:, np.array(video_ids) == v].sum() for v in videos])
    counts = np.array([difference[:, np.array(video_ids) == v].size for v in videos])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(videos), size=(iterations, len(videos)))
    draws = totals[samples].sum(1) / counts[samples].sum(1)
    return dict(difference=float(difference.mean()), ci90=np.quantile(draws, [.05, .95]).tolist(),
                cluster_unit='video', clusters=len(videos), draws=iterations, seed=seed,
                interpretation='Post-hoc descriptive interval conditional on these six trained policies and historical reference; not seed-resampled or causal.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-root', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    parquet = Path(cfg['dataset']['root']) / 'data/test-00000-of-00001.parquet'
    assert digest(parquet) == cfg['dataset']['sha256']
    source = sorted(pq.read_table(parquet).to_pylist(), key=lambda r: r['question_id'])
    assert [r['question_id'] for r in source] == list(range(268))
    identity_fields = ['question_id', 'video_path', 'category', 'answer']

    def verify_answers(answer_rows):
        assert len(answer_rows) == 268
        for actual, expected in zip(answer_rows, source, strict=True):
            assert all(actual[k] == expected[k] for k in identity_fields)
            match = re.search(r'\b([ABCD])\b', actual['prediction'].strip(), re.I)
            letter = match.group(1).upper() if match else ''
            assert actual['prediction_letter'] == letter
            assert actual['is_correct'] == (letter == expected['answer'].upper())

    ref = rows(args.reference / 'results.jsonl')
    verify_answers(ref)
    ref_summary = read(args.reference / 'summary.json')
    assert digest(args.reference / 'summary.json') == 'd08fc9b0fa8d89cf584d2fad56549c19d3a79788f065662304dc3a5e0a6e132e'
    assert sum(r['is_correct'] for r in ref) == ref_summary['num_correct'] == 130
    for key, value in dict(num_video_frames=128, num_video_frames_thumbnail=64, max_tiles_video=48,
                           max_batch_size_autogaze=16, max_batch_size_siglip=32, torch_dtype='bfloat16').items():
        assert ref_summary[key] == value
    ref_values = np.array([r['is_correct'] for r in ref], dtype=float)
    per_seed, provenance, matrices, paired_rows, resource_rows = [], [], [], [], []
    states_total = Counter()
    for seed_cfg in cfg['seeds']:
        seed = seed_cfg['base_seed']
        matches = list(args.raw_root.glob(f'*_seed{seed}'))
        assert len(matches) == 1
        run = matches[0]
        answer_rows = rows(run / 'results.jsonl')
        verify_answers(answer_rows)
        summary = read(run / 'summary.json')
        manifest = read(run / 'run_manifest.json')
        identity = manifest['inference_identity']
        assert manifest['compatibility_key'] == resume_compatibility_key(identity)
        assert summary['code_commit'] == 'b1cf8c2f572033b84d4c4d47e65fa3faca6bfcda'
        assert summary['code_dirty'] is False and summary['human_gaze_split_accessed'] is False
        assert summary['base_seed'] == seed and identity['exact_budget'] == 8
        assert summary['dataset_parquet_sha256'] == cfg['dataset']['sha256']
        assert identity['autogaze_checkpoint_sha256'] == seed_cfg['checkpoint_sha256']['model.safetensors']
        for name, expected in seed_cfg['checkpoint_sha256'].items():
            assert digest(Path(seed_cfg['checkpoint']) / name) == expected
        for file, key in [('results.jsonl', 'results_jsonl_sha256'), ('run_manifest.json', 'run_manifest_sha256'), ('events.jsonl', 'event_output_sha256')]:
            assert digest(run / file) == summary[key]
        protocol = identity['protocol']
        for key in ['num_video_frames', 'num_video_frames_thumbnail', 'max_tiles_video', 'tile_len', 'max_new_tokens', 'torch_dtype']:
            assert protocol[key] == cfg['protocol'][key]
        assert protocol['generation'] == 'greedy; do_sample=false'
        assert protocol['allow_eos'] is False and protocol['fixed_fine_only'] is True
        completed, attempts, terminal, record_hashes = {}, set(), set(), {}
        states = Counter()
        telemetry = []
        for path in sorted((run / 'evidence_records').glob('*.json')):
            event = read(path)
            record_hashes[path.name] = digest(path)
            _validate_record(event, manifest['compatibility_key'])
            states[event['state']] += 1
            pair = (event['example_key'], event['attempt_id'])
            if event['state'] == 'attempt_started':
                assert pair not in attempts
                attempts.add(pair)
            if event['state'] in ('completed_answer', 'failed'):
                assert pair not in terminal
                terminal.add(pair)
            if event['state'] != 'completed_answer':
                continue
            qid = event['question_id']
            assert qid not in completed and event['answer'] == answer_rows[qid]
            completed[qid] = event['answer']
            decode = event['decode']
            expected = np.round(np.linspace(0, decode['source_frame_count'] - 1, 128)).astype(int).tolist()
            assert decode['decode_usable'] and not decode['sampled_read_failure']
            assert decode['tail_padding_count'] == 0 and not decode['legacy_output_substituted']
            assert decode['effective_indices'] == decode['intended_indices'] == expected
            for call in event['raw_decoder_calls']:
                action_ids = np.asarray(call['decoder_action_ids'])
                assert action_ids.shape[-1] == 8 and ((action_ids >= 69) & (action_ids <= 264)).all()
                assert (np.diff(np.sort(action_ids, axis=-1), axis=-1) != 0).all()
            context = event['context']
            assert context['context_truncated'] is False
            counters = event['counters']
            for key, expected in [('raw_decoder_spatial_actions_per_tile_frame', 8), ('post_adaptation_valid_patches_per_tile_frame', 32), ('post_adaptation_padded_slots_per_tile_frame', 32)]:
                c = counters[key]
                assert c['availability'] == 'complete' and c['min'] == c['max'] == expected
            telemetry.append(context['expanded_visual_tokens'])
        assert len(completed) == 268 and terminal <= attempts
        states_total.update(states)
        values = np.array([r['is_correct'] for r in answer_rows], dtype=float)
        assert summary['num_examples'] == 268 and summary['num_correct'] == int(values.sum())
        assert abs(summary['accuracy'] - values.mean()) < 1e-12
        wins = int(((values == 1) & (ref_values == 0)).sum())
        losses = int(((values == 0) & (ref_values == 1)).sum())
        per_seed.append(dict(seed=seed, correct=int(values.sum()), total=268, accuracy=float(values.mean()),
                             difference_vs_reference=float((values-ref_values).mean()), wins=wins, losses=losses))
        resource_rows.append(dict(seed=seed, raw_actions_per_crop=8, retained_patches_per_tile_frame=32,
                                  padded_slots_per_tile_frame=32, visual_tokens_mean=float(np.mean(telemetry)),
                                  visual_tokens_min=min(telemetry), visual_tokens_max=max(telemetry)))
        for row in answer_rows:
            paired_rows.append(dict(seed=seed, question_id=row['question_id'], video_path=row['video_path'], category=row['category'],
                                    trained_correct=int(row['is_correct']), reference_correct=int(ref_values[row['question_id']])))
        provenance.append(dict(run_id=run.name, artifact_path=str(run), hashes={p.name: digest(p) for p in run.iterdir() if p.is_file()},
                               evidence_index_sha256=hashlib.sha256(json.dumps(record_hashes, sort_keys=True).encode()).hexdigest(),
                               record_count=len(record_hashes), states=dict(states), unterminated_attempts=len(attempts-terminal),
                               checkpoint_hashes=seed_cfg['checkpoint_sha256']))
        matrices.append(values)
        print(f'Verified seed {seed}: {int(values.sum())}/268; evidence states {dict(states)}', flush=True)
    values = np.array(matrices)
    comparison = cluster_interval(values-ref_values, [r['video_path'] for r in source])
    accounting = list(csv.DictReader((out/'slurm_allocations.psv').open(), delimiter='|'))
    gpu_hours = sum(int(r['ElapsedRaw']) for r in accounting) / 3600
    metrics = dict(status='complete_verified', seed_accuracy_mean=float(values.mean()),
                   seed_accuracy_population_sd=float(values.mean(1).std()), per_seed=per_seed,
                   historical_reference=dict(correct=130, total=268, accuracy=float(ref_values.mean())),
                   paired_comparison=comparison, resources=resource_rows, evidence_states=dict(states_total),
                   completed_logical_answers=int(values.size), allocated_h100_hours=gpu_hours,
                   logical_rollout_boundary='1608 completed answer passes; unfinished attempts may add work, see evidence states and allocation ledger')
    write(out/'metrics.json', metrics)
    write(out/'manifest.json', dict(execution_commit='b1cf8c2f572033b84d4c4d47e65fa3faca6bfcda', runs=provenance,
         dataset_sha256=digest(parquet), historical_reference_path=str(args.reference),
         historical_reference_hashes={name:digest(args.reference/name) for name in ['results.jsonl','summary.json']},
         protocol_comparability='Question IDs, answers, videos, recorded frame/thumbnail/tile settings, model path, dtype and batch sizes match. Historical reference lacks per-question exact-decode, runtime/model hashes and token telemetry; strict runtime/token matching is unverified.',
         accounting_includes='Both decode failures, both explicit retries, and both preempted/requeued seed-440831 segments.',
         analysis_script_sha256=digest(__file__)))
    analysis_cfg = dict(cfg, analysis_posthoc=dict(cluster_bootstrap_draws=10000, seed=170926, interval=0.90,
                        cluster='video', trained_seed_set='fixed six seeds', reference='single historical normal AutoGaze run',
                        rationale='Descriptive paired uncertainty following existing HLVid curation convention; not a preregistered pass/fail test'))
    (out/'config.yaml').write_text(yaml.safe_dump(analysis_cfg, sort_keys=False))
    table(out/'per_seed.csv', per_seed)
    table(out/'paired_correctness.csv', paired_rows)
    table(out/'resources.csv', resource_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.scatter(range(6), values.mean(1)*100, color='#24688d', s=45, label='K16-trained, evaluated K8')
    ax.axhline(ref_values.mean()*100, color='#686868', linestyle='--', label='Historical normal AutoGaze')
    ax.set(xticks=range(6), xticklabels=[str(r['seed']) for r in per_seed], ylabel='HLVid accuracy (%)', xlabel='Training seed', ylim=(46,52))
    ax.legend(fontsize=8)
    fig.tight_layout()
    for extension in ['png','pdf']:
        fig.savefig(out/f'accuracy_by_seed.{extension}', dpi=180)
    print(json.dumps(dict(mean=metrics['seed_accuracy_mean'], comparison=comparison, gpu_hours=gpu_hours)))


if __name__ == '__main__':
    main()
