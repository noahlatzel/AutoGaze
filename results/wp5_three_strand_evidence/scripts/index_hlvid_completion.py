#!/usr/bin/env python3
"""Index published HLVid controls/paired accuracy; no scientific recomputation.

Run at the AutoGaze repository root. --patch emits an additive apply_patch
transfer; --check verifies the indexed package against its frozen published base.
Only compact Git inputs are read. No NumPy, model, data or scheduler access.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import shlex

BASE = "473662d93d9005becb5e996e92b87ef6fa8b6368"
ROOT = Path("results/wp5_three_strand_evidence")
VARIABLE = Path("results/human_gaze/20260914-0202_r2f-hlvid-complete-paired-accuracy_6b4eef7")
CONTROLS = Path("experiments/human_gaze/results/r2e_hlvid_k16_causal_controls/20260914-0149_r2e-k16-causal-curation_084d563")
VCOMMIT = "03cd606257b9f3f28ac295e62533453b2e2b2612"
CCOMMIT = "51b41662b801afb4b3ec3ae870b80403ce3abcb3"
EDITABLE = {"README.md", "manifest.json", "metrics.csv", "claims.csv",
            "artifacts.csv", "figures.csv", "studies.csv", "owner_status.csv",
            "readiness.json", "READINESS.md", "FIGURE_REGENERATION.md"}


def git(*args):
    return subprocess.check_output(["git", *map(str, args)])


def frozen(path):
    return git("show", f"{BASE}:{path}")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def csv_bytes(fields, rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writerows({k: "" if v is None else v for k, v in row.items()} for row in rows)
    return out.getvalue().encode()


def append_csv(outputs, name, rows):
    original = frozen(ROOT / name)
    fields = next(csv.reader(io.StringIO(original.decode())))
    outputs[ROOT / name] = original + csv_bytes(fields, rows)


def source_addendum(outputs, label, bundle, commit, contract):
    manifest = load(bundle / "manifest.json")
    hashes = manifest.get("outputs", manifest.get("bundle_file_hashes"))
    if (bundle / "supplement_manifest.json").exists():
        supplement = load(bundle / "supplement_manifest.json")
        assert supplement["core_manifest_sha256"] == sha((bundle / "manifest.json").read_bytes())
        hashes = dict(hashes, **supplement["supplement_file_hashes"])
    files = git("ls-tree", "-r", "--name-only", commit, bundle).decode().splitlines()
    index = {}
    copies = []
    for source in files:
        relative = str(Path(source).relative_to(bundle))
        data = Path(source).read_bytes()
        assert data == git("show", f"{commit}:{source}"), source
        if relative in hashes:
            assert sha(data) == hashes[relative], source
        index[relative] = {"sha256": sha(data), "bytes": len(data),
                           "git_blob": git("rev-parse", f"{commit}:{source}").decode().strip(),
                           "url": f"https://github.com/noahlatzel/AutoGaze/blob/{commit}/{source}"}
        # Existing paired question audit stays linked; do not duplicate answers.
        if Path(source).suffix not in {".pdf", ".png", ".svg"} and relative != "paired_questions.csv":
            outputs[ROOT / "addenda" / label / "source_bundle" / relative] = data
            copies.append(relative)
    addendum = {"schema_version": 1, "repository": "noahlatzel/AutoGaze",
                "result_commit": commit, "result_bundle": str(bundle),
                "curator_commit": manifest.get("curation_commit", manifest.get("curation_code_commit")),
                "source_files": index, "local_source_copies": copies,
                "contract": contract, "new_inference": False,
                "new_bootstrap": False, "new_render": False,
                "heavy_artifacts_reverified_here": False}
    outputs[ROOT / "addenda" / label / "manifest.json"] = json_bytes(addendum)
    return addendum


def build():
    outputs = {}
    c = load(CONTROLS / "metrics.json")
    v = load(VARIABLE / "metrics.json")
    assert c["primary_metric"] == "question_micro_accuracy"
    assert v["primary_estimand"] == "equal-seed_mean_macro_video_exact_accuracy"
    assert v["allocation_status"] == "unavailable_pending_validated_replay"
    source_addendum(outputs, "r2e_controls", CONTROLS, CCOMMIT, {
        "status": "complete_descriptive", "primary_metric": c["primary_metric"],
        "sensitivity_metric": c["sensitivity_metric"], "training_seed_count": 6,
        "base_seed_to_training_seed": {str(x): x + 100000 for x in range(440826, 440832)},
        "deterministic_control_training_seed_count": None,
        "seed_interval": "90% Student t, df5, six independent trained endpoints",
        "video_interval": "90% paired 77-video bootstrap, conditional on six endpoints",
        "split": "HLVid official test; historically exposed; protected human-gaze test unopened",
        "hardware": "trained H100 NVL; controls A40; QA comparison approved; timing not hardware-matched",
        "compute": "A40 reservation segments include preemptions; per-example stage telemetry separate",
        "source_checkpoint_code_data_identities": "source_bundle/manifest.json; source_bundle/config.yaml",
    })
    source_addendum(outputs, "r2f_variable_accuracy", VARIABLE, VCOMMIT, {
        "status": "complete_accuracy_only_costs_missing", "primary_metric": "macro_video_accuracy",
        "sensitivity_metric": "question_micro_accuracy", "training_seed_count": 3,
        "base_seed_to_training_seed": {str(x): x + 300000 for x in range(440826, 440829)},
        "seed_interval": "95% Student t across three paired trained endpoints, df2",
        "video_interval": "90% paired 77-video bootstrap, same draws across both policies and all seeds",
        "allocation_status": v["allocation_status"], "variable_costs": None,
        "split": "HLVid official test; reused historically exposed benchmark; protected human-gaze test unopened",
        "hardware": "A40; actual allocation nodes preserved in scheduler/sacct_segments.psv",
        "compute": "reservation accounting is not inference latency or measured efficiency",
        "comparison": "deployment policy contrast; EOS changes later spatial ordering; not allocation-only",
        "source_checkpoint_code_data_identities": "source_bundle/manifest.json; source_bundle/config.yaml",
    })
    rows = []
    seeds6 = ";".join(map(str, range(440826, 440832)))
    seeds3 = ";".join(map(str, range(440826, 440829)))

    def metric(identifier, artifact, method, name, value, *, interval=None,
               level=None, interval_method="", seed=None, n_seeds=None,
               primary=False, contrast="", source_field="", role="level",
               status=None, bootstrap="", unit="fraction"):
        is_variable = artifact == "gaze_r2f_paired_accuracy"
        if status is None:
            status = "complete_secondary_descriptive" if is_variable else "complete_descriptive"
        row = {"metric_id": identifier, "study_id": "spatial_gaze", "artifact_id": artifact,
               "subset": "HLVid official test", "method": method,
               "budget": "variable_EOS" if method == "variable" else 16,
               "contrast": contrast, "metric": name, "value": value,
               "ci_low": interval[0] if interval else None,
               "ci_high": interval[1] if interval else None, "level": level,
               "interval_method": interval_method, "n_seeds": n_seeds,
               "seed_ids": seed if seed is not None else (seeds3 if is_variable else seeds6) if n_seeds else None,
               "primary": str(primary).lower(), "status": status,
               "seed_id_role": "base_seed" if seed is not None or n_seeds else "inapplicable_deterministic_control",
               "metric_unit": unit, "aggregation_unit": "equal-seed video macro" if name == "macro_video_accuracy" else "equal-seed question micro" if name == "question_micro_accuracy" else "declared source field",
               "n_questions": 268, "n_videos": 77, "bootstrap_seed": bootstrap,
               "source_field": source_field, "metric_role": role,
               "protocol_validity": "approved_cross_hardware_QA" if not is_variable else "paired_same_checkpoint_deployment",
               "replicate_kind": "paired_video_cluster_bootstrap" if bootstrap else "training_seed" if n_seeds else "deterministic_evaluation",
               "replicate_count": 10000 if bootstrap else n_seeds if n_seeds else 1}
        rows.append(row)
        return identifier

    ca = "gaze_r2e_controls"
    for name, values in c["trained"].items():
        metric(f"controls_trained_{name}", ca, "trained_k16", name, values["mean"],
               interval=[values["training_seed_ci90_low"], values["training_seed_ci90_high"]],
               level=0.90, interval_method="Student t across six training seeds, df5", n_seeds=6,
               primary=name == c["primary_metric"], source_field=f"metrics.json:trained.{name}")
    for endpoint in c["control_metrics"]:
        for name in (c["primary_metric"], c["sensitivity_metric"], "num_correct"):
            metric(f"controls_{endpoint['method']}_{name}", ca, endpoint["method"], name,
                   endpoint[name], primary=name == c["primary_metric"],
                   source_field=f"metrics.json:control_metrics[{endpoint['method']}].{name}",
                   unit="questions" if name == "num_correct" else "fraction")
    for endpoint in csv.DictReader((CONTROLS / "per_seed.csv").open()):
        if endpoint["method"] != "trained_k16":
            continue
        for name in (c["primary_metric"], c["sensitivity_metric"]):
            metric(f"controls_trained_seed{endpoint['base_seed']}_{name}", ca, "trained_k16", name,
                   float(endpoint[name]), n_seeds=1, seed=endpoint["base_seed"],
                   primary=name == c["primary_metric"], source_field=f"per_seed.csv:{endpoint['base_seed']}:{name}")
    for contrast in c["trained_minus_controls"]:
        for uncertainty, low, high in (("video90", "paired_video_ci90_low", "paired_video_ci90_high"),
                                       ("seed90", "training_seed_ci90_low", "training_seed_ci90_high")):
            metric(f"controls_diff_{contrast['control']}_{contrast['metric']}_{uncertainty}", ca,
                   "trained_k16", contrast["metric"], contrast["difference"],
                   interval=[contrast[low], contrast[high]], level=0.90, n_seeds=6,
                   interval_method="paired video bootstrap; six endpoints fixed" if uncertainty == "video90" else "Student t across six paired training seeds, df5",
                   bootstrap=20260909 if uncertainty == "video90" else "",
                   primary=contrast["metric"] == c["primary_metric"], role="paired_difference",
                   contrast=f"trained_k16_minus_{contrast['control']}", source_field=f"contrast_summary.csv:{contrast['control']}:{contrast['metric']}:{low}/{high}")
    for contrast in csv.DictReader((CONTROLS / "paired_per_seed.csv").open()):
        metric(f"controls_diff_{contrast['control']}_seed{contrast['base_seed']}_{contrast['metric']}_video90", ca,
               "trained_k16", contrast["metric"], float(contrast["difference"]),
               interval=[float(contrast["paired_video_ci90_low"]), float(contrast["paired_video_ci90_high"])],
               level=0.90, interval_method="paired video bootstrap within one endpoint", n_seeds=1,
               seed=contrast["base_seed"], bootstrap=20260909, role="paired_difference",
               primary=contrast["metric"] == c["primary_metric"], contrast=f"trained_k16_minus_{contrast['control']}",
               source_field=f"paired_per_seed.csv:{contrast['control']}:{contrast['base_seed']}:{contrast['metric']}")
    for method, seconds in c["resource_allocation_seconds"].items():
        metric(f"controls_{method}_allocated_seconds", ca, method, "reserved_A40_seconds", seconds,
               source_field=f"metrics.json:resource_allocation_seconds.{method}", unit="seconds",
               role="reservation_accounting_not_latency")
    va = "gaze_r2f_paired_accuracy"
    for endpoint in v["per_seed"]:
        for method in ("variable", "forced_k16"):
            for name, value in endpoint[method].items():
                if name in {"allocation_evidence_source", "legacy_process_counter_scope_complete", "num_videos"}:
                    continue
                is_accuracy = name in {"macro_video_accuracy", "question_micro_accuracy"}
                metric(f"variable_{method}_seed{endpoint['base_seed']}_{name}", va, method, name, value,
                       n_seeds=1, seed=endpoint["base_seed"], primary=name == "macro_video_accuracy",
                       source_field=f"metrics.json:per_seed[{endpoint['base_seed']}].{method}.{name}",
                       role="accuracy" if is_accuracy else "nominal_contract" if method == "forced_k16" and value is not None and name != "num_correct" else "source_value",
                       status=("missing_pending_replay" if method == "variable" else "missing_legacy_context_counter") if value is None else "complete_secondary_descriptive",
                       unit="fraction" if is_accuracy or name == "decoder_eos_action_rate" else "count")
        for name in ("macro_video", "question_micro"):
            metric(f"variable_diff_seed{endpoint['base_seed']}_{name}", va, "variable", f"{name}_accuracy",
                   endpoint[f"variable_minus_forced_k16_{name}"], n_seeds=1, seed=endpoint["base_seed"],
                   primary=name == "macro_video", role="paired_difference", contrast="variable_minus_forced_k16",
                   source_field=f"metrics.json:per_seed[{endpoint['base_seed']}].variable_minus_forced_k16_{name}")
    for method in ("variable", "forced_k16"):
        endpoint = v["aggregate"][method]
        for name in ("macro_video", "question_micro"):
            metric(f"variable_{method}_{name}_seed95", va, method, f"{name}_accuracy",
                   endpoint[f"mean_{name}_accuracy_over_seeds"], interval=endpoint[f"{name}_accuracy_seed_t95_interval"],
                   n_seeds=3, level=0.95, interval_method="Student t across three training seeds, df2",
                   primary=name == "macro_video", source_field=f"metrics.json:aggregate.{method}.{name}")
        for name in ("decoder_spatial_actions", "retained_patches", "visual_tokens", "expanded_context_tokens"):
            value = endpoint[f"mean_{name}_over_seeds"]
            metric(f"variable_{method}_mean_{name}", va, method, f"mean_{name}", value, n_seeds=3,
                   source_field=f"metrics.json:aggregate.{method}.mean_{name}_over_seeds", unit="count",
                   role="nominal_contract" if value is not None else "resource_missing",
                   status=("missing_pending_replay" if method == "variable" else "missing_legacy_context_counter") if value is None else "nominal_exact_k_contract")
    difference = v["aggregate"]["variable_minus_forced_k16"]
    for name in ("macro_video", "question_micro"):
        for uncertainty in ("video90", "seed95"):
            bootstrap = difference["paired_video_bootstrap" if name == "macro_video" else "question_micro_paired_video_bootstrap"]
            metric(f"variable_diff_{name}_{uncertainty}", va, "variable", f"{name}_accuracy",
                   difference[f"mean_{name}_accuracy_difference"], n_seeds=3,
                   interval=bootstrap["interval"] if uncertainty == "video90" else difference["seed_t95_interval" if name == "macro_video" else "question_micro_seed_t95_interval"],
                   level=0.90 if uncertainty == "video90" else 0.95,
                   interval_method="paired video bootstrap; same samples across policies and all three seeds; seeds fixed" if uncertainty == "video90" else "Student t across three paired training seeds, df2",
                   bootstrap=bootstrap["seed"] if uncertainty == "video90" else "", primary=name == "macro_video",
                   contrast="variable_minus_forced_k16", role="paired_difference",
                   source_field=f"metrics.json:aggregate.variable_minus_forced_k16:{name}:{uncertainty}")
    append_csv(outputs, "metrics.csv", rows)
    append_csv(outputs, "artifacts.csv", [
        dict(artifact_id=ca, study_id="spatial_gaze", result_bundle=str(CONTROLS),
             source_table="addenda/r2e_controls/source_bundle/contrast_summary.csv;addenda/r2e_controls/source_bundle/per_seed.csv;addenda/r2e_controls/source_bundle/control_per_example_telemetry.csv",
             repository="noahlatzel/AutoGaze", commit=CCOMMIT, data_hash=load(CONTROLS / "manifest.json")["parquet_sha256"],
             split="HLVid official test; historically exposed; human-gaze protected test unopened",
             uncertainty_unit="six training seeds and 77 paired video clusters kept separate; deterministic controls evaluated once",
             compute_boundary=c["resource_boundary"], status="complete_descriptive",
             notes="536 control QA;1608 unchanged trained QA; execution b795de0/7835a4b; actual A40 allocation segments retained; not cross-hardware latency"),
        dict(artifact_id=va, study_id="spatial_gaze", result_bundle=str(VARIABLE),
             source_table="addenda/r2f_variable_accuracy/source_bundle/paired_aggregate.csv;addenda/r2f_variable_accuracy/source_bundle/paired_videos.csv;addenda/r2f_variable_accuracy/source_bundle/per_seed.csv",
             repository="noahlatzel/AutoGaze", commit=VCOMMIT,
             data_hash=load(CONTROLS / "manifest.json")["parquet_sha256"],
             split="HLVid official test; reused historically exposed benchmark; human-gaze protected test unopened",
             uncertainty_unit="three paired training seeds, t95; 77 shared paired video clusters, bootstrap90; no joint interval",
             compute_boundary="accuracy only; variable actions/recovered patches/expanded tokens/context missing pending validated replay",
             status="complete_accuracy_only_costs_missing", notes="1608 QA; macro primary; EOS deployment contrast not allocation-only; actual allocation nodes in copied scheduler records"),
    ])
    append_csv(outputs, "claims.csv", [
        dict(claim_id="controls_primary_descriptive", study_id="spatial_gaze", claim_kind="negative_finding",
             claim_text="Six trained K16 endpoints averaged 50.3109% question-micro QA versus pretrained 50.0000% and Center16 51.1194%. Both paired 90% video intervals and primary seed90 contrast intervals include zero; superiority is not established.",
             artifact_ids=ca, metric_ids="controls_trained_question_micro_accuracy;controls_pretrained_exact_k16_question_micro_accuracy;controls_stavis_center16_question_micro_accuracy;controls_diff_pretrained_exact_k16_question_micro_accuracy_video90;controls_diff_pretrained_exact_k16_question_micro_accuracy_seed90;controls_diff_stavis_center16_question_micro_accuracy_video90;controls_diff_stavis_center16_question_micro_accuracy_seed90",
             status="complete_descriptive", limitations="Historically exposed official HLVid test; approved cross-hardware QA; deterministic controls not independent training seeds; no equivalence or cross-hardware latency inference."),
        dict(claim_id="variable_primary_descriptive", study_id="spatial_gaze", claim_kind="negative_descriptive",
             claim_text="Variable EOS averaged 48.2955% macro-video QA versus 50.0415% forced K16 across three paired endpoints. The primary difference is -1.7460 percentage points; separate video90 and seed95 intervals include zero. All three primary seed differences are negative.",
             artifact_ids=va, metric_ids="variable_variable_macro_video_seed95;variable_forced_k16_macro_video_seed95;variable_diff_macro_video_video90;variable_diff_macro_video_seed95;variable_diff_seed440826_macro_video;variable_diff_seed440827_macro_video;variable_diff_seed440828_macro_video",
             status="complete_secondary_descriptive", limitations="Deployment EOS also changes later spatial ordering; not allocation-only. Reused historical benchmark; three seed and video uncertainties not combined; micro sensitivity does not replace primary macro."),
        dict(claim_id="variable_costs_missing", study_id="spatial_gaze", claim_kind="limitation",
             claim_text="Complete paired QA does not establish variable allocation cost, efficiency or speedup. Actual variable actions, recovered patches, visual tokens and expanded context counts remain missing pending validated replay; forced-K16 actions and recovered patches are nominal contract values.",
             artifact_ids=va, metric_ids="variable_variable_mean_decoder_spatial_actions;variable_variable_mean_retained_patches;variable_variable_mean_visual_tokens;variable_variable_mean_expanded_context_tokens;variable_forced_k16_mean_decoder_spatial_actions;variable_forced_k16_mean_retained_patches",
             status="final_limitation", limitations="Calibration lengths, process-local resumed tails and reservation walltime are not full benchmark resource measurements."),
    ])
    figures = []
    for bundle, commit, artifact, figure, title, caption, notes in (
        (CONTROLS, CCOMMIT, ca, "control_comparison", "HLVid K16 controls: primary micro QA and separate uncertainties", "README.md", "Published PDF/PNG only; SVG absent, not invented. Six H100 trained endpoints versus deterministic A40 controls; timing not hardware-matched."),
        (VARIABLE, VCOMMIT, va, "actual_vs_forced_k16", "Variable EOS versus forced K16: primary macro QA", "captions.md", "Published PDF/PNG only; SVG absent. Three paired endpoints; accuracy only, costs missing."),
        (VARIABLE, VCOMMIT, va, "paired_accuracy_deltas", "Variable deployment contrast: separate seed/video uncertainty", "captions.md", "Primary macro; secondary micro; separate video90/seed95 intervals. Not allocation-only; no efficiency claim."),
    ):
        prefix = f"https://github.com/noahlatzel/AutoGaze/blob/{commit}/{bundle}"
        label = "r2e_controls" if artifact == ca else "r2f_variable_accuracy"
        figures.append(dict(figure_id=f"hlvid_{figure}", study_id="spatial_gaze", title=title,
                            source_table=f"addenda/{label}/source_bundle/{'contrast_summary.csv;addenda/r2e_controls/source_bundle/per_seed.csv' if artifact == ca else 'paired_aggregate.csv;addenda/r2f_variable_accuracy/source_bundle/per_seed.csv'}",
                            script=f"https://github.com/noahlatzel/AutoGaze/blob/{commit}/scripts/human_gaze/{'curate_hlvid_k16_controls.py' if artifact == ca else 'curate_r2d_hlvid_secondary.py'}",
                            regeneration_command="See addenda/HLVID_COMPLETION.md for immutable curator commands; reproduction includes source validation and bootstrap, unlike this index-only integration",
                            svg_path="", pdf_path=f"{prefix}/{figure}.pdf", png_path=f"{prefix}/{figure}.png",
                            status="published_source_hash_verified", notes=notes,
                            caption_path=f"{prefix}/{caption}", metadata_path=f"{prefix}/manifest.json",
                            data_checks_path=f"addenda/{label}/manifest.json"))
    append_csv(outputs, "figures.csv", figures)
    return outputs, rows


def check(outputs, rows):
    for path, expected in outputs.items():
        assert path.read_bytes() == expected, f"index differs: {path}"
    protected = []
    for line in git("ls-tree", "-r", "--name-only", BASE, ROOT).decode().splitlines():
        path = Path(line)
        if str(path.relative_to(ROOT)) not in EDITABLE:
            assert path.read_bytes() == frozen(path), f"protected file modified: {path}"
            protected.append({"path": str(path), "sha256": sha(path.read_bytes())})
    original = list(csv.DictReader(io.StringIO(frozen(ROOT / "metrics.csv").decode())))
    final = list(csv.DictReader((ROOT / "metrics.csv").open()))
    assert len(original) == 305 and final[:305] == original
    assert (ROOT / "metrics.csv").read_bytes().startswith(frozen(ROOT / "metrics.csv"))
    ids = {x["metric_id"] for x in final}
    assert len(ids) == len(final)
    artifacts = {x["artifact_id"] for x in csv.DictReader((ROOT / "artifacts.csv").open())}
    assert all(x["artifact_id"] in artifacts for x in final)
    for claim in csv.DictReader((ROOT / "claims.csv").open()):
        assert set(filter(None, claim["metric_ids"].split(";"))) <= ids
        assert set(filter(None, claim["artifact_ids"].split(";"))) <= artifacts
    missing = [x for x in rows if x["status"].startswith("missing_")]
    assert missing and all(x["value"] is None for x in missing)
    meta = load(ROOT / "manifest.json")["hlvid_completion_index"]
    assert meta["primary_control_metric"] == "question_micro_accuracy"
    assert meta["primary_variable_metric"] == "macro_video_accuracy"
    assert meta["variable_actual_costs"] is None
    assert meta["added_metric_rows"] == len(rows) and meta["total_metric_rows"] == len(final)
    source_counts = {}
    for label in ("r2e_controls", "r2f_variable_accuracy"):
        addendum = load(ROOT / "addenda" / label / "manifest.json")
        source_counts[label] = {"published_files_hashed": len(addendum["source_files"]),
                                "exact_local_copies": len(addendum["local_source_copies"])}
    assert sum(x["exact_local_copies"] for x in source_counts.values()) == 26
    table_rows = list(csv.DictReader((ROOT / "tables.csv").open()))
    assert all(x["artifact_id"] in artifacts for x in table_rows)
    assert all((ROOT / x["source_table"]).exists() for x in table_rows)
    return {"status": "passed", "base": BASE, "prior_metric_rows_unchanged": 305,
            "added_metric_rows": len(rows), "total_metric_rows": len(final),
            "missing_metric_rows": len(missing), "protected_files_unchanged": protected,
            "exact_generated_files_checked": len(outputs), "source_counts": source_counts,
            "table_inventory_rows": len(table_rows), "new_inference": False,
            "new_bootstrap": False, "new_render": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source-copy-commands", action="store_true",
                        help="Emit exact cp commands for byte-preserving compact source copies")
    args = parser.parse_args()
    outputs, rows = build()
    if args.source_copy_commands:
        commands = []
        for bundle, label in ((CONTROLS, "r2e_controls"), (VARIABLE, "r2f_variable_accuracy")):
            for relative in json.loads(outputs[ROOT / "addenda" / label / "manifest.json"])["local_source_copies"]:
                destination = ROOT / "addenda" / label / "source_bundle" / relative
                commands.append(f"mkdir -p {shlex.quote(str(destination.parent))} && cp {shlex.quote(str(bundle / relative))} {shlex.quote(str(destination))}")
        print(json.dumps(commands))
    elif args.check:
        print(json.dumps(check(outputs, rows), indent=2))
    elif args.patch:
        print("*** Begin Patch")
        for path, data in outputs.items():
            if "source_bundle" in path.parts:
                continue  # Use --source-copy-commands; preserve source CRLF bytes.
            if path.exists():
                original = path.read_bytes()
                assert data.startswith(original), f"not append-only: {path}"
                print(f"*** Update File: {path}")
                print("@@")
                print(" " + original.decode().splitlines()[-1])
                for line in data[len(original):].decode().splitlines():
                    print("+" + line)
            else:
                print(f"*** Add File: {path}")
                for line in data.decode().splitlines():
                    print("+" + line)
        print("*** End Patch")
    else:
        parser.error("choose --patch or --check")


if __name__ == "__main__":
    main()
