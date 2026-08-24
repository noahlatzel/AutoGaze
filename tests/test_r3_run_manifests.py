from scripts.human_gaze.write_r3_run_manifests import manifest


def test_paired_manifest_changes_only_arm_and_treatment():
    control = manifest("control", 440826, 1680265, "abc123")
    treatment = manifest("position", 440826, 1680265, "abc123")
    ignored = {"run_id", "arm", "slurm_array_task_id", "temporal_position_encoding"}
    assert {key: value for key, value in control.items() if key not in ignored} == {
        key: value for key, value in treatment.items() if key not in ignored
    }
    assert control["training_seed"] == treatment["training_seed"] == 640826
    assert control["source_checkpoint"] == treatment["source_checkpoint"]
