from scripts.human_gaze.write_r6_run_manifests import manifest


def test_manifest_preserves_paired_seed_and_fixed_protocol():
    record = manifest(440826, 12345, 0, "abc123")
    assert record["training_seed"] == 640826
    assert record["fixed_updates"] == 10000
    assert record["validation_interval"] == 100
    assert record["paired_control_run"].endswith(
        "r3b_temporal_control_base440826_seed640826"
    )
    assert record["temporal_position_encoding"] == (
        "selection_conditioned_recurrent_state_logit_bias"
    )
    assert record["recurrent_update"] == "once_after_completed_frame"
