# Supervised K16 real-model preflight

Status: **passed**. This is a non-scientific implementation and resource smoke
test, not a training seed or an evaluation result.

The run loaded the shared original AutoGaze checkpoint, accepted only the known
historical absent `gazing_model.gaze_decoder.output_token_logit_bias` buffer
after verifying that its runtime tensor was persistent, floating, finite,
vocabulary-shaped (`[266]`) and exactly zero, and exercised one deterministic
TRAIN clip from each of the six AV-gaze-STAViS sources.

Six consecutive microbatch-one forward/backward iterations succeeded through
single-rank NCCL DDP with `find_unused_parameters=False`. Every supervised loss
and trainable gradient was finite, the vision model and connector stayed frozen,
and the only zero-gradient trainable tensor was the deliberately zero-connected
task-loss prediction head. Every one of the 1,536 teacher action rows had legal,
normalized support over the remaining 196-cell fine vocabulary. A separate pass
gave the model only `video` and returned unique greedy exact-K16 fine cells for
all six clips.

On the NVIDIA RTX A2000 12GB VM, peak CUDA allocation was 407,873,536 bytes,
peak CUDA reservation was 450,887,680 bytes, and peak process RSS was
2,149,191,680 bytes. The measured in-script duration was 17.672 seconds. These
figures establish that the requested microbatch-one preflight fits the VM; they
are not a full-training throughput or memory estimate.

The immutable scientific implementation base is
`d85c6558bf9e8f02ece1f3516b4709a715ebec63`. The receipt-producing harness ran
at `21e9ab8415be7f3515b6ed75539cf12060d42ecc`. See `config.yaml` for the exact
invocation, `metrics.json` for compact checks, and `manifest.json` for hashes and
the ignored full receipt location.

