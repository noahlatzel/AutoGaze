# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced human cell-mass supervision for exact-budget fine actions.

The teacher samples annotation mass without replacement. At each action, the
target is the human mass conditional on the *previous* actions in that frame.
If the annotation support is exhausted, uniform legal actions complete the
fixed-length history, but those rows contribute no distribution loss.
"""

from collections.abc import Mapping

import torch
import torch.nn.functional as F


_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


class HumanGazeNTP:
    """Soft-target cross-entropy, with forward KL reported separately."""

    uses_teacher_forcing = True

    def __init__(
        self,
        clip_len=16,
        actions_per_frame=265,
        fine_action_offset=69,
        exact_budget=16,
        teacher_seed=0,
    ):
        values = {
            "clip_len": clip_len,
            "actions_per_frame": actions_per_frame,
            "fine_action_offset": fine_action_offset,
            "exact_budget": exact_budget,
        }
        if any(type(value) is not int for value in values.values()):
            raise ValueError("Teacher dimensions must be integers")
        if clip_len <= 0 or actions_per_frame <= 0 or exact_budget <= 0:
            raise ValueError("Teacher dimensions must be positive")
        if not 0 <= fine_action_offset < actions_per_frame:
            raise ValueError("fine_action_offset must leave at least one fine action")
        self.clip_len = clip_len
        self.actions_per_frame = actions_per_frame
        self.fine_action_offset = fine_action_offset
        self.exact_budget = exact_budget
        self.num_fine_cells = actions_per_frame - fine_action_offset
        if exact_budget > self.num_fine_cells:
            raise ValueError("exact_budget exceeds the number of unique fine cells")
        if type(teacher_seed) is not int or not 0 <= teacher_seed < 2**63:
            raise ValueError("teacher_seed must be an integer in [0, 2**63)")
        self.teacher_seed = teacher_seed
        self.teacher_generator = torch.Generator(device="cpu").manual_seed(teacher_seed)

    def _contract(self):
        return {
            "clip_len": self.clip_len,
            "actions_per_frame": self.actions_per_frame,
            "fine_action_offset": self.fine_action_offset,
            "exact_budget": self.exact_budget,
        }

    def state_dict(self):
        """Return an independent CPU RNG snapshot for exact training resume."""
        return {
            "schema_version": 1,
            "contract": self._contract(),
            "teacher_seed": self.teacher_seed,
            "teacher_rng_state": self.teacher_generator.get_state().clone(),
        }

    def load_state_dict(self, state):
        if not isinstance(state, Mapping):
            raise ValueError("Teacher state must be a mapping")
        expected_keys = {"schema_version", "contract", "teacher_seed", "teacher_rng_state"}
        if set(state) != expected_keys or state["schema_version"] != 1:
            raise ValueError("Unsupported or malformed teacher state")
        if state["contract"] != self._contract():
            raise ValueError("Teacher state has a different action contract")
        seed = state["teacher_seed"]
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Invalid teacher seed in saved state")
        rng_state = state["teacher_rng_state"]
        if not isinstance(rng_state, torch.Tensor) or rng_state.dtype != torch.uint8 or rng_state.ndim != 1:
            raise ValueError("Teacher RNG state must be a one-dimensional uint8 tensor")
        replacement = torch.Generator(device="cpu")
        try:
            replacement.set_state(rng_state.detach().cpu())
        except RuntimeError as error:
            raise ValueError("Invalid teacher RNG state") from error
        self.teacher_seed = seed
        self.teacher_generator = replacement

    def _validate_mass(self, inputs):
        mass = inputs.get("cell_mass")
        if not isinstance(mass, torch.Tensor) or not mass.is_floating_point():
            raise ValueError("cell_mass must be a floating-point tensor")
        if mass.ndim != 3 or mass.shape[0] == 0 or tuple(mass.shape[1:]) != (self.clip_len, self.num_fine_cells):
            raise ValueError(f"cell_mass must have shape [B,{self.clip_len},{self.num_fine_cells}] with B > 0")
        if not torch.isfinite(mass).all() or (mass < 0).any():
            raise ValueError("cell_mass must be finite and nonnegative")
        # Float64 accumulation avoids accepting a positive row whose lower
        # precision sum overflowed, and permits tiny positive annotation mass.
        totals = mass.detach().to(torch.float64).sum(dim=-1)
        if not torch.isfinite(totals).all() or (totals <= 0).any():
            raise ValueError("Every initial frame must contain positive finite human mass")
        return mass

    def preprocess_inputs(self, inputs):
        mass = self._validate_mass(inputs)
        batch = mass.shape[0]
        weights = mass.detach().to(device="cpu", dtype=torch.float64).reshape(-1, self.num_fine_cells)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        available = torch.ones_like(weights, dtype=torch.bool)
        selected = []
        for _ in range(self.exact_budget):
            residual = weights * available
            exhausted = residual.sum(dim=-1, keepdim=True) == 0
            sampling_weights = torch.where(exhausted, available.to(weights.dtype), residual)
            action = torch.multinomial(sampling_weights, 1, generator=self.teacher_generator)
            selected.append(action.squeeze(-1))
            available.scatter_(1, action, False)
        fine = torch.stack(selected, dim=-1).reshape(batch, self.clip_len, self.exact_budget)
        offsets = torch.arange(self.clip_len).reshape(1, self.clip_len, 1) * self.actions_per_frame
        positions = (fine + self.fine_action_offset + offsets).reshape(batch, -1).to(mass.device)
        result = dict(inputs)
        result["gt_gazing_info"] = {
            "gazing_pos": positions,
            "num_gazing_each_frame": torch.full((self.clip_len,), self.exact_budget, dtype=torch.long, device=mass.device),
            "if_padded_gazing": torch.zeros_like(positions, dtype=torch.bool),
        }
        return result

    def _validate_trajectory(self, info, batch):
        if not isinstance(info, Mapping):
            raise ValueError("Teacher gaze information must be a mapping")
        positions = info.get("gazing_pos")
        shape = (batch, self.clip_len * self.exact_budget)
        if not isinstance(positions, torch.Tensor) or positions.dtype not in _INTEGER_DTYPES or tuple(positions.shape) != shape:
            raise ValueError(f"gazing_pos must be an integer tensor of shape {shape}")
        counts = info.get("num_gazing_each_frame")
        if not isinstance(counts, torch.Tensor) or counts.dtype not in _INTEGER_DTYPES or tuple(counts.shape) != (self.clip_len,) or not (counts == self.exact_budget).all():
            raise ValueError("num_gazing_each_frame must be a shared 1D exact-budget vector")
        padded = info.get("if_padded_gazing")
        if not isinstance(padded, torch.Tensor) or padded.dtype != torch.bool or tuple(padded.shape) != shape or padded.any():
            raise ValueError("Exact-budget human trajectories must have an all-false padding mask")
        offsets = torch.arange(self.clip_len, device=positions.device).reshape(1, self.clip_len, 1) * self.actions_per_frame
        fine = positions.to(torch.long).reshape(batch, self.clip_len, self.exact_budget) - offsets - self.fine_action_offset
        if (fine < 0).any() or (fine >= self.num_fine_cells).any():
            raise ValueError("Teacher trajectory contains a non-fine or wrong-frame action")
        ordered = fine.sort(dim=-1).values
        if (ordered[..., 1:] == ordered[..., :-1]).any():
            raise ValueError("Teacher fine actions must be unique within each frame")
        return fine

    def conditional_targets(self, inputs):
        """Return [B,T*K,C] targets, valid rows and the legal-action mask.

        This function uses the observed teacher prefix, not its current or future
        action, and starts a fresh availability mask at every frame boundary.
        """
        mass = self._validate_mass(inputs)
        fine = self._validate_trajectory(inputs.get("gt_gazing_info"), mass.shape[0]).to(mass.device)
        chosen = F.one_hot(fine, num_classes=self.num_fine_cells)
        previous = chosen.cumsum(dim=2) - chosen
        available = previous == 0
        residual = mass.detach().to(torch.float64).unsqueeze(2) * available
        totals = residual.sum(dim=-1, keepdim=True)
        valid = totals.squeeze(-1) > 0
        targets = residual / torch.where(totals > 0, totals, torch.ones_like(totals))
        return (
            targets.reshape(mass.shape[0], -1, self.num_fine_cells),
            valid.reshape(mass.shape[0], -1),
            available.reshape(mass.shape[0], -1, self.num_fine_cells),
        )

    def __call__(self, inputs, gaze_outputs, task_outputs=None):
        targets, valid, available = self.conditional_targets(inputs)
        teacher_fine = self._validate_trajectory(inputs["gt_gazing_info"], targets.shape[0])
        scored_fine = self._validate_trajectory(gaze_outputs, targets.shape[0])
        if not torch.equal(teacher_fine.detach().cpu(), scored_fine.detach().cpu()):
            raise ValueError("Scored gaze trajectory does not match the teacher history")
        log_probs = gaze_outputs.get("supervised_action_log_probs_all")
        if not isinstance(log_probs, torch.Tensor) or not log_probs.is_floating_point() or tuple(log_probs.shape) != tuple(targets.shape):
            raise ValueError("supervised_action_log_probs_all must contain the aligned normalized fine-action log distribution")
        available = available.to(log_probs.device)
        if not torch.isfinite(log_probs[available]).all() or not torch.isneginf(log_probs[~available]).all():
            raise ValueError("Policy log distribution must be finite exactly on the remaining legal fine actions")
        row_sums = log_probs.detach().to(torch.float64).exp().sum(dim=-1)
        tolerance = 1e-10 if log_probs.dtype == torch.float64 else (3e-5 if log_probs.dtype == torch.float32 else 2e-2)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=tolerance, rtol=0):
            raise ValueError("Policy log distribution is not normalized")
        targets = targets.to(device=log_probs.device, dtype=log_probs.dtype)
        valid = valid.to(log_probs.device)
        positive = targets > 0
        # Do not multiply zero targets by masked -inf values: this would create
        # NaNs and poison gradients even for an otherwise valid distribution.
        safe_log_probs = torch.where(positive, log_probs, torch.zeros_like(log_probs))
        row_ce = -(targets * safe_log_probs).sum(dim=-1)
        safe_target_logs = torch.where(positive, targets, torch.ones_like(targets)).log()
        row_entropy = -(targets * safe_target_logs).sum(dim=-1)
        counts = valid.sum(dim=-1)
        per_clip_ce = (row_ce * valid).sum(dim=-1) / counts
        per_clip_entropy = (row_entropy * valid).sum(dim=-1) / counts
        loss = per_clip_ce
        prediction = gaze_outputs.get("task_loss_prediction")
        if prediction is not None:
            # Historical decoder-only DDP includes this head in its parameter
            # set with find_unused_parameters=False. Keep its zero-gradient
            # connection without adding a task-prediction training objective.
            loss = loss + prediction.mean() * 0.0
        return {
            "loss": loss,
            "metrics": {
                "supervised_soft_ce": per_clip_ce.mean(),
                "supervised_forward_kl": (per_clip_ce - per_clip_entropy).mean(),
                "supervised_target_entropy": per_clip_entropy.mean(),
                "supervised_batch_valid_actions": valid.sum(),
                "supervised_batch_exhausted_actions": (~valid).sum(),
                "supervised_exhausted_fraction": (~valid).to(log_probs.dtype).mean(),
            },
            "per_sample_metrics": {
                "supervised_valid_actions": counts,
                "supervised_exhausted_actions": (~valid).sum(dim=-1),
            },
        }
