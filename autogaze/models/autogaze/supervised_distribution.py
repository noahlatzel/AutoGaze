"""Stable fine-action probabilities for supplied human-gaze training histories."""

import torch
import torch.nn.functional as F


def supervised_action_log_probs(
    logits_multi_token_pred,
    gaze_token_mask,
    gaze_pred_source_relative,
    gaze_pos_ids_split,
    allowed_token_ids,
):
    """Score the existing prediction heads, conditional on no repeated action.

    This is deliberately separate from legacy probability-space rescoring. The
    result uses local fine-cell order (action IDs 69..264), float32 arithmetic,
    and exact negative infinity for cells selected earlier in the same frame.
    """
    if list(allowed_token_ids) != list(range(69, 265)):
        raise ValueError("Supervised distributions require fine action IDs 69..264")
    if logits_multi_token_pred.ndim != 4:
        raise ValueError("Expected logits with shape [batch, position, head, vocabulary]")
    device = logits_multi_token_pred.device
    gaze_positions = torch.nonzero(gaze_token_mask, as_tuple=True)[0].to(device)
    relative = gaze_pred_source_relative.to(device)[gaze_positions]
    sources = gaze_positions + relative
    heads = -relative - 1
    if (
        (sources < 0).any()
        or (sources >= gaze_positions).any()
        or (heads < 0).any()
        or (heads >= logits_multi_token_pred.shape[2]).any()
    ):
        raise ValueError("Invalid multi-token prediction source or head")
    selected_logits = logits_multi_token_pred[:, sources, heads, 69:265].float()
    if selected_logits.shape[-1] != 196 or not torch.isfinite(selected_logits).all():
        raise ValueError("Supervised fine-action logits must be finite and contain 196 cells")
    if sum(frame_ids.shape[1] for frame_ids in gaze_pos_ids_split) != len(gaze_positions):
        raise ValueError("Teacher frame lengths do not match prediction positions")

    available = torch.ones_like(selected_logits, dtype=torch.bool)
    offset = 0
    for frame_ids in gaze_pos_ids_split:
        if frame_ids.ndim != 2 or frame_ids.shape[0] != selected_logits.shape[0]:
            raise ValueError("Teacher frames must have shape [batch, actions]")
        if frame_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Teacher action IDs must be integer tensors")
        if frame_ids.shape[1] == 0 or frame_ids.shape[1] > 196:
            raise ValueError("Teacher frames must contain 1..196 fine actions")
        fine_cells = frame_ids.to(device=device, dtype=torch.long) - 69
        if (fine_cells < 0).any() or (fine_cells >= 196).any():
            raise ValueError("Teacher actions must use fine action IDs 69..264")
        sorted_cells = fine_cells.sort(dim=1).values
        if (sorted_cells[:, 1:] == sorted_cells[:, :-1]).any():
            raise ValueError("Teacher actions must not repeat within a frame")
        for step in range(1, frame_ids.shape[1]):
            available[:, offset + step].scatter_(1, fine_cells[:, :step], False)
        offset += frame_ids.shape[1]

    # This is log-space conditioning on availability. Mask before normalizing
    # to avoid subtracting two large log normalizers (and gradient cancellation)
    # when a previously selected cell dominates the unconditioned distribution.
    return F.log_softmax(selected_logits.masked_fill(~available, -float("inf")), dim=-1)
