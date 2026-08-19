import pytest

from autogaze.tasks.human_heatmap_coverage import HumanHeatmapCoverage


@pytest.mark.parametrize(
    ("budget", "report_budgets"),
    [
        (24, (1, 2, 4, 8, 16, 24)),
        (36, (1, 2, 4, 8, 16, 24, 36)),
    ],
)
def test_human_coverage_task_accepts_fixed_budget_sweep(
    budget: int,
    report_budgets: tuple[int, ...],
) -> None:
    task = HumanHeatmapCoverage(
        exact_budget=budget,
        report_budgets=report_budgets,
    )
    assert task.exact_budget == budget
    assert task.report_budgets == report_budgets
    assert task.gaze_model_kwargs["max_gaze_tokens_each_frame"] == budget
    assert len(task.gaze_model_kwargs["allowed_token_ids"]) == 196


def test_human_coverage_task_rejects_report_budget_above_fixed_budget() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        HumanHeatmapCoverage(exact_budget=24, report_budgets=(16, 36))
