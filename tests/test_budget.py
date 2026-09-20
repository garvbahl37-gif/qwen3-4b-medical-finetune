from __future__ import annotations

from training.budget import check, project


def test_project_extrapolates_linearly_from_observed_steps():
    p = project(steps_done=50, seconds_elapsed=100, total_steps=500)
    assert p["seconds_per_step"] == 2.0
    assert p["projected_seconds"] == 1000
    assert p["remaining_seconds"] == 900


def test_project_refuses_to_divide_by_zero_steps():
    p = project(steps_done=0, seconds_elapsed=10, total_steps=100)
    assert p["seconds_per_step"] == 0.0
    assert p["projected_seconds"] == 0


def test_check_passes_a_run_that_fits():
    assert check(project(50, 100, 500), budget_seconds=21600) is None


def test_check_allows_a_run_that_lands_just_inside_the_budget():
    # 40s/step x 500 steps = 20,000s against a 21,600s budget. It fits, so
    # the probe must stay silent -- a false abort costs a queue wait and a
    # restart for nothing.
    assert check(project(50, 2000, 500), budget_seconds=21600) is None


def test_check_aborts_a_run_that_overruns_and_names_the_fix():
    # 60s/step x 500 steps = 30,000s = 8.3h against a 6.0h budget.
    msg = check(project(50, 3000, 500), budget_seconds=21600)
    assert msg is not None
    assert "8.3h" in msg          # the projection
    assert "6.0h" in msg          # the budget it is measured against
    assert "--batch-size" in msg  # the fix it names


def test_check_is_a_no_op_before_any_step_has_run():
    assert check(project(0, 0, 500), budget_seconds=21600) is None
