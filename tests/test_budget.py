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


def test_check_aborts_a_run_that_overruns_and_names_the_fix():
    msg = check(project(50, 2000, 500), budget_seconds=21600)
    assert msg is not None
    assert "batch" in msg.lower()
    assert "5h" in msg or "hours" in msg.lower()


def test_check_is_a_no_op_before_any_step_has_run():
    assert check(project(0, 0, 500), budget_seconds=21600) is None
