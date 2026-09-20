from __future__ import annotations


def project(steps_done: int, seconds_elapsed: int, total_steps: int) -> dict:
    """Extrapolate total training time from the steps observed so far."""
    if steps_done <= 0:
        return {"seconds_per_step": 0.0, "projected_seconds": 0,
                "remaining_seconds": 0}
    per_step = seconds_elapsed / steps_done
    projected = int(per_step * total_steps)
    return {
        "seconds_per_step": per_step,
        "projected_seconds": projected,
        "remaining_seconds": int(per_step * (total_steps - steps_done)),
    }


# The probe extrapolates from a 50-step sample -- noisy on its own -- and a
# real run still has to save the adapter/tokenizer and run a final eval pass,
# neither of which shows up in the step-timing loop it is measuring. Ten
# percent of headroom keeps the probe from waving through a run that has no
# slack left for either.
SAFETY_MARGIN = 0.9


def check(projection: dict, budget_seconds: int) -> str | None:
    """Return an abort message if the run will not fit, else None.

    The sibling text2sql notebook asks the operator to watch it/s and
    multiply out before walking away. Nobody does. Failing loudly at
    step 50 costs a minute; discovering it at hour six costs the day
    and a chunk of the weekly GPU quota.
    """
    projected = projection["projected_seconds"]
    if projected <= 0 or projected <= budget_seconds * SAFETY_MARGIN:
        return None
    return (
        f"\nSTOP. Projected training time is {projected / 3600:.1f} hours "
        f"against a budget of {budget_seconds / 3600:.1f} hours "
        f"({projection['seconds_per_step']:.2f}s/step).\n"
        f"FIX: halve --batch-size and double --grad-accum, which keeps the "
        f"effective batch identical, or cut the training set with the "
        f"--medmcqa / --chatdoctor flags on prepare_data.py."
    )
