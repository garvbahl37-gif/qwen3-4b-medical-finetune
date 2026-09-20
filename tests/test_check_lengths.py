from __future__ import annotations

from training.check_lengths import percentile, summarise


def test_percentile_picks_the_nearest_rank():
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 0.50) == 50
    assert percentile(values, 0.99) == 99
    assert percentile(values, 1.0) == 100


def test_percentile_handles_a_single_value():
    assert percentile([7], 0.99) == 7


def test_percentile_is_order_independent():
    assert percentile([5, 1, 3, 2, 4], 0.5) == percentile([1, 2, 3, 4, 5], 0.5)


def test_summarise_reports_the_fields_the_notebook_reads():
    stats = summarise([10, 20, 30, 40, 50])
    assert stats["n"] == 5
    assert stats["max"] == 50
    assert set(stats) == {"n", "median", "p90", "p99", "max"}
