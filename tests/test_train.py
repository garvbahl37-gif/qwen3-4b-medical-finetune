from __future__ import annotations

import pytest

from training.train import pick_kwarg


def test_pick_kwarg_picks_the_old_name_when_that_is_all_the_signature_takes():
    def old_sft_trainer_init(self, model=None, *, tokenizer=None):
        pass

    assert pick_kwarg(old_sft_trainer_init, "processing_class", "tokenizer") == "tokenizer"


def test_pick_kwarg_picks_the_new_name_when_that_is_all_the_signature_takes():
    def new_sft_trainer_init(self, model=None, *, processing_class=None):
        pass

    assert pick_kwarg(new_sft_trainer_init, "processing_class", "tokenizer") == "processing_class"


def test_pick_kwarg_prefers_the_first_candidate_when_both_are_accepted():
    def both_init(self, *, processing_class=None, tokenizer=None):
        pass

    assert pick_kwarg(both_init, "processing_class", "tokenizer") == "processing_class"
    assert pick_kwarg(both_init, "tokenizer", "processing_class") == "tokenizer"


def test_pick_kwarg_falls_back_to_the_first_candidate_under_a_kwargs_catchall():
    def catchall_init(self, *args, **kwargs):
        pass

    assert pick_kwarg(catchall_init, "processing_class", "tokenizer") == "processing_class"


def test_pick_kwarg_raises_a_clear_error_when_neither_name_is_accepted():
    def unrelated_init(self, *, model=None):
        pass

    with pytest.raises(TypeError) as exc_info:
        pick_kwarg(unrelated_init, "processing_class", "tokenizer")

    message = str(exc_info.value)
    assert "processing_class" in message
    assert "tokenizer" in message
