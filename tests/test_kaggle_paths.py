from __future__ import annotations

import pytest

from training.kaggle_paths import REQUIRED, find_code_dir


def _touch_modules(directory, names=REQUIRED, extra: tuple[str, ...] = ()):
    directory.mkdir(parents=True, exist_ok=True)
    for name in list(names) + list(extra):
        (directory / f"{name}.py").touch()
    return directory


def test_find_code_dir_locates_modules_nested_two_levels_deep(tmp_path):
    # Exactly what actually happened on Kaggle: the dataset did not land at
    # /kaggle/input/<slug>, it turned up under /kaggle/input/datasets/<slug>/.
    real = _touch_modules(tmp_path / "datasets" / "medical-ft-code")
    assert find_code_dir(tmp_path) == real


def test_find_code_dir_ignores_a_decoy_directory_with_partial_modules(tmp_path):
    # A directory that holds *some* of the modules (e.g. a stale or partial
    # upload, or an unrelated package) must not be mistaken for the real one.
    _touch_modules(tmp_path / "decoy", names={"records", "prompts", "sources"})
    real = _touch_modules(tmp_path / "medical-ft-code")
    assert find_code_dir(tmp_path) == real


def test_find_code_dir_raises_when_no_directory_has_every_module(tmp_path):
    # Every directory is missing at least one required module -- there is no
    # complete set anywhere under root.
    _touch_modules(tmp_path / "a", names={"records", "prompts"})
    _touch_modules(tmp_path / "b", names={"sources", "prepare_data"})
    with pytest.raises(SystemExit) as exc_info:
        find_code_dir(tmp_path)
    message = str(exc_info.value)
    assert "No directory under" in message
    assert "records" in message  # names the missing set, via REQUIRED


def test_find_code_dir_raises_when_root_does_not_exist(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(SystemExit) as exc_info:
        find_code_dir(missing)
    assert "does not exist" in str(exc_info.value)


def test_find_code_dir_picks_the_lexicographically_first_match_when_tied(tmp_path):
    # Two complete, independent copies is not a layout Kaggle should ever
    # produce, but the tie-break must still be deterministic rather than
    # depend on filesystem iteration order.
    first = _touch_modules(tmp_path / "aaa")
    _touch_modules(tmp_path / "zzz")
    assert find_code_dir(tmp_path) == first
