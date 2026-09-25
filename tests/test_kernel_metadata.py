from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


@pytest.mark.parametrize("path", ["training/kernel-metadata.json",
                                  "evaluation/kernel-metadata.json",
                                  "run2/kernel-metadata.json"])
def test_kernel_id_matches_the_slug_kaggle_derives_from_the_title(path):
    # Kaggle names a kernel by slugifying its title and ignores a mismatched
    # id. The first push then lands somewhere else and the second 409s.
    meta = json.loads(Path(path).read_text())
    _owner, slug = meta["id"].split("/")
    assert slug == slugify(meta["title"])


@pytest.mark.parametrize("path", ["training/kernel-metadata.json",
                                  "evaluation/kernel-metadata.json",
                                  "run2/kernel-metadata.json"])
def test_kernel_is_pinned_to_a_t4(path):
    # Without machine_shape a push most likely lands on a P100, which the
    # hardware-check cell then stops on at cell 1.
    meta = json.loads(Path(path).read_text())
    assert meta["machine_shape"] == "NvidiaTeslaT4"


def test_the_evaluation_kernel_attaches_code_and_adapter_on_a_private_gpu():
    meta = json.loads(Path("evaluation/kernel-metadata.json").read_text())
    assert set(meta["dataset_sources"]) == {"gb1105/medical-ft-code",
                                            "gb1105/medical-ft-adapter"}
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert meta["code_file"] == "kaggle_eval.ipynb"


def test_the_run2_kernel_attaches_code_and_data_on_a_private_gpu():
    meta = json.loads(Path("run2/kernel-metadata.json").read_text())
    assert meta["id"] == "gb1105/qwen3-4b-medical-fine-tune-run-2"
    assert meta["dataset_sources"] == ["gb1105/medical-ft-code", "gb1105/medical-ft-data"]
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert meta["code_file"] == "kaggle_run2.ipynb"
