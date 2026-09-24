from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


@pytest.mark.parametrize("path", ["training/kernel-metadata.json",
                                  "evaluation/kernel-metadata.json"])
def test_kernel_id_matches_the_slug_kaggle_derives_from_the_title(path):
    # Kaggle names a kernel by slugifying its title and ignores a mismatched
    # id. The first push then lands somewhere else and the second 409s.
    meta = json.loads(Path(path).read_text())
    _owner, slug = meta["id"].split("/")
    assert slug == slugify(meta["title"])


def test_the_evaluation_kernel_attaches_code_and_adapter_on_a_private_gpu():
    meta = json.loads(Path("evaluation/kernel-metadata.json").read_text())
    assert set(meta["dataset_sources"]) == {"gb1105/medical-ft-code",
                                            "gb1105/medical-ft-adapter"}
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert meta["code_file"] == "kaggle_eval.ipynb"
