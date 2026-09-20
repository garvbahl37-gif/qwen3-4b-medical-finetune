from __future__ import annotations

from pathlib import Path

# `scripts/build_notebook.py` reads everything below the marker line and
# inlines it verbatim into the Kaggle notebook's code-fetch cell. That cell
# cannot `import training.kaggle_paths` -- the whole point of `find_code_dir`
# is to locate the code before it has been copied anywhere importable, so
# there is no package to import from yet. Keeping the real implementation
# here (rather than only inline in the notebook) is what makes it testable:
# tests/test_kaggle_paths.py imports this module directly.
#
# Everything above this line (the imports) must already be satisfied by the
# notebook cell it gets inlined into; everything below must not depend on
# anything the cell does not already provide.
# --- inline below ---
REQUIRED = {"records", "prompts", "sources", "prepare_data",
            "check_lengths", "budget", "train"}


def find_code_dir(root: Path) -> Path:
    """Locate the uploaded modules wherever Kaggle mounted them.

    The mount path is not stable: a dataset declared as gb1105/medical-ft-code
    turned up under /kaggle/input/datasets/... rather than at
    /kaggle/input/medical-ft-code. Two runs died on that assumption, so this
    searches for the directory that actually holds the modules instead.
    """
    if not root.exists():
        raise SystemExit(
            "\nSTOP. /kaggle/input does not exist -- no inputs are attached.\n"
            "FIX: sidebar -> + Add Input -> Datasets -> medical-ft-code.")
    candidates = []
    for path in root.rglob("*.py"):
        stems = {p.stem for p in path.parent.glob("*.py")}
        if REQUIRED <= stems:
            candidates.append(path.parent)
    if not candidates:
        found = sorted(str(p.relative_to(root)) for p in root.rglob("*.py"))[:20]
        tree = sorted(str(p.relative_to(root)) for p in root.rglob("*"))[:30]
        raise SystemExit(
            f"\nSTOP. No directory under {root} contains all of {sorted(REQUIRED)}.\n"
            f".py files found: {found or 'none'}\n"
            f"First entries under /kaggle/input: {tree}\n"
            "FIX: re-run scripts/push_kaggle.sh, then confirm the "
            "medical-ft-code dataset is attached in the sidebar.")
    return sorted(set(candidates))[0]
