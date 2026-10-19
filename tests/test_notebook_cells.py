"""Run the e2t notebook's orchestration cells for real, with the GPU work
stood in for.

Kaggle sessions twice died in notebook glue the module tests never touched
(an offline tokenizer lookup, then a log opened in a folder that did not
exist). This executes the generated cells themselves: the phase runner and
the train-and-evaluate cell, in a scratch working directory. Only the
processes that need a GPU (training and the evaluation workers) are replaced,
by stand-ins that write the files the real ones write. eval_report runs for
real on those files.
"""
from __future__ import annotations

import json
import shlex
import subprocess as real_subprocess
import sys
import time as real_time
import types
from pathlib import Path

import pytest

from training.experiments_config import E2
from training.records import Record

NOTEBOOK = Path("e2t/kaggle_e2t.ipynb")
REPO = Path(__file__).resolve().parent.parent
BENCHES = ("dev_medqa", "dev_medmcqa", "medqa", "medmcqa", "pubmedqa", "mmlu_medical")


def cell(nb: dict, header: str) -> str:
    for c in nb["cells"]:
        src = "".join(c["source"])
        if src.startswith(header):
            return src
    raise AssertionError(f"no cell starting with {header!r}")


class FakeProc:
    def __init__(self, returncode: int = 0):
        self.returncode = returncode
        self.pid = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def fake_popen_factory(work: Path, launched: list):
    def popen(cmd, stdout=None, stderr=None, env=None, start_new_session=False):
        args = list(cmd)
        launched.append((args, dict(env or {})))
        if "training.train" in args:
            out = work / args[args.index("--out") + 1]
            out.mkdir(parents=True, exist_ok=True)
            (out / "adapter_config.json").write_text('{"r": 16}')
            (out / "adapter_model.safetensors").write_text("x")
            stdout.write("training done\n")
            stdout.flush()
            return FakeProc()
        assert "training.run_plan" in args, args
        session = args[args.index("--session") + 1]
        gpu = int(args[args.index("--gpu") + 1])
        out = work / args[args.index("--out-dir") + 1]
        for entry in E2["sessions"][session][gpu]:
            model, stages = entry[0], entry[1]
            d = out / model
            d.mkdir(parents=True, exist_ok=True)
            (d / "meta.json").write_text("{}")
            for stage in stages:
                mode, bench = stage.split(":")
                rows = [{"id": f"{bench}-{i}", "letter": "A"} for i in range(2)]
                (d / f"{mode}__{bench}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows))
        stdout.write(f"{session} gpu{gpu} done\n")
        stdout.flush()
        return FakeProc()
    return popen


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="build e2t first")
def test_e2t_orchestration_cells_run_end_to_end(tmp_path, monkeypatch):
    nb = json.loads(NOTEBOOK.read_text())
    work, out = tmp_path / "ft", tmp_path / "working"
    out.mkdir()
    for bench in BENCHES:
        d = work / "data" / "v3"
        d.mkdir(parents=True, exist_ok=True)
        recs = [Record(id=f"{bench}-{i}", source="x", kind="mcq", question=f"Q{i}",
                       options={"A": "a", "B": "b"}, answer="A", rationale=None,
                       response=None, subject="s") for i in range(2)]
        (d / f"eval_{bench}.jsonl").write_text("".join(json.dumps(r.to_dict()) + "\n" for r in recs))
    (work / "data" / "e2").mkdir(parents=True)
    (work / "data" / "e2" / "e2_report.json").write_text(json.dumps({"max_seq": 512, "rows": 3}))
    (work / "outputs").mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    launched: list = []
    fake_subprocess = types.SimpleNamespace(Popen=fake_popen_factory(work, launched),
                                            run=real_subprocess.run, STDOUT=real_subprocess.STDOUT,
                                            TimeoutExpired=real_subprocess.TimeoutExpired)
    fake_time = types.SimpleNamespace(time=real_time.time, sleep=lambda s: None,
                                      strftime=real_time.strftime, localtime=real_time.localtime)

    def argv(cmd: str) -> list[str]:
        args = shlex.split(cmd)
        return [sys.executable] + args[1:] if args[0] == "python" else args

    def step(cmd: str, env=None):
        p = real_subprocess.run(argv(cmd), env=None)
        if p.returncode != 0:
            raise SystemExit(f"step failed: {cmd}")

    import os
    import shutil
    ns = {"json": json, "os": os, "shlex": shlex, "shutil": shutil, "sys": sys, "Path": Path,
          "subprocess": fake_subprocess, "time": fake_time, "argv": argv, "step": step,
          "N_GPUS": 2, "SESSION_START": real_time.time(),
          "DEADLINE": real_time.time() + 11 * 3600, "ENGINE": "vllm",
          "VLLM_PY": "/tmp/vllm-env/bin/python", "VLLM_EAGER": "", "WORK": work}
    for header in ("# --- 6. Phase runner", "# --- 9. Train on GPU 0"):
        src = cell(nb, header).replace('Path("/kaggle/working")', f'Path("{out}")')
        exec(compile(src, header, "exec"), ns)

    report = json.loads((out / "e2_report.json").read_text())
    assert {"base", "e2"} <= set(report["models"])
    assert "e2" in report["stages"]["reasoning:dev_medqa"]["vs_baseline"]
    assert (out / "e2-adapter" / "adapter_model.safetensors").exists()
    assert (out / "e2_per_question.jsonl").exists() and not work.exists()
    train_env = next(env for args, env in launched if "training.train" in args)
    assert "HF_HUB_OFFLINE" not in train_env          # training may reach the Hub
    worker_envs = [env for args, env in launched if "training.run_plan" in args]
    assert worker_envs and all(env.get("HF_HUB_OFFLINE") == "1" for env in worker_envs)
