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


# --- E3 ---------------------------------------------------------------------
from training.experiments_config import E3  # noqa: E402

E3A, E3B = REPO / "e3a/kaggle_e3a.ipynb", REPO / "e3b/kaggle_e3b.ipynb"
# How often each stand-in teacher reaches the gold letter; t30b fails to load.
TEACHER_SKILL = {"t14b": 0.8, "t8b": 0.3}


def completion(rec_id: str, letter: str) -> dict:
    return {"id": rec_id, "letter": letter, "closed": True, "forced": False,
            "think_tokens": 50, "new_tokens": 60,
            "text": f"<think>\nI reason about {rec_id}.\n</think>\n\nAnswer: {letter}\n\nBecause."}


def e3_popen_factory(work: Path, launched: list, weak_ids: list[str]):
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
        session = args[args.index("--session") + 1]
        gpu = int(args[args.index("--gpu") + 1])
        out = work / args[args.index("--out-dir") + 1]
        code = 0
        for entry in E3["sessions"][session][gpu]:
            model, stages = entry[0], entry[1]
            if model == "t30b":
                code = 1                                  # e.g. unsupported on a T4
                continue
            d = out / model
            d.mkdir(parents=True, exist_ok=True)
            seconds = {}
            for stage in stages:
                mode, bench = stage.split(":")
                if bench == "weak":
                    k = int(len(weak_ids) * TEACHER_SKILL[model])
                    rows = [completion(q, "A" if i < k else "B") for i, q in enumerate(weak_ids)]
                else:
                    rows = [{"id": f"{bench}-{i}", "letter": "A"} for i in range(2)]
                (d / f"{mode}__{bench}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                seconds[stage] = {"scored": len(rows), "seconds": 600}
            (d / "meta.json").write_text(json.dumps({"stages": seconds}))
        stdout.write(f"{session} gpu{gpu} done\n")
        stdout.flush()
        return FakeProc(code)
    return popen


def e3_namespace(tmp_path, monkeypatch, weak_ids):
    work, out = tmp_path / "ft", tmp_path / "working"
    out.mkdir()
    v3 = work / "data" / "v3"
    v3.mkdir(parents=True)
    for bench in BENCHES:
        recs = [Record(id=f"{bench}-{i}", source="x", kind="mcq", question=f"Q{i}",
                       options={"A": "a", "B": "b"}, answer="A", rationale=None,
                       response=None, subject="s") for i in range(2)]
        (v3 / f"eval_{bench}.jsonl").write_text("".join(json.dumps(r.to_dict()) + "\n" for r in recs))
    weak = [Record(id=q, source="medqa", kind="mcq", question=f"Which {q}?",
                   options={"A": "a", "B": "b"}, answer="A", rationale=None, response=None,
                   subject=None) for q in weak_ids]
    (v3 / "eval_weak.jsonl").write_text("".join(json.dumps(r.to_dict()) + "\n" for r in weak))
    (work / "outputs").mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    launched: list = []
    fake_subprocess = types.SimpleNamespace(
        Popen=e3_popen_factory(work, launched, weak_ids), run=real_subprocess.run,
        STDOUT=real_subprocess.STDOUT, TimeoutExpired=real_subprocess.TimeoutExpired)
    fake_time = types.SimpleNamespace(time=real_time.time, sleep=lambda s: None,
                                      strftime=real_time.strftime, localtime=real_time.localtime)

    def argv(cmd: str) -> list[str]:
        args = shlex.split(cmd)
        return [sys.executable] + args[1:] if args[0] == "python" else args

    def step(cmd: str, env=None):
        if real_subprocess.run(argv(cmd)).returncode != 0:
            raise SystemExit(f"step failed: {cmd}")

    import os
    import shutil
    ns = {"json": json, "os": os, "shlex": shlex, "shutil": shutil, "sys": sys, "Path": Path,
          "subprocess": fake_subprocess, "time": fake_time, "argv": argv, "step": step,
          "N_GPUS": 2, "SESSION_START": real_time.time(),
          "DEADLINE": real_time.time() + 11 * 3600, "ENGINE": "vllm",
          "VLLM_PY": "/tmp/vllm-env/bin/python", "VLLM_EAGER": "", "WORK": work}
    return ns, work, out, launched


def run_cells(nb_path: Path, headers: tuple[str, ...], ns: dict, out: Path) -> None:
    nb = json.loads(nb_path.read_text())
    for header in headers:
        src = cell(nb, header).replace('Path("/kaggle/working")', f'Path("{out}")')
        exec(compile(src, header, "exec"), ns)


@pytest.mark.skipif(not E3A.exists(), reason="build e3a first")
def test_e3a_pilots_teachers_then_keeps_the_winners_verified_traces(tmp_path, monkeypatch):
    weak_ids = [f"medqatrain-{i}" for i in range(10)]
    monkeypatch.setitem(E3["traces"], "min_rows", 1)           # ten stand-in questions
    ns, work, out, launched = e3_namespace(tmp_path, monkeypatch, weak_ids)
    run_cells(E3A, ("# --- 6. Phase runner", "# --- 7. Pilot", "# --- 8. The teacher"), ns, out)
    assert ns["TEACHER"] == "t14b"                                  # 80% beats 30%; t30b failed
    pilot = json.loads((out / "e3_pilot_report.json").read_text())
    assert set(pilot["stats"]) == {"t8b", "t14b"} and pilot["chosen"] == "t14b"
    traces = (out / "e3t" / "train.jsonl").read_text().splitlines()
    report = json.loads((out / "e3t" / "e2_report.json").read_text())
    assert report["questions_with_a_kept_trace"] == 8 and len(traces) == report["rows"]
    assert all(json.loads(t)["source"] == "e3_teacher" for t in traces)
    assert (out / "e3t" / "pilot_report.json").exists() and not work.exists()
    tp = next(env for args, env in launched if "pilot_tp" in args)
    assert tp["CUDA_VISIBLE_DEVICES"] == "0,1"                      # the MoE spans both GPUs


@pytest.mark.skipif(not E3B.exists(), reason="build e3b first (needs data/e3t)")
def test_e3b_merges_trains_and_scores_on_dev_only(tmp_path, monkeypatch):
    ns, work, out, launched = e3_namespace(tmp_path, monkeypatch, ["medqatrain-0"])
    rec = Record(id="x", source="medqa", kind="mcq", question="Q", options={"A": "a", "B": "b"},
                 answer="A", rationale=None, response=None, subject=None, reasoning="r")
    for name, n in (("e2", 3), ("e3t", 2)):
        d = work / "data" / name
        d.mkdir(parents=True)
        (d / "train.jsonl").write_text("".join(json.dumps({**rec.to_dict(), "id": f"{name}{i}"}) + "\n"
                                               for i in range(n)))
        (d / "e2_report.json").write_text(json.dumps({"max_seq": 512, "rows": n}))
    run_cells(E3B, ("# --- 6. Phase runner", "# --- 9. Train on GPU 0"), ns, out)
    train = next(args for args, env in launched if "training.train" in args)
    assert train[train.index("--data") + 1] == "data/e3"
    report = json.loads((out / "e3_report.json").read_text())
    assert set(report["models"]) == {"base", "e3"}
    assert set(report["stages"]) == {"reasoning:dev_medqa", "letter:dev_medqa",
                                     "reasoning:dev_medmcqa", "letter:dev_medmcqa"}
    assert (out / "e3-adapter" / "adapter_model.safetensors").exists() and not work.exists()
