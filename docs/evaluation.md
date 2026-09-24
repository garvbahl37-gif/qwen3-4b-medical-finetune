# Evaluation

The evaluation compares the run-1 adapter with the base model on 5,456 exam
questions that training never saw. It is built and has been tested locally,
but it has not run on Kaggle, so **there are no results yet**. This page
describes what it measures and how, so that the result can be checked when it
arrives.

The code is `training/evaluate.py` (scoring and the report),
`training/evalcore.py` (statistics), `training/modeling.py` (loading) and
`training/prompts.py` (prompts and answer extraction). The Kaggle notebook is
`evaluation/kaggle_eval.ipynb`.

## What is measured

| Benchmark | Dataset and split | Loaded with | Questions |
|---|---|---|---|
| MedQA-USMLE test | [`GBaker/MedQA-USMLE-4-options`](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options), `test` | the `load()` defaults | 1,273 |
| MedMCQA validation | [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa), `validation` | `require_rationale=False`, `require_single_choice=False` | 4,183 |

These are the sets that training was decontaminated against
([training.md](training.md#decontamination-and-the-held-out-sets)). The
notebook builds them with the same `sources.load()` calls and stops if either
count differs from `EXPECTED_HOLDOUT_SIZES` in `evaluate.py`, which would mean
the dataset on the Hugging Face Hub has changed. `evaluate.py` checks the
counts again in the file it is given.

Two models answer every question:

- the base: [`unsloth/Qwen3-4B`](https://huggingface.co/unsloth/Qwen3-4B) at
  full precision, with the adapter switched off;
- the fine-tune: the same weights with the run-1 LoRA adapter on.

The comparison is paired. Both models see the same questions, the same prompts
and the same greedy decoding, so each question gives one paired outcome.

The patient-dialogue half of the training mix is not scored. The
[design spec](superpowers/specs/2026-09-20-medical-llm-finetune-design.md)
rules out an LLM-judge score for chat quality and plans a hand-written set of
patient questions, read and reported as prose; that set is not in the
repository yet.

## Loading without Unsloth

`modeling.load_model()` loads the base with plain transformers
(`AutoModelForCausalLM`, `AutoTokenizer`) and wraps it with `peft.PeftModel`.
The evaluation never imports Unsloth, and `tests/test_evaluate.py` fails if
`evaluate.py` or `modeling.py` imports it. The
[evaluation plan](superpowers/plans/2026-09-24-evaluation.md) gives two
reasons: Unsloth's patching of TRL caused three failed training runs, and
Unsloth cannot run on the Mac where this code is tested before it reaches a GPU.

The adapter was trained on the 4-bit `unsloth/Qwen3-4B-unsloth-bnb-4bit`, and
the evaluation scores it on the full-precision `unsloth/Qwen3-4B`. The comment
in `modeling.py` gives the reasoning: a LoRA adapter's deltas apply to the
unquantised base unchanged, and the full-precision model is the one that would
be served.

The device and precision come from `modeling.choose_device()`:

| Device | Precision | Reason given in `modeling.py` |
|---|---|---|
| `cuda` | float16 | the T4 has no native bfloat16 |
| `mps` | bfloat16 | Apple silicon supports it, and it is Qwen3's native precision |
| `cpu` | float32 | half precision on CPU is slow where it works at all |

Without `--device`, it takes CUDA if available, then MPS, then the CPU. A
device that is unknown or unavailable stops the run. The tokenizer pads on the
left, and uses the EOS token as the pad token if it has none.

On Kaggle, the notebook pins `transformers==5.5.0` and `peft==0.19.1`. That is
the pair the training session installed when it produced this adapter (run 1's
output shows transformers 5.5.0 and peft 0.19.1, and the adapter config was
written by peft 0.19.1). An older transformers could ignore the `dtype=`
argument that `modeling.py` passes and load the 4B weights in float32, about
16 GB, onto a 15 GB T4.

## One load, two models

`evaluate.py` loads the model once and scores it twice. The base is scored
inside `model.disable_adapter()`, then the fine-tune with the adapter on. Each
pass runs constrained scoring on every question and generative scoring on the
first `--gen-limit`. Because the base is the same weights with the adapter
switched off, both models share identical base weights, and the T4 never holds
a second copy of the 4B model.

The local smoke test checks this on `Qwen/Qwen3-0.6B` with a random adapter:
the disabled adapter scores exactly like a freshly loaded base. Loaded locally
on the full-precision base, the run-1 adapter itself attaches: [`TODO.md`](../TODO.md)
records 504 of 504 weights loaded, 252 of 252 LoRA matrices non-zero, and that
the adapter changes the model's answers.

## Prompt parity with training

Scoring prompts come from the same `build_messages()` as training: the
`SYSTEM_MCQ` system prompt, then the question with options `A.` to `D.`, with
no answer. `prompts.render_chat()` renders them with

```python
tok.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
```

Every training example put its answer after an empty think block,
`<think>\n\n</think>\n\n`. Qwen3's chat template emits that block at inference
only when `enable_thinking=False`. At the template's default the model starts
in thinking mode instead: the fine-tune would be scored on a format it never
saw, and the base would spend its token budget thinking. `TODO.md` records
that an earlier scoring prompt left Qwen3 in thinking mode, and that writing
the evaluation plan caught it.

For constrained scoring, `render_inference_prompt(..., answer_prefix=True)`
appends `Answer:`, so the next token is the choice itself. In every training
target the token after `Answer:` is ` A`, ` B`, ` C` or ` D`. The unit tests
use a stand-in tokenizer, so `scripts/smoke_eval.py` checks with Qwen3's own
tokenizer that the template renders a prompt ending in
`<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:`.

## Constrained scoring

Constrained scoring runs on all 5,456 questions. The design spec makes it the
headline number: one forward pass per batch, no decoding, deterministic.

For each batch of `--batch-size` questions (default 16), `score_constrained()`:

1. renders the prompts with `Answer:` appended and tokenizes them with left
   padding;
2. derives position ids from the attention mask, so a left-padded prompt sees
   the same positions it would see alone;
3. runs one forward pass with `logits_to_keep=1`, which materialises only the
   last position's logits (the full vocabulary at every position would be
   4 GB per batch of 16 on a T4), and with `use_cache=False`, since nothing
   reads the KV cache back (this frees about 2.2 GiB at the widest batch on a
   T4);
4. compares the logits of the four letter tokens and takes the highest, with
   ties going to the earliest letter.

The letter tokens are ` A` to ` D`, encoded with a leading space as they
appear after `Answer:`, taking the last token of each encoding. The space
matters: on the Qwen3 tokenizer ` A` is token 362 and `A` is token 32
([`TODO.md`](../TODO.md)).

Two guards stop the run instead of scoring noise:

- Non-finite letter logits. A float16 overflow on the T4 turns logits into inf
  or NaN, and an argmax over them would return A for every question: an
  accuracy that looks plausible and measures nothing. The message says that
  half precision overflowed on this GPU, that there is no T4 fallback, and
  that the result must not be reported. `--device cpu` runs in float32 and is
  a local diagnostic only.
- A right-padding tokenizer. The last position of every shorter sequence would
  be a pad token.

Constrained scoring always yields one of the four letters, so it has no
unparseable answers. The smoke test checks that batch size 1 and batch size 4
give identical predictions; `TODO.md` notes the test batches carried up to 18
tokens of padding.

## Generative scoring

Generative scoring runs on the first `--gen-limit` questions of each test file:
300 in the notebook. The notebook writes the files in `sources.load()` order,
which is a seed-42 shuffle of the split, so these are the first 300 of that
shuffle, not of the published file.

`score_generative()` renders the prompt without `Answer:` and calls
`model.generate()` with `do_sample=False` and `temperature`, `top_p` and
`top_k` unset, which is greedy decoding, up to `--max-new-tokens` (default
512). With left padding every prompt in a batch ends at the same column, so
the completion is everything after it, decoded without special tokens. The
answer is whatever `extract_letter()` finds in the completion, or nothing.

A completion hit the cap when none of its generated tokens is an
end-of-sequence id. The end-of-sequence ids are every id in
`model.generation_config.eos_token_id` (an int or a list) plus the tokenizer's
`eos_token_id`. The report counts capped completions per model and marks each
completion, so an answer lost to the token limit can be told apart from a
wrong one.

The spec runs both modes because they can disagree: constrained accuracy alone
can hide a model that knows the answer and cannot state it, which is the
failure that would break a chatbot.

## Answer extraction

`prompts.extract_letter()` is lenient about format and strict about evidence:

1. It looks for a cue word (`answer`, `answers`, `option`, `options`,
   `choice`, `choose`, `select`, `selected`, `correct`, `best`) within 28
   characters of a standalone letter A to D on the same line, in either
   order: "the correct option is D" and "C is the correct choice" both count.
2. It skips a letter when the 24 characters after it reject it: `not`, `n't`,
   `never`, `wrong`, `incorrect`, `excluded`, `ruled out` or `unlikely`,
   optionally after a verb such as "is". Elimination reasoning ("Option A is
   wrong. The correct answer is C.") then reads as C, not A.
3. Of the matches left, the one that ends last wins, because a model that
   reconsiders states its conclusion last.
4. If nothing matched, it accepts the final non-empty line when that line is a
   bare letter (`C`, `(C)`, `C.`) or opens with a choice marker (`C) Vitamin D`).
5. Otherwise the answer is unparseable.

Anchoring to cue words matters for medical text. An extractor that takes the
first A to D anywhere reads "A 45-year-old man ... choice C." as A, and
vignettes open that way. The reverse direction (letter, then cue) is there
because the base model's phrasing varies most, and missing its answers would
understate the base and flatter the fine-tune. In the aggregate, a letter
guessed from prose that states no answer looks the same as one the model
stated, while an unparseable answer costs both models equally.

An unparseable answer counts as wrong, not excluded. Dropping them, the
`evalcore.py` docstring notes, would flatter whichever model rambles more,
which is usually the base.

## Statistics

`training/evalcore.py` uses only the standard library.

Accuracy is correct answers over all questions. Each question then falls into
one cell:

| | fine-tune right | fine-tune wrong |
|---|---|---|
| base right | `both_correct` | `regressions` |
| base wrong | `wins` | `both_wrong` |

The exact McNemar test uses only the discordant pairs. With
n = wins + regressions and k = min(wins, regressions), the two-sided p-value is

```text
p = min(1, 2 * (C(n, 0) + C(n, 1) + ... + C(n, k)) / 2^n)
```

and p is 1 when n is 0. It is rounded to six decimals. Two worked examples:
the 16-question local check had 0 wins and 3 regressions, so n = 3 and
p = 2 × 1/8 = 0.25; the
[training plan](superpowers/plans/2026-09-20-training-and-evaluation.md)
checks that 10 wins against 2 regressions gives p = 0.038574.

The per-subject breakdown groups questions by the record's `subject`: MedMCQA's
`subject_name`, or for MedQA the dataset's `meta_info` value. Questions without
one go under `unknown`. Each subject gets its question count and both models'
accuracy, and subjects are sorted by count, largest first. The spec asks for
this breakdown as the place where dilution would show.

The results cell in the notebook prints "not significant at p<0.05:
indistinguishable from base" for any comparison with a p-value of 0.05 or
more. Four
comparisons are reported (two benchmarks, two modes), each with its own
p-value and no correction for multiple comparisons.

## The report JSON

`evaluate.py` writes one JSON file per test file. The notebook's are
`eval_medqa.json` and `eval_medmcqa.json`.

| Key | Contents |
|---|---|
| `test_set` | the path of the test file |
| `base` | the base weights, `unsloth/Qwen3-4B` unless `--base` says otherwise |
| `adapter` | the adapter directory |
| `adapter_sha256` | the SHA-256 of `adapter_model.safetensors` in that directory, so the report names the exact adapter it scored |
| `device` | `cuda`, `mps` or `cpu` |
| `timing` | `load_s`, plus `constrained_s_per_example` and `generative_s_per_example`, each per model |
| `reports` | `constrained` and `generative`, one paired report each |
| `samples` | the first 12 generatively scored questions: `id`, `answer`, and each model's full completion |
| `predictions` | one entry per scored question, in order |
| `generation` | `max_new_tokens`, and `hit_cap` with the number of capped completions for `base` and `tuned` |
| `environment` | the `torch`, `transformers` and `peft` versions, and `device_name`: the CUDA device's name, or the device string elsewhere |

Each paired report holds `mode`, `n`, `base_accuracy`, `tuned_accuracy`,
`base_unparseable`, `tuned_unparseable`, `mcnemar` (`wins`, `regressions`,
`both_correct`, `both_wrong`, `discordant`, `p_value`) and `by_subject`
(for each subject: `n`, `base`, `tuned`).

Each `predictions` entry has this shape:

```text
{
  "id": str, "answer": "A" | "B" | "C" | "D", "subject": str or null,
  "constrained": {"base": letter, "tuned": letter},
  "generative": {
    "base": letter or null, "tuned": letter or null,
    "base_text": str, "tuned_text": str,
    "base_hit_cap": bool, "tuned_hit_cap": bool
  }
}
```

`generative` appears only on the questions that were also scored
generatively. With every prediction saved, the one Kaggle session keeps the
evidence for why a result happened: for example, whether a MedMCQA gain comes
from answering A more often.

## The Kaggle evaluation notebook

`scripts/build_notebook.py` generates `evaluation/kaggle_eval.ipynb`. Its
metadata, `evaluation/kernel-metadata.json`, makes the kernel private with a
GPU and internet access, pins the machine to a T4 (`"machine_shape":
"NvidiaTeslaT4"`), and attaches two datasets: `medical-ft-code` and
`medical-ft-adapter`. [`docs/kaggle.md`](kaggle.md) covers the uploads and the
push.

The notebook runs in one session. A smoke pass goes through the exact code the
full run uses, measures its speed, and stops the session if the full run would
not fit. Only then does the full run start.

| Cell | What it does | It stops the session when |
|---|---|---|
| 1. Hardware check | sets `CUDA_VISIBLE_DEVICES=0` before importing torch, and prints the GPU, its compute capability and the torch version | compute capability is below 7, as on a P100 (6.0). The message says the metadata pins a T4, and to set the accelerator to GPU T4 x2 in the sidebar if a P100 arrived anyway |
| install | installs `transformers==5.5.0`, then `peft==0.19.1` with `--no-deps`, then `datasets`, and uninstalls the image's `torchao`, which peft 0.19.1 rejects below 0.16.0 | |
| 2. Verify the install | prints the torch, transformers and peft versions, then wraps one tiny layer with LoRA the way loading the adapter will | transformers is not 5.5.0 or peft is not 0.19.1: the install did not take effect on this kernel image, so restart the session and Run All again. Or peft cannot wrap the tiny layer: the message names the optional package to uninstall in the install cell |
| 3. Get the code and the adapter | finds the code and the adapter wherever Kaggle mounted them, and copies the modules into `/kaggle/working/ft/training` | no directory holds every evaluation module; the code's fingerprint differs from the one built into the notebook; no directory holds `adapter_config.json` next to `adapter_model.safetensors` |
| 4. Build the held-out sets | loads both benchmarks as training did, and writes `data/holdout_medqa.jsonl` and `data/holdout_medmcqa.jsonl` | a count differs from 1,273 or 4,183 |
| 5. Smoke pass | runs `evaluate.py` on 32 questions per benchmark, 16 of them generative, prints both models' unparseable and capped counts, then projects the full run from the measured speed | either model's generative answers are unparseable for more than half the smoke questions; the projected full run exceeds 27,000 s (7.5 hours of the 9-hour session) |
| 6. Full evaluation | runs `evaluate.py` on all 1,273 and all 4,183 questions, with `--gen-limit 300` | any command fails |
| 7. Results | prints, for each benchmark and mode, `n`, both accuracies, the change, wins, regressions and p, with both models' unparseable and capped counts, then copies `eval_medqa.json` and `eval_medmcqa.json` to `/kaggle/working` for the Output panel | |

Some details behind the gates:

- Each command runs through `step()`, which stops the notebook on a non-zero
  exit. A failing `!python` line does not raise in Jupyter, so without it the
  notebook would carry on past a dead step.
- The adapter search prefers a final adapter over any `checkpoint-*`
  directory, so a stray checkpoint cannot be scored in place of the finished
  run.
- The fingerprint is the first 16 hex characters of a SHA-256 over the name
  and bytes of every `.py` file, in name order. `scripts/push_kaggle.sh`
  rebuilds both notebooks before uploading the code, so the fingerprint built
  into the notebook matches the upload. A mismatch means the attached version
  of the code dataset is not that upload: a new version still processing, or
  an older one attached by hand.
- The projection scales each benchmark's measured per-example times to
  2 models × (all its constrained questions + 300 generative ones) and sums
  both benchmarks. If it does not fit, the fix it prints is to lower
  `GEN_LIMIT`, because generation dominates the runtime.
- The unparseable gate checks both models. A fine-tune whose answers mostly
  cannot be read means truncated output rather than a result, and a base that
  cannot answer (truncated, or producing fp16 garbage) would flatter the
  fine-tune. Its message says to read the completions in
  `outputs/smoke_<name>.json`, to raise `--max-new-tokens` if they are cut
  off, and not to report the result if they are garbage.

The evaluation plan expects the smoke pass to report within about 15 minutes
of the start.

## Known biases and how to read the result

[`TODO.md`](../TODO.md) lists five statements the result must carry, whatever
it finds:

1. The adapter was trained on the 4-bit base and is scored on the
   full-precision base. If that biases anything, it biases against the
   fine-tune.
2. Constrained scoring asks for the letter immediately after `Answer:`. MedQA
   was trained exactly that way, since its targets are the answer line alone.
   MedMCQA training put a rationale before the answer line, so constrained
   scoring may understate the fine-tune on MedMCQA.
3. MedMCQA's answers lean towards A (32%). A model can gain on MedMCQA by
   guessing A more often, which is not medical knowledge, and the per-question
   predictions will show whether that happened. MedQA's answers are balanced,
   so it is the cleaner test.
4. Generation stops at 512 new tokens. A completion cut off before its final
   answer loses that answer, so the cap costs whichever model writes longer
   answers. The report's `generation.hit_cap` counts show how often each model
   hit it.
5. The fine-tune was trained to answer MedQA at once. If the base model reasons
   before it answers, generative MedQA compares reasoning first with answering
   directly, which can favour the base.

The method adds a few more:

- Both models are scored with thinking disabled. That matches the fine-tune's
  training format, but it is not Qwen3's default mode, so the base number
  describes Qwen3-4B without thinking.
- Generative scoring covers 300 questions per benchmark, so its accuracies are
  less precise than the constrained ones.
- MedMCQA validation keeps the rows that training left out as noisy (marked
  multi-choice, or without a usable explanation). They cost both models alike
  and keep the score comparable with published MedMCQA figures.
- Decontamination is exact matching after normalisation, so a paraphrase of a
  benchmark question in the training data would not have been removed.
- The early signal in `TODO.md` (16 MedQA questions; base 12/16, fine-tune
  9/16, p = 0.25) is too few questions to conclude anything. It is not a
  result.

## Run it locally

Set up an environment and run the unit tests, which need no GPU and no network:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt -r requirements-smoke.txt
.venv/bin/python -m pytest
```

`requirements-dev.txt` accepts any `transformers>=4`, and an older release may
ignore the `dtype=` argument that `modeling.py` passes. To match the Kaggle
notebook, install `transformers==5.5.0` and `peft==0.19.1`.

The smoke test runs the exact evaluation code on `Qwen/Qwen3-0.6B`, which
shares Qwen3-4B's tokenizer and chat template, with a small random LoRA
adapter, on the CPU:

```bash
.venv/bin/python scripts/smoke_eval.py
```

It checks that Qwen3's chat template renders the trained prefix followed by
`Answer:`, that batch sizes 1 and 4 give the same predictions, that the
disabled adapter scores exactly like a freshly loaded base, and that the
`evaluate.py` command line writes a complete report. It prints `SMOKE PASSED`
at the end. The first run downloads Qwen3-0.6B, about 1.5 GB.

To score the run-1 adapter locally you need the adapter, which is not
published: the training notebook produces it as `run1-adapter.zip` (see
[`docs/kaggle.md`](kaggle.md)). The held-out files are built by the notebook's
cell 4; this is the same loading code without its count check, run from the
repository root:

```python
import json
from pathlib import Path
from training.sources import load

Path("data").mkdir(exist_ok=True)
for name, split, flags in (
        ("medqa", "test", {}),
        ("medmcqa", "validation", {"require_rationale": False,
                                   "require_single_choice": False})):
    recs = load(name, limit=0, split=split, **flags)
    with open(f"data/holdout_{name}.jsonl", "w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec.to_dict()) + "\n")
```

Then score a slice of one benchmark. This scores the first 16 MedQA questions
by constrained scoring and the first 4 of them by generation:

```bash
.venv/bin/python -m training.evaluate --adapter training/outputs/run1 \
    --test data/holdout_medqa.jsonl --out outputs/local_medqa.json \
    --limit 16 --gen-limit 4
```

| Flag | Default | Meaning |
|---|---|---|
| `--adapter` | required | the LoRA adapter directory |
| `--base` | `unsloth/Qwen3-4B` | the full-precision base weights |
| `--test` | required | a JSONL file of records |
| `--out` | required | where to write the report JSON |
| `--limit` | 0 | score only the first N questions; 0 scores all |
| `--gen-limit` | 300 | how many of those also get generative scoring |
| `--batch-size` | 16 | questions per forward pass or `generate()` call |
| `--max-new-tokens` | 512 | the generation cap |
| `--device` | chosen automatically | `cuda`, `mps` or `cpu` |
