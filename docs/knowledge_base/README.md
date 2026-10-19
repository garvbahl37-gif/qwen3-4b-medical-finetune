# Project knowledge base

This is what is known about making Qwen3-4B more accurate on medical
multiple-choice QA on a Kaggle T4, and how well each fact is supported. Read
the document for your task before changing anything.

| Task | Read first |
|---|---|
| any change to training | [fine_tuning_reference](fine_tuning_reference.md), [known_failures](known_failures.md), [decisions](decisions.md) |
| any change to prompts or templates | [qwen3_technical_reference](qwen3_technical_reference.md) |
| any change to evaluation or a new benchmark | [evaluation_reference](evaluation_reference.md) |
| new data or a new data source | [medical_dataset_reference](medical_dataset_reference.md), `reports/DATASET_AUDIT.md` |
| sizing a Kaggle run | [gpu_optimization_reference](gpu_optimization_reference.md) |
| choosing the next experiment | [open_questions](open_questions.md), [experiment_registry](experiment_registry.md) |
| finding the code | [project_architecture](project_architecture.md) |
| citing a paper or doc | [research_sources](research_sources.md) |
| what other medical fine-tunes did | [reference_models](reference_models.md) |
| what tools and skills exist | [skill_inventory](skill_inventory.md) |

## Evidence tags

Each claim that drives a decision carries one of these tags:

| Tag | Meaning |
|---|---|
| **[code]** | Verified by reading this repository's code |
| **[measured]** | Measured in a run whose artifacts are in `results/` |
| **[docs]** | Stated in official documentation (URL in research_sources) |
| **[paper]** | Reported in a research paper (URL in research_sources) |
| **[hypothesis]** | Believed but untested here. It names the experiment that would test it. |
| **[unknown]** | Not established |

Knowledge about LLM training in general is kept apart from evidence about
this model. A paper's result on Llama-8B is **[paper]**, not proof that the
same holds for Qwen3-4B. Only a run here can make it **[measured]**.

## The workflow for any model-improvement experiment

1. Read the documents for the task, above.
2. Write the hypothesis and its success and failure criteria in
   [open_questions](open_questions.md) **before** seeing a result.
3. Change one main variable. Define the experiment in
   `training/experiments_config.py`.
4. Smoke-test locally (`pytest`, then a rehearsal on Qwen3-0.6B), then run
   on Kaggle.
5. Score with `training.eval_report`. Select on the **dev** split; report the
   benchmarks.
6. Record the result in `experiments/<id>/experiment.json`, regenerate
   `experiments/results.csv`, and update [experiment_registry](experiment_registry.md),
   [decisions](decisions.md) and [known_failures](known_failures.md).

Last updated 2026-10-19.
