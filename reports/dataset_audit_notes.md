These notes are checks done by hand on 2026-10-03, on top of the generated
numbers above. They are observations about the data, not measurements of what
the data did to the model. Phase 3 (the reasoning-damage analysis) tests that.

1. **135 ReasonMed rows (20% of ReasonMed, 1% of all rows) nest a second think
   block.** Their trace is a complete `<think>…</think>` block followed by a
   summary. Training renders the trace inside Qwen3's own think block. The
   inner tags tokenize as the real special tokens: a rendered row (reasonmed-2017)
   holds `<think>` (151667) twice and `</think>` (151668) twice. So these rows
   teach the model to open thinking twice, and to close it, keep writing, and
   close it again.

2. **MedReason traces were written knowing the answer.** 85 of its 4,125 traces
   say so: "This conclusion aligns with the provided answer", "the provided
   answer of **10%**". None of its 2,750 multiple-choice traces state a letter.
   They end in a prose conclusion, so their agreement with the gold letter can
   only be estimated from option text (the *text* columns).

3. **The text heuristic is rough.** I read six MedReason rows it marks *other
   option named*:
   - Two genuinely contradict their training target:
     - medreason-10693 concludes "allopurinol" but is trained to answer D, "All of the above".
     - medreason-2943 concludes "contemplation" but is trained to answer C, "Precontemplation".
   - One is partial: medreason-241 names only one half of "Both a & c".
   - Three are heuristic errors: "260nm" written as one token; a conclusion
     that also names a consequence ("intestinal obstruction" after "volvulus");
     a long option.

   Treat the 45 MedReason and 49 ReasonMed *other option named* rows as
   candidates for review, not as confirmed errors.

4. **Most training reasoning is not written the way Qwen3-4B thinks.** Of the
   10,179 rows with reasoning, 6,859 (67%) come from MedReason, UltraMedical
   and ReasonMed. Those sources:
   - almost never open in the first person (0–1%, against Qwen3's 100%);
   - almost never re-check themselves (0.01–0.08 "wait / actually / let me
     double-check" per 1,000 words, against Qwen3's 7.1);
   - and, apart from UltraMedical, are mostly markdown with headings.

   Only R1-Distill (19% of reasoning rows) and medical-o1 (14%) resemble
   Qwen3's style. Qwen3's traces are cut at the 1,536-token budget (p90 = 1,534),
   so its true median is above the measured 898 tokens. The training traces'
   medians are 285–800 tokens; ReasonMed's is 1,324. This fits run 3's result
   (median thinking fell from 1,036 to 403 tokens after fine-tuning), but does
   not prove the data caused it.

5. **Two sources' agreement figures are circular** (see the notes under the
   agreement table):
   - ReasonMed's training letter was read from its own trace.
   - UltraMedical rows were kept only if the trace agreed with the answer field.

6. **Truncation was negligible.** The build dropped rows over the
   3,072-token cap and 133 more over 2,496. Re-rendered with run 3's /think
   switch, one ReasonMed row is 3 tokens over 2,496 (2,499) and would have been
   cut at its end.

7. **Open: two of Phase 1's questions are not answered here.** "Is the
   reasoning correct?" and "is the reasoning incorrect while the final answer
   happens to be correct?" both need someone, or a strong judge model, to read
   the traces. A script can check whether a trace ends on the gold option. It
   cannot check whether the medicine on the way there is right.
   `dataset_audit_samples.md` holds three random rows per source for that
   reading, and the judge model is not chosen yet. Until one of those is done,
   *answer_verified* means "ends on the gold option", not "correct reasoning".
