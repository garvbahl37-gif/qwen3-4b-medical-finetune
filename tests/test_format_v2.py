from __future__ import annotations

from training.format_v2 import (SYSTEM_MCQ_V2, answer_text, final_letter,
                                forced_suffix, format_question_v2, option_letters,
                                render_letter_prompt, render_reasoning_prompt,
                                render_training_text, short_explanation, v2_messages)
from training.prompts import SYSTEM_CHAT
from training.records import Record


def mcq(**kw) -> Record:
    fields = dict(id="q1", source="medqa", kind="mcq", question="Which drug?",
                  options={"A": "Aspirin", "B": "Heparin", "C": "Warfarin"},
                  answer="B", rationale=None, response=None, subject=None,
                  reasoning=None)
    fields.update(kw)
    return Record(**fields)


def chat(**kw) -> Record:
    fields = dict(id="c1", source="medical_o1", kind="dialogue", question="Why?",
                  options=None, answer=None, rationale=None, response="Because.",
                  subject=None, reasoning=None)
    fields.update(kw)
    return Record(**fields)


class FakeTok:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append((messages, kw))
        body = "|".join(f"{m['role']}:{m.get('reasoning_content', '')}:{m['content']}"
                        for m in messages)
        return body + ("|GEN" if kw.get("add_generation_prompt") else "")


def test_record_without_a_reasoning_key_still_loads():
    row = mcq().to_dict()
    row.pop("reasoning")
    assert Record.from_dict(row).reasoning is None


def test_option_letters_are_sorted_and_empty_for_dialogue():
    assert option_letters(mcq(options={"C": "c", "A": "a", "B": "b"})) == ["A", "B", "C"]
    assert option_letters(chat()) == []


def test_format_question_lists_every_option_in_order():
    assert format_question_v2(mcq()) == "Which drug?\n\nA. Aspirin\nB. Heparin\nC. Warfarin"
    assert format_question_v2(chat()) == "Why?"


def test_short_explanation_strips_the_label_and_keeps_two_sentences():
    assert short_explanation("Explanation: One. Two! Three? Four.") == "One. Two!"


def test_short_explanation_caps_length_on_a_word_boundary():
    out = short_explanation("word " * 200, max_chars=50)
    assert len(out) <= 53 and out.endswith("...") and not out.startswith(" ")


def test_answer_text_puts_the_letter_first():
    assert (answer_text(mcq(rationale="Heparin acts fast. It is IV. Extra."))
            == "Answer: B\n\nHeparin acts fast. It is IV.")
    assert answer_text(mcq()) == "Answer: B. Heparin"
    assert answer_text(chat()) == "Because."


def test_v2_messages_carry_reasoning_as_reasoning_content():
    msgs = v2_messages(mcq(reasoning="  think  "), with_answer=True)
    assert msgs[0] == {"role": "system", "content": SYSTEM_MCQ_V2}
    assert msgs[-1]["reasoning_content"] == "think"
    assert "reasoning_content" not in v2_messages(mcq(), with_answer=True)[-1]
    assert v2_messages(chat(), with_answer=False)[0]["content"] == SYSTEM_CHAT
    assert len(v2_messages(chat(), with_answer=False)) == 2


def test_training_text_thinks_only_when_there_is_reasoning():
    tok = FakeTok()
    render_training_text(tok, mcq(reasoning="r"))
    render_training_text(tok, mcq())
    assert [kw["enable_thinking"] for _, kw in tok.calls] == [True, False]
    assert all(kw["tokenize"] is False for _, kw in tok.calls)


def test_letter_prompt_is_non_thinking_and_ends_with_answer_colon():
    tok = FakeTok()
    text = render_letter_prompt(tok, mcq())
    kw = tok.calls[0][1]
    assert kw["enable_thinking"] is False and kw["add_generation_prompt"] is True
    assert text.endswith("|GENAnswer:")


def test_reasoning_prompt_turns_thinking_on():
    tok = FakeTok()
    render_reasoning_prompt(tok, mcq())
    assert tok.calls[0][1]["enable_thinking"] is True
    assert tok.calls[0][1]["add_generation_prompt"] is True


def test_final_letter_prefers_the_answer_line_and_respects_options():
    assert final_letter("Answer: C\n\nbecause", ["A", "B", "C"]) == "C"
    assert final_letter("Answer: D", ["A", "B", "C"]) is None
    assert final_letter("I think the correct answer is B.", ["A", "B", "C"]) == "B"
    assert final_letter("", ["A", "B"]) is None


def test_forced_suffix_closes_an_open_think_block():
    assert forced_suffix(closed=False) == "\n</think>\n\nAnswer:"
    assert forced_suffix(closed=True) == "\n\nAnswer:"


def test_short_explanation_drops_medmcqas_answer_restatement():
    assert (short_explanation("Ans. is 'c' i.e., Polypectomy. It removes the polyp. Extra.")
            == "Polypectomy. It removes the polyp.")


def test_short_explanation_does_not_split_at_abbreviations():
    assert short_explanation("See Dr. Shaw p. 373 for this. Then rest. More.") == \
        "See Dr. Shaw p. 373 for this. Then rest."


def test_every_prompt_states_its_mode_the_way_qwen3_was_trained():
    thinking = v2_messages(mcq(reasoning="r"), with_answer=True)[1]["content"]
    direct = v2_messages(mcq(), with_answer=True)[1]["content"]
    assert thinking.endswith("\n\n/think") and direct.endswith("\n\n/no_think")
    tok = FakeTok()
    assert "/no_think" in render_letter_prompt(tok, mcq(reasoning="r"))
    assert "/think" in render_reasoning_prompt(tok, mcq()) and "/no_think" not in render_reasoning_prompt(tok, mcq())
