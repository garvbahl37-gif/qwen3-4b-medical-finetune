# Medical Fine-tune: Serving and Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the fine-tuned model behind a safety screen, and give it a
frontend people can ask questions in and read the evaluation from.

**Architecture:** A FastAPI server streams answers as server-sent events. Every
stream opens with a `screen` event from `screen_message()`, so emergency guidance
reaches the page before the first token of the answer. The model backend reuses
Plan 2's `load_model`; a mock backend gives canned replies with the real screen,
so the interface builds and tests without the model. A Next.js app proxies to the
server and has three views: Ask, Results (laid out as a laboratory report), and
Method.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, `transformers` + `peft` (model
backend only); Next.js 16, React 19, TypeScript, plain CSS custom properties,
vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-09-20-medical-llm-finetune-design.md`
(sections 4 and 5). Depends on Plan 2,
`docs/superpowers/plans/2026-09-24-evaluation.md`, for `training/modeling.py` and
`prompts.render_chat`.

## Global Constraints

- Every Python module starts with `from __future__ import annotations`. No
  `torch`, `transformers` or `peft` import at module scope in any file under
  test; the server and its tests run on the mock backend with no model.
- The server binds `127.0.0.1:8008` by default. `MOCK_BACKEND=1` gives canned
  replies with the real safety screen.
- **The `screen` event is the first event of every `/chat` stream.**
- **Red `#C8102E` appears on emergency guidance and nowhere else** — not in
  charts, not on errors, not on links.
- Emergency crisis lines, exactly: **988** (US and Canada), **116 123**
  (Samaritans, UK and Ireland), **14416** (Tele-MANAS, India).
- Messages are capped at **4,000 characters**; requests carry at most **40**
  messages; the model sees at most the last **12** that fit in **12,000**
  characters.
- Fonts: **Atkinson Hyperlegible Next** for the interface and the person's words;
  **STIX Two Text** for the model's answers and the Method page. No other families.
- Copy is sentence case. No all-caps labels, no middle-dot separators, no `→` in
  button text. Errors say what happened and how to fix it; they never apologise.
- Accessibility floor: visible keyboard focus (3px `#2F5D4E` outline), text
  contrast at least 4.5:1, `prefers-reduced-motion` respected, emergency guidance
  announced with `role="alert"`, usable at 360px wide with 16px gutters and no
  horizontal scroll.
- Next.js 16, React 19, TypeScript, plain CSS with custom properties — no
  Tailwind — matching the sibling project's `web/`.
- Commits name **garvbahl37-gif as sole contributor**. The repo's git config sets
  the author; do not override it. No `Co-Authored-By` trailer and no "Generated
  with Claude Code" line on any commit.

### Design tokens

| token | value | used for |
|---|---|---|
| `--paper` | `#F2F5F1` | page background: a cool green-white |
| `--surface` | `#FFFFFF` | the composer, the report sheet, the safety panel |
| `--ink` | `#16283A` | text: chart-ink navy |
| `--ink-2` | `#4A5A68` | secondary text, axis labels |
| `--theatre` | `#2F5D4E` | structure, links, focus, primary button |
| `--theatre-data` | `#1F7A55` | the chart line only — validated: chroma ≥ 0.1, contrast ≥ 3:1 on paper |
| `--rule` | `#C9D8CF` | dividers, gridlines, input borders |
| `--alarm` | `#C8102E` | emergency guidance only |
| `--caution` | `#8A5A00` | notes on questions it shouldn't answer |

Operating-theatre green was adopted in surgery because it complements red; on a
green field a small amount of red cannot be missed, which is how the emergency
panel is unmissable without being loud. Type scale is a major third from a 17px
base: 14, 17, 21, 27. Serif answers run 18px on a 1.62 line height; sans text
1.55. One column, left-aligned, at most 44rem wide.

## Review Focus

1. **An exam vignette is not an emergency.** "A 60-year-old man presents with
   crushing chest pain radiating to his jaw. Which of the following…" must get no
   emergency panel, or the panel fires on every pasted question and stops being
   read. "My dad has crushing chest pain radiating to his jaw" must get one.
   Pinned in Task 1.
2. **Lay phrasing and phone keyboards.** Emergencies arrive without clinical
   words ("his face is drooping and he can't lift his arm") and with curly
   apostrophes from phone keyboards ("can’t"). Both must fire. Pinned in Task 1.
3. **The warning before the answer.** Emergency guidance must be on the page
   before the first answer token, and must stay there if generation fails. Pinned
   in Task 2 (event order, failure mid-stream) and Task 6 (the panel renders above
   the answer).
4. **The model server is not running.** The page must say so and say how to start
   it, not hang or show nothing. Pinned in Task 2 (503 with the reason) and
   Task 4 (the proxy's unreachable-server message).
5. **A long paste or a long conversation.** An over-long message is refused with
   a message saying the limit; a long history is trimmed to the most recent turns
   that fit, starting on a question. Never a crash, never a silent overflow.
   Pinned in Task 2.

---

### Task 1: The safety screen

**Files:**
- Create: `serving/__init__.py` (empty)
- Create: `serving/safety.py`
- Create: `tests/test_safety.py`
- Modify: `training/prompts.py` (add `looks_like_mcq`)
- Modify: `tests/test_prompts.py` (append two tests)

**Interfaces:**
- Consumes: nothing beyond the standard library.
- Produces:
  - `prompts.looks_like_mcq(text: str) -> bool`
  - `safety.Flag` — frozen dataclass: `category: str`, `heading: str`, `advice: str`
  - `safety.Screen` — frozen dataclass: `emergency: tuple[Flag, ...]`,
    `out_of_scope: tuple[Flag, ...]`, `exam_context: bool`; `to_dict() -> dict`
    with keys `emergency`, `out_of_scope` (lists of `{category, heading, advice}`)
    and `exam_context`
  - `safety.screen_message(text: str) -> Screen`
  - Emergency categories: `stroke`, `cardiac`, `breathing`, `anaphylaxis`,
    `suicidal`, `bleeding`, `infant_fever`. Out-of-scope categories: `dosing`,
    `stopping_medication`, `diagnosis`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prompts.py`, adding `looks_like_mcq` to its import:

```python
def test_looks_like_mcq_detects_lettered_options():
    assert looks_like_mcq("Which?\nA. one\nB. two\nC. three\nD. four")
    assert looks_like_mcq("Which?\n(A) one\n(B) two\n(C) three")


def test_looks_like_mcq_ignores_prose_that_starts_with_a_capital_a():
    assert not looks_like_mcq("I have a headache.\nA doctor told me to rest.")
```

Create `tests/test_safety.py`:

```python
from __future__ import annotations

import pytest

from serving.safety import screen_message

EMERGENCIES = {
    "stroke": ["My dad's face is drooping on one side and his words are slurred",
               "My mum suddenly can't lift her left arm"],
    "cardiac": ["I have crushing chest pain spreading to my left arm and I'm sweating",
                "I think I'm having a heart attack"],
    "breathing": ["I can't breathe properly",
                  "My son's lips are turning blue"],
    "anaphylaxis": ["My throat feels like it's closing after I ate peanuts",
                    "I got stung by a bee and now my lips are swelling and I feel dizzy"],
    "suicidal": ["I want to end my life",
                 "I've been thinking about killing myself"],
    "bleeding": ["My cut is bleeding and it won't stop",
                 "I'm coughing up blood"],
    "infant_fever": ["My 6 week old baby has a fever of 38.5",
                     "My newborn feels hot and has a high temperature"],
}


@pytest.mark.parametrize("category,text", [
    (category, text) for category, texts in EMERGENCIES.items() for text in texts])
def test_each_emergency_fires_on_lay_phrasing(category, text):
    screen = screen_message(text)
    assert category in {f.category for f in screen.emergency}, text


# A guard that fires on everything is no guard. None of these may fire.
BENIGN = [
    "What vitamins are found in spinach?",
    "How does insulin lower blood sugar?",
    "What's the difference between a virus and a bacterium?",
    "Explain the stages of mitosis.",
    "What are the common causes of chest pain?",
    "How is a stroke diagnosed on a CT scan?",
    "How is anaphylaxis treated in hospital?",
    "What does a normal ECG look like?",
    "Why do we get a fever when we have an infection?",
    "How long does a common cold usually last?",
    "What does the kidney do?",
    "Which foods are high in iron?",
    "How much water should an adult drink each day?",
    "What are the side effects of ibuprofen?",
    "Summarise how beta blockers work.",
    "What is the treatment for a bee sting?",
    "Can you explain what causes lip swelling?",
]


@pytest.mark.parametrize("text", BENIGN)
def test_benign_questions_raise_nothing(text):
    screen = screen_message(text)
    assert screen.emergency == () and screen.out_of_scope == (), text


VIGNETTE = (
    "A 60-year-old man presents to the emergency department with crushing chest "
    "pain radiating to his jaw and diaphoresis. Which of the following is the "
    "most likely diagnosis?\nA. Aortic dissection\nB. Myocardial infarction\n"
    "C. Pericarditis\nD. Pulmonary embolism")


def test_an_exam_vignette_is_not_an_emergency():
    screen = screen_message(VIGNETTE)
    assert screen.exam_context is True
    assert screen.emergency == ()


def test_the_same_symptoms_in_the_first_person_are():
    screen = screen_message(
        "My dad has crushing chest pain radiating to his jaw and he's sweating")
    assert screen.exam_context is False
    assert "cardiac" in {f.category for f in screen.emergency}


def test_a_curly_apostrophe_from_a_phone_keyboard_still_fires():
    screen = screen_message("My dad can’t lift his arm")
    assert "stroke" in {f.category for f in screen.emergency}


def test_several_emergencies_in_one_message_are_all_reported():
    screen = screen_message("I can't breathe and my throat is swelling")
    assert {"breathing", "anaphylaxis"} <= {f.category for f in screen.emergency}


@pytest.mark.parametrize("category,text", [
    ("dosing", "How many paracetamol tablets can I take in a day?"),
    ("dosing", "Should I double my dose of metformin?"),
    ("stopping_medication", "Can I stop taking my antidepressants?"),
    ("diagnosis", "Do I have diabetes?"),
])
def test_personal_care_decisions_get_an_out_of_scope_note(category, text):
    assert category in {f.category for f in screen_message(text).out_of_scope}


def test_an_educational_dosing_question_gets_no_note():
    # Standard doses are textbook material. The note is for personal decisions.
    screen = screen_message("What dose of amoxicillin is used for otitis media?")
    assert screen.out_of_scope == ()


def test_suicide_guidance_names_the_crisis_lines():
    flag = next(f for f in screen_message("I want to end my life").emergency
                if f.category == "suicidal")
    for line in ("988", "116 123", "14416"):
        assert line in flag.advice


def test_to_dict_has_the_shape_the_frontend_reads():
    out = screen_message("I can't breathe").to_dict()
    assert set(out) == {"emergency", "out_of_scope", "exam_context"}
    assert set(out["emergency"][0]) == {"category", "heading", "advice"}
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_safety.py tests/test_prompts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'serving'` and
`ImportError: cannot import name 'looks_like_mcq'`.

- [ ] **Step 3: Add `looks_like_mcq` to `training/prompts.py`**

After `render_inference_prompt`:

```python
_OPTION_LINE = re.compile(r"^\s*\(?[A-Da-d][.):]\s+\S", re.MULTILINE)


def looks_like_mcq(text: str) -> bool:
    """True when the text carries at least three lettered options.

    Training paired questions with lettered options with SYSTEM_MCQ, and the
    safety screen reads them as exam context rather than as a person's own
    symptoms.
    """
    return len(_OPTION_LINE.findall(text or "")) >= 3
```

- [ ] **Step 4: Write `serving/safety.py`**

```python
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from training.prompts import looks_like_mcq

_I = re.IGNORECASE
_S = r"[^.?!\n]"   # stays inside one sentence


@dataclass(frozen=True)
class Flag:
    category: str
    heading: str
    advice: str


@dataclass(frozen=True)
class Screen:
    emergency: tuple[Flag, ...] = ()
    out_of_scope: tuple[Flag, ...] = ()
    exam_context: bool = False

    def to_dict(self) -> dict:
        return {
            "emergency": [asdict(f) for f in self.emergency],
            "out_of_scope": [asdict(f) for f in self.out_of_scope],
            "exam_context": self.exam_context,
        }


@dataclass(frozen=True)
class _Rule:
    flag: Flag
    patterns: tuple[re.Pattern, ...]

    def matches(self, text: str) -> bool:
        return any(p.search(text) for p in self.patterns)


def _rule(category: str, heading: str, advice: str, *patterns: str,
          flags: int = _I) -> _Rule:
    return _Rule(Flag(category, heading, advice),
                 tuple(re.compile(p, flags) for p in patterns))


EMERGENCIES = (
    _rule("stroke",
          "These can be signs of a stroke. Call emergency services now.",
          "Note the time the symptoms started and don't wait to see if they "
          "pass. Treatment works best in the first few hours.",
          rf"\b(?:face|mouth|smile)\b{_S}{{0,40}}\b(?:droop\w*|lopsided|uneven)",
          r"\bslurr(?:ed|ing)\b",
          r"\b(?:can'?t|cannot|can not|unable to)\s+(?:lift|raise|move)\s+"
          r"(?:(?:his|her|my|their|one|the|an?)\s+)?(?:(?:left|right)\s+)?"
          r"(?:arm|leg|hand)",
          rf"\bsudden(?:ly)?\b{_S}{{0,30}}\b(?:weak\w*|numb\w*)\b{_S}{{0,30}}"
          r"\b(?:one side|left side|right side|arm|leg|face)"),
    _rule("cardiac",
          "Chest pain like this can be a heart attack. Call emergency services now.",
          "Stop what you're doing and sit down while you wait. Don't drive "
          "yourself to hospital.",
          rf"\bchest\b{_S}{{0,25}}\b(?:pain|pressure|tight\w*|crushing|squeez\w*|"
          rf"heav\w*)\b{_S}{{0,70}}\b(?:arm|jaw|neck|back|shoulder|sweat\w*|clammy|"
          r"breath\w*|nause\w*|sick|faint\w*|dizzy)",
          r"\b(?:having|have|had)\s+(?:a\s+)?heart attack\b"),
    _rule("breathing",
          "Serious trouble breathing is an emergency. Call emergency services now.",
          "Stay with the person, help them sit upright if they can, and loosen "
          "anything tight around the neck.",
          r"\b(?:can'?t|cannot|can not|struggling to|unable to)\s+(?:breathe|"
          r"catch\s+(?:my|his|her|their)\s+breath)\b",
          rf"\b(?:lips|face|skin|fingers)\b{_S}{{0,20}}\b(?:turn\w*\s+)?"
          r"(?:blue|grey|gray)\b",
          r"\bchoking\b"),
    _rule("anaphylaxis",
          "Swelling of the throat, tongue or lips can be anaphylaxis. "
          "Call emergency services now.",
          "If an adrenaline auto-injector such as an EpiPen has been prescribed, "
          "use it now.",
          rf"\b(?:throat|tongue)\b{_S}{{0,25}}\b(?:swell\w*|swollen|clos\w*|tight\w*)\b",
          rf"\b(?:lips?|face|mouth)\b{_S}{{0,25}}\b(?:swell\w*|swollen)\b{_S}{{0,60}}"
          r"\b(?:breath\w*|allerg\w*|reaction|sting|stung|peanuts?|ate|eating|"
          r"dizzy|faint\w*)",
          rf"\b(?:allerg\w*|reaction|sting|stung|peanuts?|ate|eating)\b{_S}{{0,60}}"
          rf"\b(?:lips?|face|mouth|throat|tongue)\b{_S}{{0,25}}\b(?:swell\w*|swollen)"),
    _rule("suicidal",
          "If you're thinking about ending your life, please talk to someone now.",
          "Call your local emergency number, or a crisis line: 988 in the US and "
          "Canada, 116 123 (Samaritans) in the UK and Ireland, 14416 (Tele-MANAS) "
          "in India.",
          r"\b(?:kill|killing|hurt|hurting|harm|harming)\s+my\s?self\b",
          r"\b(?:want|wanted|wanting|plan|planning)\s+to\s+(?:die|end\s+(?:it all|"
          r"it|my life|things))\b",
          r"\bend(?:ing)?\s+my\s+life\b",
          r"\bsuicid\w*",
          r"\bno\s+reason\s+to\s+live\b",
          r"\bbetter\s+off\s+(?:dead|without\s+me)\b"),
    _rule("bleeding",
          "Heavy bleeding, or bleeding that won't stop, is an emergency. "
          "Call emergency services now.",
          "If the blood is coming from a wound, press firmly on it with a clean "
          "cloth and keep pressing until help arrives.",
          rf"\bbleed\w*\b{_S}{{0,30}}\b(?:won'?t|will not|isn'?t|doesn'?t|can'?t)"
          r"\s+stop",
          rf"\bbleed\w*\b{_S}{{0,20}}\b(?:heavily|a lot|badly|spurting|pouring|"
          r"everywhere)\b",
          r"\b(?:cough\w*|vomit\w*|throw\w*)\s+(?:up\s+)?blood\b",
          r"\bsoak(?:ed|ing)\s+through\b"),
    _rule("infant_fever",
          "A fever in a baby under 3 months old needs urgent medical care.",
          "Call your doctor or emergency services now, even if the baby otherwise "
          "seems well.",
          r"(?=.*\b(?:baby|infant|newborn)\b)(?=.*\b(?:fever\w*|temperature|temp)\b)"
          r"(?=.*(?:\bnewborn\b|\b(?:[1-9]|1[0-2])\s*-?\s*(?:weeks?|wks?)\b|"
          r"\b(?:1|2|one|two)\s*-?\s*months?\b))",
          flags=_I | re.DOTALL),
)

OUT_OF_SCOPE = (
    _rule("dosing",
          "It can't tell you how much of a medicine to take.",
          "A pharmacist or your doctor can, based on your weight, age and the "
          "other medicines you take.",
          rf"\bhow\s+(?:much|many)\b{_S}{{0,40}}\b(?:mg|milligrams?|pills?|tablets?|"
          r"capsules?|ml|doses?|units)\b",
          r"\b(?:what|which)\s+(?:dose|dosage)\b",
          r"\b(?:double|increase|decrease|raise|lower|change)\s+(?:my|the)\s+"
          r"(?:dose|dosage)\b"),
    _rule("stopping_medication",
          "Stopping some medicines suddenly can be dangerous.",
          "Talk to the doctor who prescribed it before you change anything.",
          rf"\b(?:stop|stopping|quit|quitting|come off|coming off|skip|skipping)\s+"
          rf"(?:taking\s+)?(?:my|the|these|this)\b{_S}{{0,20}}\b(?:medication|"
          r"medicine|meds|pills|tablets|antidepressants?|insulin|inhaler|steroids?|"
          r"blood thinners?)\b"),
    _rule("diagnosis",
          "It can explain what symptoms can mean, but it can't examine or "
          "diagnose you.",
          "A doctor can.",
          r"\bdo\s+i\s+have\b",
          r"\bdiagnos\w*\s+me\b",
          r"\bwhat\s+do\s+i\s+have\b",
          r"\bis\s+(?:this|it)\s+(?:cancer|serious)\b"),
)

# First person means the person is describing their own situation, or someone
# close to them. A bare "I" is left out on purpose: exam questions are full of
# "Type I" and "Class I", and those must not turn a vignette into a confession.
_FIRST_PERSON = re.compile(
    r"\b(?:i'm|i am|i've|i have|i had|i was|i feel|i felt|i think|i can'?t|"
    r"i cannot|i keep|i got|i took|i take|i need|i woke|me|my|mine|myself|"
    r"we|we're|our|us)\b|\b(?:can|should|do|am|could|would)\s+i\b", _I)
_VIGNETTE = re.compile(r"\b(?:a|an)\s+\d{1,3}[\s-]*(?:year|month|week|day)s?"
                       r"[\s-]*old\b", _I)
_EXAM_STEM = re.compile(
    r"\bwhich of the following\b|\bmost likely (?:diagnosis|cause)\b|"
    r"\bmost appropriate (?:next step|treatment|management)\b|"
    r"\bpresents? (?:to|with)\b|\bis brought (?:to|in)\b|"
    r"\bcomes to the (?:clinic|office|emergency department|physician)\b", _I)


def _first_person(text: str) -> bool:
    return bool(_FIRST_PERSON.search(text))


def _is_exam_question(text: str) -> bool:
    exam_shaped = (_VIGNETTE.search(text) or _EXAM_STEM.search(text)
                   or looks_like_mcq(text))
    return bool(exam_shaped) and not _first_person(text)


def screen_message(text: str) -> Screen:
    """Flag emergencies and personal-care questions in one message.

    The model is an exam assistant, so people paste vignettes constantly. A
    third-person vignette is read as exam context and flags nothing: a panel
    that fires on every pasted question stops being read. The same symptoms
    described in the first person are flagged. Out-of-scope notes are for
    personal decisions only, so they need the first person too.
    """
    text = (text or "").replace("’", "'").replace("‘", "'")
    if _is_exam_question(text):
        return Screen(exam_context=True)
    emergency = tuple(r.flag for r in EMERGENCIES if r.matches(text))
    out_of_scope = (tuple(r.flag for r in OUT_OF_SCOPE if r.matches(text))
                    if _first_person(text) else ())
    return Screen(emergency, out_of_scope, False)
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_safety.py tests/test_prompts.py -q`
Expected: PASS. `test_safety.py` collects 14 emergency cases, 17 benign cases and
11 further cases.

If a benign case fires or an emergency case does not, fix the pattern in
`serving/safety.py`, never the test. The benign list is the specification of
what must stay quiet.

- [ ] **Step 6: Run the full suite, then commit**

Run: `.venv/bin/python -m pytest -q` — every test passes.

```bash
git add serving/__init__.py serving/safety.py tests/test_safety.py \
        training/prompts.py tests/test_prompts.py
git commit -m "Screen each message for emergencies before the model answers

Seven emergencies in lay language -- stroke signs, cardiac chest pain,
trouble breathing, anaphylaxis, suicidal thoughts, heavy bleeding, and
fever in a baby under three months -- plus notes on dosing, stopping
medication, and asking for a diagnosis.

The hard case is the model's own job: people paste exam vignettes, and
'A 60-year-old man presents with crushing chest pain' would fire on
every one. Third-person vignettes are read as exam context and flag
nothing; the same symptoms in the first person are flagged. A panel
that fires on everything stops being read.

Seventeen benign questions pin what must stay quiet, and curly
apostrophes from phone keyboards are normalised before matching."
```

---

### Task 2: The server

**Files:**
- Create: `serving/backends.py`
- Create: `serving/app.py`
- Create: `serving/start_local.sh`
- Modify: `training/prompts.py` (add `system_prompt_for`, `trim_history`)
- Modify: `tests/test_prompts.py` (append three tests)
- Create: `tests/test_server.py`
- Modify: `requirements-dev.txt` (add `fastapi`, `httpx`, `uvicorn`)

**Interfaces:**
- Consumes: `safety.screen_message`, `prompts.looks_like_mcq`,
  `prompts.render_chat`, `prompts.SYSTEM_MCQ`, `prompts.SYSTEM_CHAT`,
  `modeling.load_model`, `modeling.DEFAULT_BASE`.
- Produces:
  - `prompts.system_prompt_for(text: str) -> str`
  - `prompts.trim_history(messages: list[dict], *, max_messages: int = 12, max_chars: int = 12_000) -> list[dict]`
  - `backends.MockBackend(delay: float = 0.03)`,
    `backends.ModelBackend(base, adapter, device)`,
    `backends.make_backend(env) -> Backend`; every backend has `name`, `ready`,
    `detail`, `load()`, `stream(messages) -> Iterator[str]`
  - `app.create_app(backend=None) -> FastAPI` and module-level `app`
  - HTTP: `GET /health` → `{"ready": bool, "backend": str, "detail": str}`;
    `POST /chat` with `{"messages": [{"role": "user"|"assistant", "content": str}]}`
    → `text/event-stream` of `screen`, then `token`×N, then `done` or `error`.
    Each event is `event: <name>\ndata: <json>\n\n`. `token` data is
    `{"text": str}`, `done` data `{"finish_reason": "stop"}`, `error` data
    `{"message": str}`, `screen` data is `Screen.to_dict()`.
  - Error statuses carry `{"detail": str}`: 413 over-long message, 422 empty or
    last-not-user, 503 backend not ready.

- [ ] **Step 1: Install the server dependencies**

```bash
printf 'fastapi>=0.115\nhttpx>=0.27\nuvicorn>=0.30\n' >> requirements-dev.txt
.venv/bin/pip install -q -r requirements-dev.txt
.venv/bin/python -c "import fastapi, httpx, uvicorn; print('fastapi', fastapi.__version__)"
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_prompts.py`, adding `system_prompt_for, trim_history,
SYSTEM_MCQ, SYSTEM_CHAT` to its import:

```python
def test_system_prompt_for_matches_the_format_the_model_was_trained_on():
    assert system_prompt_for("Which?\nA. a\nB. b\nC. c\nD. d") == SYSTEM_MCQ
    assert system_prompt_for("Why do I get headaches?") == SYSTEM_CHAT


def test_trim_history_keeps_the_most_recent_turns_and_starts_on_a_question():
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(31)]
    kept = trim_history(msgs, max_messages=12)
    assert kept[-1]["content"] == "m30"
    assert len(kept) <= 12 and kept[0]["role"] == "user"


def test_trim_history_respects_a_character_budget():
    x = {"role": "user", "content": "x" * 5000}
    y = {"role": "assistant", "content": "y" * 5000}
    z = {"role": "user", "content": "z" * 3000}
    # y + z fit in 9,000 but would start on the assistant, so y goes too.
    assert trim_history([x, y, z], max_chars=9000) == [z]
```

Create `tests/test_server.py`:

```python
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from serving.app import MAX_MESSAGE_CHARS, create_app
from serving.backends import MockBackend, ModelBackend


def events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((fields["event"], json.loads(fields["data"])))
    return out


def ask(client, text, history=()):
    return client.post("/chat", json={"messages": [
        *history, {"role": "user", "content": text}]})


@pytest.fixture
def client():
    return TestClient(create_app(MockBackend(delay=0)))


def test_health_reports_the_mock_backend_ready(client):
    assert client.get("/health").json() == {
        "ready": True, "backend": "mock",
        "detail": "mock backend: canned replies, real safety screen"}


def test_chat_streams_the_screen_then_tokens_then_done(client):
    names = [name for name, _ in events(ask(client, "How do beta blockers work?").text)]
    assert names[0] == "screen" and names[-1] == "done"
    assert set(names[1:-1]) == {"token"}


def test_emergency_guidance_is_the_first_thing_sent(client):
    name, data = events(ask(client, "My dad's face is drooping").text)[0]
    assert name == "screen"
    assert data["emergency"][0]["category"] == "stroke"


def test_an_exam_vignette_is_screened_as_exam_context(client):
    _, data = events(ask(client, "A 60-year-old man presents with crushing chest "
                                 "pain radiating to his jaw. Which of the "
                                 "following is the most likely diagnosis?").text)[0]
    assert data["exam_context"] is True and data["emergency"] == []


def test_an_empty_question_is_refused(client):
    response = ask(client, "   ")
    assert response.status_code == 422
    assert response.json()["detail"] == "Type a question first."


def test_an_over_long_message_is_refused_with_the_limit(client):
    response = ask(client, "x" * (MAX_MESSAGE_CHARS + 1))
    assert response.status_code == 413
    assert "4,000" in response.json()["detail"]


def test_the_last_message_must_be_the_question(client):
    response = client.post("/chat", json={"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}]})
    assert response.status_code == 422


def test_a_backend_that_is_not_ready_says_why(tmp_path):
    backend = ModelBackend(base="unused", adapter=str(tmp_path / "missing"))
    backend.load()
    response = ask(TestClient(create_app(backend)), "hello")
    assert response.status_code == 503
    assert "No adapter" in response.json()["detail"]


class _FailsAfterOneToken(MockBackend):
    def stream(self, messages):
        yield "Partly "
        raise RuntimeError("out of memory")


def test_a_failure_mid_answer_becomes_an_error_event_after_the_screen():
    client = TestClient(create_app(_FailsAfterOneToken(delay=0)))
    names = [n for n, _ in events(ask(client, "My dad's face is drooping").text)]
    assert names == ["screen", "token", "error"]
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_server.py tests/test_prompts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'serving.app'` and
`ImportError: cannot import name 'system_prompt_for'`.

- [ ] **Step 4: Add the two helpers to `training/prompts.py`**

After `looks_like_mcq`:

```python
def system_prompt_for(text: str) -> str:
    """The system prompt the model was trained with for this kind of input.

    Training paired lettered-option questions with SYSTEM_MCQ and everything
    else with SYSTEM_CHAT. Serving the other one asks the model to answer in a
    format it never learned.
    """
    return SYSTEM_MCQ if looks_like_mcq(text) else SYSTEM_CHAT


def trim_history(messages: list[dict], *, max_messages: int = 12,
                 max_chars: int = 12_000) -> list[dict]:
    """The most recent turns that fit, always starting on a question.

    The newest message is always kept, whatever its length; the request limit
    already caps it.
    """
    kept: list[dict] = []
    total = 0
    for message in reversed(messages):
        over_budget = total + len(message["content"]) > max_chars
        if len(kept) >= max_messages or (over_budget and kept):
            break
        kept.append(message)
        total += len(message["content"])
    kept.reverse()
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept
```

- [ ] **Step 5: Write `serving/backends.py`**

```python
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterator, Protocol

from training.modeling import DEFAULT_BASE

DEFAULT_ADAPTER = "training/outputs/run1"
MAX_NEW_TOKENS = 1024

MOCK_REPLY = (
    "This is the mock backend, so this reply is canned rather than written by "
    "the model. The safety screen above it is real. Start the model with "
    "bash serving/start_local.sh to get real answers."
)


class Backend(Protocol):
    name: str
    ready: bool
    detail: str

    def load(self) -> None: ...

    def stream(self, messages: list[dict]) -> Iterator[str]: ...


class MockBackend:
    """Canned replies and no model, so the interface can be built and tested
    without a GPU or 8GB of weights. The safety screen still runs for real: it
    lives in the server, not in the backend."""

    name = "mock"

    def __init__(self, delay: float = 0.03) -> None:
        self.delay = delay
        self.ready = True
        self.detail = "mock backend: canned replies, real safety screen"

    def load(self) -> None:
        return None

    def stream(self, messages: list[dict]) -> Iterator[str]:
        for word in MOCK_REPLY.split(" "):
            if self.delay:
                time.sleep(self.delay)
            yield word + " "


class ModelBackend:
    """Full-precision Qwen3-4B with the trained LoRA adapter."""

    name = "model"

    def __init__(self, base: str = DEFAULT_BASE,
                 adapter: str | None = DEFAULT_ADAPTER,
                 device: str | None = None) -> None:
        self.base, self.adapter, self.device = base, adapter, device
        self.ready = False
        self.detail = "not loaded yet"
        self._lock = threading.Lock()

    def load(self) -> None:
        if self.adapter and not (Path(self.adapter) / "adapter_config.json").exists():
            self.detail = (f"No adapter at {self.adapter}. Put the trained adapter "
                           "there, or set ADAPTER to its directory.")
            return
        from training.modeling import load_model

        self.model, self.tok, choice = load_model(self.base, self.adapter,
                                                  device=self.device)
        self.ready = True
        self.detail = f"{self.base} + {self.adapter or 'no adapter'} on {choice.device}"

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield the answer as it is generated.

        Generation runs on a worker thread feeding a streamer. If the client
        goes away, the stop flag ends generation at the next token instead of
        letting it run to max_new_tokens holding the model.
        """
        import torch
        from transformers import (StoppingCriteria, StoppingCriteriaList,
                                  TextIteratorStreamer)

        from training.prompts import render_chat

        stop = threading.Event()

        class _StopWhenAsked(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop.is_set()

        enc = self.tok(render_chat(self.tok, messages), return_tensors="pt",
                       add_special_tokens=False).to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True,
                                        skip_special_tokens=True)
        failure: list[BaseException] = []

        def generate() -> None:
            try:
                with torch.no_grad():
                    self.model.generate(
                        **enc, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True, temperature=0.3, top_p=0.9,
                        pad_token_id=self.tok.pad_token_id,
                        stopping_criteria=StoppingCriteriaList([_StopWhenAsked()]))
            except BaseException as exc:  # surfaced to the caller below
                failure.append(exc)
                streamer.end()

        with self._lock:
            worker = threading.Thread(target=generate, daemon=True)
            worker.start()
            try:
                for piece in streamer:
                    yield piece
            finally:
                stop.set()
                worker.join()
        if failure:
            raise failure[0]


def make_backend(env) -> Backend:
    if env.get("MOCK_BACKEND") == "1":
        return MockBackend()
    return ModelBackend(base=env.get("BASE") or DEFAULT_BASE,
                        adapter=env.get("ADAPTER") or DEFAULT_ADAPTER,
                        device=env.get("DEVICE") or None)
```

- [ ] **Step 6: Write `serving/app.py`**

```python
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from serving.backends import Backend, make_backend
from serving.safety import screen_message
from training.prompts import system_prompt_for, trim_history

MAX_MESSAGE_CHARS = 4_000
MAX_MESSAGES = 40


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=MAX_MESSAGES)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(backend: Backend | None = None) -> FastAPI:
    """Build the app. With no backend, one is made from the environment at
    startup, so importing this module never loads a model."""
    state: dict = {"backend": backend}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["backend"] is None:
            state["backend"] = make_backend(os.environ)
            state["backend"].load()
        yield

    app = FastAPI(title="Medical study assistant", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        b = state["backend"]
        if b is None:
            return {"ready": False, "backend": None, "detail": "starting"}
        return {"ready": b.ready, "backend": b.name, "detail": b.detail}

    @app.post("/chat")
    def chat(req: ChatRequest) -> StreamingResponse:
        last = req.messages[-1]
        if last.role != "user":
            raise HTTPException(422, "The last message must be the question.")
        if not last.content.strip():
            raise HTTPException(422, "Type a question first.")
        if any(len(m.content) > MAX_MESSAGE_CHARS for m in req.messages):
            raise HTTPException(
                413, f"Messages are limited to {MAX_MESSAGE_CHARS:,} characters. "
                     "Shorten it and try again.")
        backend = state["backend"]
        if backend is None or not backend.ready:
            raise HTTPException(503, backend.detail if backend else
                                "The model is still starting. Try again shortly.")

        screen = screen_message(last.content)
        conversation = ([{"role": "system",
                          "content": system_prompt_for(last.content)}]
                        + trim_history([m.model_dump() for m in req.messages]))

        def events():
            # The screen goes first, before generation starts, so emergency
            # guidance is on the page even if the model is slow or fails.
            yield sse("screen", screen.to_dict())
            try:
                for piece in backend.stream(conversation):
                    yield sse("token", {"text": piece})
            except Exception as exc:
                yield sse("error", {"message": f"The model stopped with "
                                    f"{type(exc).__name__}. Ask again to retry."})
                return
            yield sse("done", {"finish_reason": "stop"})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app


app = create_app()
```

- [ ] **Step 7: Write `serving/start_local.sh`**

```bash
#!/usr/bin/env bash
# Serve the fine-tuned model on this machine.
#
#   bash serving/start_local.sh                   # the real model
#   MOCK_BACKEND=1 bash serving/start_local.sh    # canned replies, no model
#
# The real model needs requirements-smoke.txt installed and the adapter at
# training/outputs/run1 (or ADAPTER=...). On Apple silicon it runs on MPS.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8008}"
exec .venv/bin/python -m uvicorn serving.app:app --host 127.0.0.1 --port "$PORT"
```

Then `chmod +x serving/start_local.sh`.

- [ ] **Step 8: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_server.py tests/test_prompts.py -q`
Expected: PASS — 9 in `test_server.py`.

- [ ] **Step 9: Check the real server end to end in mock mode**

```bash
MOCK_BACKEND=1 PORT=8011 bash serving/start_local.sh > /tmp/medserve.log 2>&1 &
until curl -sf http://127.0.0.1:8011/health >/dev/null; do sleep 0.5; done
curl -s http://127.0.0.1:8011/health; echo
curl -sN -X POST http://127.0.0.1:8011/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"My dad can’t lift his arm"}]}' | head -4
kill %1
```

Expected: the health JSON, then `event: screen` with a `stroke` flag, then the
first `event: token`.

- [ ] **Step 10: Run the full suite, then commit**

Run: `.venv/bin/python -m pytest -q` — every test passes.

```bash
git add serving/ training/prompts.py tests/test_prompts.py tests/test_server.py \
        requirements-dev.txt
git commit -m "Serve the model as a stream that opens with the safety screen

Every answer streams as server-sent events, and the first event is
always the screen, sent before generation starts. Emergency guidance is
on the page even if the model is slow, and it stays there if
generation fails partway: the failure becomes an error event after it.

The model backend reuses the evaluation loader and stops generating
when the client goes away. A mock backend gives canned replies with
the real screen, so the interface can be built without 8GB of weights.

The system prompt follows the input -- lettered options get the one
the model was trained with for exam questions -- and history is
trimmed to the most recent turns that fit, starting on a question."
```

---

### Task 3: The web app's shell and design system

**Files:**
- Create: `web/package.json`, `web/tsconfig.json`, `web/next.config.mjs`,
  `web/vitest.config.ts`, `web/.env.example`
- Create: `web/scripts/sync-data.mjs`
- Create: `web/app/layout.tsx`, `web/app/globals.css`, `web/app/page.tsx`
- Create: `web/components/Nav.tsx`
- Create: `web/lib/types.ts`
- Create: `results/run1/run.json`
- Modify: `.gitignore` (add `web/data/`)

**Interfaces:**
- Consumes: the `/chat` event contract from Task 2; `results/run1/*.json`.
- Produces:
  - `web/lib/types.ts`: `ScreenFlag`, `Screen`, `ServerEvent`, `EvalReport`,
    `EvalFile`, `TrainStats`, `DataReport`, `RunInfo`, `LossPoint`
  - CSS tokens and classes in `globals.css` that Tasks 4-6 use
  - `npm run dev` / `build` / `test` / `e2e`; `predev` and `prebuild` copy
    `results/run1/*.json` into `web/data/`

- [ ] **Step 1: Record the run's facts**

```bash
cat > results/run1/run.json <<'JSON'
{
  "trained_on": "2026-09-21",
  "base": "Qwen3-4B",
  "lora_rank": 32,
  "gpu": "Tesla T4",
  "kaggle_kernel": "gb1105/qwen3-4b-medical-fine-tune-training"
}
JSON
printf '\n# synced from results/ by web/scripts/sync-data.mjs\nweb/data/\n' >> .gitignore
```

- [ ] **Step 2: Scaffold the app and install**

`web/package.json`:

```json
{
  "name": "medical-study-assistant",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "predev": "node scripts/sync-data.mjs",
    "dev": "next dev",
    "prebuild": "node scripts/sync-data.mjs",
    "build": "next build",
    "start": "next start",
    "test": "vitest run",
    "pree2e": "node scripts/sync-data.mjs",
    "e2e": "playwright test"
  }
}
```

```bash
cd web
npm install next@16 react@19 react-dom@19
npm install -D typescript @types/node @types/react @types/react-dom vitest @playwright/test
cd ..
```

`web/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": false,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

`web/next.config.mjs`:

```js
/** @type {import('next').NextConfig} */
export default { reactStrictMode: true };
```

`web/vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";

export default defineConfig({ test: { include: ["lib/**/*.test.ts"] } });
```

`web/.env.example`:

```bash
# Where the model server listens. Start it with: bash serving/start_local.sh
MODEL_URL=http://127.0.0.1:8008
```

`web/scripts/sync-data.mjs`:

```js
// Copy the run's results into web/data so the pages read one source of truth:
// results/run1 in the repository. Missing files are fine -- the evaluation
// files only exist once the evaluation has run, and the pages say so.
import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { join } from "node:path";

const from = join(process.cwd(), "..", "results", "run1");
const to = join(process.cwd(), "data");
mkdirSync(to, { recursive: true });
const files = existsSync(from) ? readdirSync(from).filter((f) => f.endsWith(".json")) : [];
for (const f of files) copyFileSync(join(from, f), join(to, f));
console.log(`synced ${files.length} result file(s) into web/data`);
```

- [ ] **Step 3: Write `web/lib/types.ts`**

```ts
export interface ScreenFlag {
  category: string;
  heading: string;
  advice: string;
}

export interface Screen {
  emergency: ScreenFlag[];
  out_of_scope: ScreenFlag[];
  exam_context: boolean;
}

export type ServerEvent =
  | { event: "screen"; data: Screen }
  | { event: "token"; data: { text: string } }
  | { event: "done"; data: { finish_reason: string } }
  | { event: "error"; data: { message: string } };

export interface McNemar {
  wins: number;
  regressions: number;
  both_correct: number;
  both_wrong: number;
  discordant: number;
  p_value: number;
}

export interface EvalReport {
  mode: "constrained" | "generative";
  n: number;
  base_accuracy: number;
  tuned_accuracy: number;
  base_unparseable: number;
  tuned_unparseable: number;
  mcnemar: McNemar;
  by_subject: Record<string, { n: number; base: number; tuned: number }>;
}

export interface EvalFile {
  test_set: string;
  base: string;
  adapter: string;
  device: string;
  timing: Record<string, number>;
  reports: { constrained: EvalReport; generative: EvalReport };
  samples: { id: string; answer: string; base: string; tuned: string }[];
}

export interface TrainStats {
  train_runtime_seconds: number;
  train_loss: number;
  examples: number;
  max_seq: number;
  batch_size: number;
  grad_accum: number;
  rank: number;
}

export interface DataReport {
  train_size: number;
  val_size: number;
  holdout_pool: number;
  decontaminated_removed: number;
  decontaminated_from: number;
  mix: Record<string, number>;
  mcq_fraction: number;
  reasoning_fraction: number;
  seed: number;
}

export interface RunInfo {
  trained_on: string;
  base: string;
  lora_rank: number;
  gpu: string;
  kaggle_kernel: string;
}

export interface LossPoint {
  step: number;
  loss: number;
}
```

- [ ] **Step 4: Write `web/app/globals.css`**

```css
:root {
  --paper: #f2f5f1;
  --surface: #ffffff;
  --ink: #16283a;
  --ink-2: #4a5a68;
  --theatre: #2f5d4e;
  --theatre-data: #1f7a55;
  --rule: #c9d8cf;
  --alarm: #c8102e;
  --caution: #8a5a00;
  --measure: 44rem;
  --radius: 6px;
  --step--1: 0.875rem;  /* 14 */
  --step-0: 1.0625rem;  /* 17 */
  --step-1: 1.3125rem;  /* 21 */
  --step-2: 1.6875rem;  /* 27 */
}

* { box-sizing: border-box; }

html { background: var(--paper); color: var(--ink); }

body {
  margin: 0;
  font-family: var(--font-sans), system-ui, sans-serif;
  font-size: var(--step-0);
  line-height: 1.55;
  background: var(--paper);
}

h1, h2, h3 { line-height: 1.2; margin: 0 0 0.5em; font-weight: 700; }
h1 { font-size: var(--step-2); }
h2 { font-size: var(--step-1); }
h3 { font-size: var(--step-0); }
p { margin: 0 0 1em; }
a { color: var(--theatre); text-underline-offset: 0.15em; }

:focus-visible { outline: 3px solid var(--theatre); outline-offset: 2px; }

.visually-hidden {
  position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0;
  overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
}

/* ---- masthead -------------------------------------------------------- */
.masthead { border-bottom: 1px solid var(--rule); background: var(--paper); }
.masthead-inner {
  max-width: var(--measure); margin: 0 auto; padding: 14px 16px;
  display: flex; flex-wrap: wrap; align-items: baseline;
  justify-content: space-between; gap: 8px 24px;
}
.product { margin: 0; font-weight: 700; color: var(--theatre); }
.views { display: flex; gap: 20px; }
.views a {
  color: var(--ink-2); text-decoration: none; padding: 4px 0;
  border-bottom: 2px solid transparent;
}
.views a[aria-current="page"] { color: var(--ink); border-bottom-color: var(--theatre); }

.page { max-width: var(--measure); margin: 0 auto; padding: 24px 16px 160px; }
.lede { color: var(--ink-2); max-width: 36rem; }

/* ---- ask --------------------------------------------------------------- */
.standing-note {
  border-left: 3px solid var(--theatre); padding: 2px 0 2px 12px;
  color: var(--ink-2); font-size: var(--step--1); margin-bottom: 28px;
}
.examples { list-style: none; padding: 0; margin: 16px 0 0; display: grid; gap: 8px; }
.examples button {
  width: 100%; text-align: left; font: inherit; color: var(--ink);
  background: transparent; border: 1px solid var(--rule); border-radius: var(--radius);
  padding: 10px 12px; cursor: pointer;
}
.examples button:hover { border-color: var(--theatre); }

.thread { list-style: none; padding: 0; margin: 0; display: grid; gap: 28px; }
.speaker { font-size: var(--step--1); color: var(--ink-2); margin: 0 0 4px; }
.said { white-space: pre-wrap; margin: 0; }
.answer {
  font-family: var(--font-serif), Georgia, serif; font-size: 1.125rem;
  line-height: 1.62;
}
.answer p, .answer ul, .answer ol { margin: 0 0 0.8em; }
.pending { color: var(--ink-2); font-style: italic; margin: 0; }
.failed { color: var(--ink); border-left: 3px solid var(--ink-2); padding-left: 12px; }

/* ---- safety ------------------------------------------------------------ */
.safety { display: grid; gap: 10px; margin-bottom: 16px; }
.alarm {
  position: relative; background: var(--surface); border-radius: var(--radius);
  padding: 16px 16px 14px; overflow: hidden;
}
.alarm::before {
  content: ""; position: absolute; inset: 0 0 auto 0; height: 4px;
  background: var(--alarm); transform-origin: left;
  animation: alarm-strip 220ms ease-out;
}
.alarm h3 { color: var(--alarm); margin: 0 0 6px; font-size: var(--step-0); }
.alarm p { margin: 0; }
.caution {
  border-left: 3px solid var(--caution); padding: 2px 0 2px 12px;
  font-size: var(--step--1);
}
.caution strong { color: var(--caution); }
@keyframes alarm-strip { from { transform: scaleX(0); } to { transform: scaleX(1); } }

/* ---- composer ---------------------------------------------------------- */
.composer {
  position: fixed; left: 0; right: 0; bottom: 0;
  background: linear-gradient(to top, var(--paper) 70%, transparent);
  padding: 20px 16px 16px;
}
.composer-inner { max-width: var(--measure); margin: 0 auto; }
.composer textarea {
  width: 100%; font: inherit; color: var(--ink); background: var(--surface);
  border: 1px solid var(--rule); border-radius: var(--radius);
  padding: 10px 12px; resize: vertical; min-height: 3.2em;
}
.composer-row { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }
.count { font-size: var(--step--1); color: var(--ink-2); }
.count.near { color: var(--caution); }
button.primary {
  font: inherit; font-weight: 700; color: var(--surface); background: var(--theatre);
  border: 0; border-radius: var(--radius); padding: 8px 20px; cursor: pointer;
}
button.primary:disabled { opacity: 0.45; cursor: default; }

/* ---- report ------------------------------------------------------------ */
.sheet {
  background: var(--surface); border: 1px solid var(--rule); border-radius: var(--radius);
  padding: 20px 16px; margin: 20px 0 32px;
}
.report-head {
  display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px;
  margin: 0 0 16px; font-size: var(--step--1);
}
.report-head dt { color: var(--ink-2); }
.report-head dd { margin: 0; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 8px 10px 8px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-size: var(--step--1); color: var(--ink-2); font-weight: 400; }
td.num, th.num { text-align: right; }
.flag-better { color: var(--theatre); font-weight: 700; }
.flag-worse { color: var(--caution); font-weight: 700; }
.footnote { font-size: var(--step--1); color: var(--ink-2); margin: 12px 0 0; }

/* ---- chart ------------------------------------------------------------- */
.chart { margin: 0; }
.chart svg { width: 100%; height: auto; display: block; overflow: visible; }
.chart .grid { stroke: var(--rule); stroke-width: 1; }
.chart .line {
  fill: none; stroke: var(--theatre-data); stroke-width: 2;
  stroke-linejoin: round; stroke-linecap: round;
}
.chart .end-dot { fill: var(--theatre-data); stroke: var(--surface); stroke-width: 2; }
.chart .crosshair line { stroke: var(--ink-2); stroke-width: 1; }
.chart .tick, .chart .axis-label, .chart .direct-label {
  fill: var(--ink-2); font-size: 13px; font-variant-numeric: tabular-nums;
}
.chart .direct-label { fill: var(--ink); }
.readout { font-size: var(--step--1); min-height: 1.55em; margin: 4px 0 0; font-variant-numeric: tabular-nums; }
.chart figcaption { font-size: var(--step--1); color: var(--ink-2); margin-top: 8px; }
.numbers { margin-top: 8px; font-size: var(--step--1); }

/* ---- method ------------------------------------------------------------ */
.method { font-family: var(--font-serif), Georgia, serif; font-size: 1.125rem; line-height: 1.62; }
.method h2 { font-family: var(--font-sans), system-ui, sans-serif; margin-top: 1.6em; }
.method ol.pipeline { padding-left: 1.4em; }
.method ol.pipeline > li { margin-bottom: 1.4em; }
.method ol.pipeline > li::marker { font-family: var(--font-sans), system-ui, sans-serif; color: var(--theatre); font-weight: 700; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
```

- [ ] **Step 5: Write the layout, the nav and a placeholder home page**

`web/app/layout.tsx`:

```tsx
import type { Metadata } from "next";
import { Atkinson_Hyperlegible_Next, STIX_Two_Text } from "next/font/google";
import { Nav } from "@/components/Nav";
import "./globals.css";

const sans = Atkinson_Hyperlegible_Next({
  subsets: ["latin"], variable: "--font-sans", display: "swap",
});
const serif = STIX_Two_Text({
  subsets: ["latin"], variable: "--font-serif", display: "swap",
});

export const metadata: Metadata = {
  title: "Medical study assistant",
  description:
    "Qwen3-4B fine-tuned on 17,285 medical exam questions and patient " +
    "conversations, with its evaluation against the model it started from.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${serif.variable}`}>
      <body>
        <header className="masthead">
          <div className="masthead-inner">
            <p className="product">Medical study assistant</p>
            <Nav />
          </div>
        </header>
        <main className="page">{children}</main>
      </body>
    </html>
  );
}
```

If `next build` reports that `Atkinson_Hyperlegible_Next` is not a known font,
import `Atkinson_Hyperlegible` instead (with `weight: ["400", "700"]`), keep the
variable name, and say so in the report.

`web/components/Nav.tsx`:

```tsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const VIEWS = [
  { href: "/", label: "Ask" },
  { href: "/results", label: "Results" },
  { href: "/method", label: "Method" },
];

export function Nav() {
  const path = usePathname();
  return (
    <nav aria-label="Views" className="views">
      {VIEWS.map((v) => (
        <Link key={v.href} href={v.href}
              aria-current={path === v.href ? "page" : undefined}>
          {v.label}
        </Link>
      ))}
    </nav>
  );
}
```

`web/app/page.tsx` (Task 4 replaces the body):

```tsx
export default function AskPage() {
  return <h1>Ask a medical question</h1>;
}
```

- [ ] **Step 6: Build it**

```bash
cd web && npm run build && cd ..
```

Expected: `synced 4 result file(s) into web/data`, then a successful Next.js
build listing the `/` route. There are no unit tests yet; `npm test` is wired up
for Task 4.

- [ ] **Step 7: Commit**

```bash
git add web/ results/run1/run.json .gitignore
git commit -m "Scaffold the web app and its design system

The palette borrows operating-theatre green, adopted in surgery because
it complements red: on a green field a little red cannot be missed, so
emergency guidance can be unmistakable without being loud. Red is
reserved for it and appears nowhere else.

Atkinson Hyperlegible, made by the Braille Institute for low-vision
readers, carries the interface. STIX Two, made for scientific and
medical publishing, carries the model's answers, so who is speaking is
set by typeface rather than by chat bubbles.

The pages read results/run1, copied in before dev and build, so the
repository keeps one record of what the run measured."
```

---

### Task 4: The Ask view

**Files:**
- Create: `web/lib/sse.ts`, `web/lib/sse.test.ts`
- Create: `web/lib/answer.ts`, `web/lib/answer.test.ts`
- Create: `web/app/api/chat/route.ts`
- Create: `web/components/Ask.tsx`, `web/components/SafetyPanel.tsx`,
  `web/components/Answer.tsx`
- Modify: `web/app/page.tsx`

**Interfaces:**
- Consumes: `ServerEvent`, `Screen` from `lib/types.ts`; `POST /chat` via
  `MODEL_URL`.
- Produces:
  - `lib/sse.ts`: `class SSEParser { feed(chunk: string): ServerEvent[] }`
  - `lib/answer.ts`: `parseAnswer(text: string): Block[]` where
    `Block = { kind: "p"; spans: Span[] } | { kind: "ul" | "ol"; items: Span[][] }`
    and `Span = { text: string; bold: boolean }`
  - `POST /api/chat` — same body as the server's `/chat`; streams its events
    through, or answers `{detail}` with the upstream status, or 502 when the
    server is unreachable
  - DOM hooks for Task 6: the question box is labelled "Your question"; the
    submit button's name is exactly "Ask"; each model turn is `.turn-model`, its
    text `.answer`; emergency sections have `role="alert"`

- [ ] **Step 1: Write the failing tests**

`web/lib/sse.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { SSEParser } from "./sse";

const screen = 'event: screen\ndata: {"emergency":[],"out_of_scope":[],"exam_context":false}\n\n';

describe("SSEParser", () => {
  it("parses one complete event", () => {
    expect(new SSEParser().feed('event: token\ndata: {"text":"Hi"}\n\n'))
      .toEqual([{ event: "token", data: { text: "Hi" } }]);
  });

  it("holds a partial event until the rest arrives", () => {
    const p = new SSEParser();
    expect(p.feed('event: token\ndata: {"te')).toEqual([]);
    expect(p.feed('xt":"Hi"}\n\n')).toEqual([{ event: "token", data: { text: "Hi" } }]);
  });

  it("parses several events from one chunk", () => {
    const out = new SSEParser().feed(screen + 'event: done\ndata: {"finish_reason":"stop"}\n\n');
    expect(out.map((e) => e.event)).toEqual(["screen", "done"]);
  });

  it("ignores keep-alive comments", () => {
    expect(new SSEParser().feed(": ping\n\n")).toEqual([]);
  });

  it("skips a malformed event instead of throwing", () => {
    expect(new SSEParser().feed("event: token\ndata: {not json\n\n")).toEqual([]);
  });

  it("handles CRLF line endings split across chunks", () => {
    const p = new SSEParser();
    expect(p.feed('event: token\r\ndata: {"text":"a"}\r')).toEqual([]);
    expect(p.feed("\n\r\n")).toEqual([{ event: "token", data: { text: "a" } }]);
  });
});
```

`web/lib/answer.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { parseAnswer } from "./answer";

describe("parseAnswer", () => {
  it("splits paragraphs on blank lines", () => {
    expect(parseAnswer("One.\n\nTwo.").map((b) => b.kind)).toEqual(["p", "p"]);
  });

  it("reads bold spans", () => {
    const [block] = parseAnswer("It is **not** urgent.");
    expect(block).toEqual({ kind: "p", spans: [
      { text: "It is ", bold: false }, { text: "not", bold: true },
      { text: " urgent.", bold: false }] });
  });

  it("reads bulleted and numbered lists", () => {
    expect(parseAnswer("- a\n- b")[0]).toMatchObject({ kind: "ul" });
    expect(parseAnswer("1. a\n2. b")[0]).toMatchObject({ kind: "ol" });
  });

  it("drops stray think tags", () => {
    expect(JSON.stringify(parseAnswer("<think>\n\n</think>\n\nAnswer: B"))).not.toContain("think");
  });

  it("never returns markup as markup", () => {
    const [block] = parseAnswer("<img src=x onerror=alert(1)>");
    expect(block).toEqual({ kind: "p", spans: [{ text: "<img src=x onerror=alert(1)>", bold: false }] });
  });
});
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `cd web && npm test; cd ..`
Expected: FAIL — cannot resolve `./sse` and `./answer`.

- [ ] **Step 3: Write `web/lib/sse.ts`**

```ts
import type { ServerEvent } from "./types";

/** Incremental server-sent-events parser. Feed it chunks as they arrive; it
 * returns the complete events and keeps any partial one for the next chunk.
 * A network chunk boundary can fall anywhere, including mid-line. */
export class SSEParser {
  private buffer = "";

  feed(chunk: string): ServerEvent[] {
    this.buffer = (this.buffer + chunk).replace(/\r\n/g, "\n");
    const out: ServerEvent[] = [];
    let sep: number;
    while ((sep = this.buffer.indexOf("\n\n")) !== -1) {
      const block = this.buffer.slice(0, sep);
      this.buffer = this.buffer.slice(sep + 2);
      const parsed = parseBlock(block);
      if (parsed) out.push(parsed);
    }
    return out;
  }
}

function parseBlock(block: string): ServerEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const i = line.indexOf(":");
    const field = i === -1 ? line : line.slice(0, i);
    const value = i === -1 ? "" : line.slice(i + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  if (data.length === 0) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) } as ServerEvent;
  } catch {
    return null;
  }
}
```

- [ ] **Step 4: Write `web/lib/answer.ts`**

```ts
export type Span = { text: string; bold: boolean };
export type Block =
  | { kind: "p"; spans: Span[] }
  | { kind: "ul" | "ol"; items: Span[][] };

const BULLET = /^\s*[-*]\s+/;
const NUMBERED = /^\s*\d+[.)]\s+/;

/** A deliberately small reading of the model's markdown: paragraphs, bold,
 * and lists. It builds data, never HTML, so nothing the model writes can
 * become markup on the page. */
export function parseAnswer(text: string): Block[] {
  const clean = text.replace(/<\/?think>/g, "").trim();
  if (!clean) return [];
  return clean.split(/\n\s*\n/).map((chunk) => {
    const lines = chunk.split("\n").filter((l) => l.trim());
    if (lines.length && lines.every((l) => BULLET.test(l)))
      return { kind: "ul", items: lines.map((l) => spans(l.replace(BULLET, ""))) };
    if (lines.length && lines.every((l) => NUMBERED.test(l)))
      return { kind: "ol", items: lines.map((l) => spans(l.replace(NUMBERED, ""))) };
    return { kind: "p", spans: spans(lines.join("\n")) };
  });
}

function spans(line: string): Span[] {
  return line
    .split(/(\*\*[^*]+\*\*)/)
    .filter(Boolean)
    .map((part) =>
      part.startsWith("**") && part.endsWith("**") && part.length > 4
        ? { text: part.slice(2, -2), bold: true }
        : { text: part, bold: false });
}
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `cd web && npm test; cd ..`
Expected: PASS — 6 in `sse.test.ts`, 5 in `answer.test.ts`.

- [ ] **Step 6: Write the proxy, `web/app/api/chat/route.ts`**

```ts
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MODEL_URL = process.env.MODEL_URL ?? "http://127.0.0.1:8008";
const MAX_BODY = 200_000;

export async function POST(req: Request) {
  const raw = await req.text();
  if (raw.length > MAX_BODY)
    return Response.json({ detail: "That conversation is too long to send. Start a new one." }, { status: 413 });

  let upstream: Response;
  try {
    upstream = await fetch(`${MODEL_URL}/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: raw,
      signal: req.signal,
    });
  } catch {
    return Response.json(
      { detail: "The model server isn't reachable. If you're running this locally, start it with: bash serving/start_local.sh" },
      { status: 502 });
  }

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.json().then((j) => j?.detail).catch(() => null);
    return Response.json(
      { detail: typeof detail === "string" ? detail : `The model server answered ${upstream.status}.` },
      { status: upstream.status });
  }

  return new Response(upstream.body, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "x-accel-buffering": "no",
    },
  });
}
```

- [ ] **Step 7: Write the three components**

`web/components/SafetyPanel.tsx`:

```tsx
import type { Screen } from "@/lib/types";

export function SafetyPanel({ screen }: { screen: Screen }) {
  if (!screen.emergency.length && !screen.out_of_scope.length) return null;
  return (
    <div className="safety">
      {screen.emergency.map((f) => (
        <section key={f.category} className="alarm" role="alert">
          <h3>{f.heading}</h3>
          <p>{f.advice}</p>
        </section>
      ))}
      {screen.out_of_scope.map((f) => (
        <aside key={f.category} className="caution">
          <p><strong>{f.heading}</strong> {f.advice}</p>
        </aside>
      ))}
    </div>
  );
}
```

`web/components/Answer.tsx`:

```tsx
import { parseAnswer, type Span } from "@/lib/answer";

function Spans({ spans }: { spans: Span[] }) {
  return <>{spans.map((s, i) => (s.bold ? <strong key={i}>{s.text}</strong> : <span key={i}>{s.text}</span>))}</>;
}

export function Answer({ text }: { text: string }) {
  return (
    <div className="answer">
      {parseAnswer(text).map((b, i) =>
        b.kind === "p" ? (
          <p key={i}><Spans spans={b.spans} /></p>
        ) : b.kind === "ul" ? (
          <ul key={i}>{b.items.map((it, j) => <li key={j}><Spans spans={it} /></li>)}</ul>
        ) : (
          <ol key={i}>{b.items.map((it, j) => <li key={j}><Spans spans={it} /></li>)}</ol>
        ))}
    </div>
  );
}
```

`web/components/Ask.tsx`:

```tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { SSEParser } from "@/lib/sse";
import type { Screen } from "@/lib/types";
import { Answer } from "./Answer";
import { SafetyPanel } from "./SafetyPanel";

const MAX_CHARS = 4000;

const EXAMPLES = [
  "What's the difference between type 1 and type 2 diabetes?",
  "How do beta blockers lower blood pressure?",
  "A 23-year-old woman who is 22 weeks pregnant has burning when she urinates. Which antibiotic is safest, and why?",
  "Deficiency of which vitamin causes scurvy?\nA. Vitamin A\nB. Vitamin C\nC. Vitamin D\nD. Vitamin K",
];

type ModelTurn = {
  role: "assistant";
  content: string;
  screen: Screen | null;
  status: "waiting" | "streaming" | "done" | "error";
  error?: string;
};
type Turn = { role: "user"; content: string } | ModelTurn;

export function Ask() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [announcement, setAnnouncement] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  function updateModel(patch: (t: ModelTurn) => ModelTurn) {
    setTurns((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role === "assistant") next[next.length - 1] = patch(last);
      return next;
    });
  }

  async function ask(question: string) {
    const text = question.trim();
    if (!text || busy) return;
    const history = turns.filter((t) => t.role === "user" || t.status === "done");
    const messages = [...history, { role: "user" as const, content: text }]
      .map((t) => ({ role: t.role, content: t.content }));

    setTurns([...turns, { role: "user", content: text },
              { role: "assistant", content: "", screen: null, status: "waiting" }]);
    setDraft("");
    setBusy(true);
    setAnnouncement("");
    const controller = new AbortController();
    abortRef.current = controller;
    let finished = false;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        const detail = await res.json().then((j) => j?.detail).catch(() => null);
        finished = true;
        updateModel((t) => ({ ...t, status: "error",
          error: typeof detail === "string" ? detail : `The server answered ${res.status}.` }));
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      const parser = new SSEParser();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        for (const ev of parser.feed(decoder.decode(value, { stream: true }))) {
          if (ev.event === "screen") updateModel((t) => ({ ...t, screen: ev.data }));
          else if (ev.event === "token")
            updateModel((t) => ({ ...t, status: "streaming", content: t.content + ev.data.text }));
          else if (ev.event === "done") { finished = true; updateModel((t) => ({ ...t, status: "done" })); }
          else if (ev.event === "error") {
            finished = true;
            updateModel((t) => ({ ...t, status: "error", error: ev.data.message }));
          }
        }
      }
      if (!finished)
        updateModel((t) => ({ ...t, status: "error", error: "The answer was cut off before it finished. Ask again to retry." }));
      setAnnouncement("Answer ready.");
    } catch (err) {
      if ((err as Error).name !== "AbortError")
        updateModel((t) => ({ ...t, status: "error",
          error: "The connection dropped before the answer finished. Ask again to retry." }));
    } finally {
      setBusy(false);
      abortRef.current = null;
      inputRef.current?.focus();
    }
  }

  const nearLimit = draft.length > MAX_CHARS - 200;

  return (
    <div className="ask">
      <p className="standing-note">
        For learning about medicine. It can be wrong, and it isn&apos;t a doctor. If this is
        an emergency, call your local emergency number now.
      </p>

      {turns.length === 0 ? (
        <div className="empty">
          <h1>Ask a medical question</h1>
          <p className="lede">
            It was fine-tuned on 17,285 exam questions and patient conversations. Try one of
            these, or ask your own.
          </p>
          <ul className="examples">
            {EXAMPLES.map((e) => (
              <li key={e}>
                <button type="button" onClick={() => { setDraft(e); inputRef.current?.focus(); }}>
                  {e.split("\n")[0]}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <ol className="thread" aria-label="Conversation">
          {turns.map((t, i) =>
            t.role === "user" ? (
              <li key={i} className="turn turn-user">
                <p className="speaker">You</p>
                <p className="said">{t.content}</p>
              </li>
            ) : (
              <li key={i} className="turn turn-model">
                {t.screen && <SafetyPanel screen={t.screen} />}
                <p className="speaker">Assistant</p>
                {t.status === "waiting" && <p className="pending">Working on it</p>}
                {t.content && <Answer text={t.content} />}
                {t.status === "error" && <p className="failed">{t.error}</p>}
              </li>
            ))}
        </ol>
      )}

      <p className="visually-hidden" aria-live="polite">{announcement}</p>

      <form className="composer" onSubmit={(e) => { e.preventDefault(); ask(draft); }}>
        <div className="composer-inner">
          <label htmlFor="question" className="visually-hidden">Your question</label>
          <textarea
            id="question" ref={inputRef} value={draft} maxLength={MAX_CHARS} rows={2}
            placeholder="Ask a medical question, or paste an exam question with its options"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(draft); }
            }}
          />
          <div className="composer-row">
            <span className={nearLimit ? "count near" : "count"}>
              {nearLimit ? `${MAX_CHARS - draft.length} characters left` : ""}
            </span>
            <button type="submit" className="primary" disabled={busy || !draft.trim()}>
              {busy ? "Answering" : "Ask"}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
```

`web/app/page.tsx`:

```tsx
import { Ask } from "@/components/Ask";

export default function AskPage() {
  return <Ask />;
}
```

- [ ] **Step 8: Build, then try it against the mock server**

```bash
cd web && npm test && npm run build && cd ..
MOCK_BACKEND=1 PORT=8011 bash serving/start_local.sh > /tmp/medserve.log 2>&1 &
(cd web && MODEL_URL=http://127.0.0.1:8011 npx next start -p 3100 > /tmp/medweb.log 2>&1 &)
until curl -sf http://127.0.0.1:3100 >/dev/null; do sleep 1; done
curl -sN -X POST http://127.0.0.1:3100/api/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"I cant breathe"}]}' | head -3
```

Expected: `event: screen` with a `breathing` flag through the proxy. Then stop
both servers.

- [ ] **Step 9: Commit**

```bash
git add web/
git commit -m "Build the Ask view: streamed answers under their safety screen

The screen arrives before the answer and renders above it: emergency
guidance in the one red on the page, announced to screen readers as an
alert, and quieter notes for questions it shouldn't answer.

The proxy keeps the model server's address off the client and turns an
unreachable server into a message saying how to start it. A stream that
ends without its closing event is shown as cut off rather than
presented as a finished answer.

The model's markdown is read into data, never HTML, so nothing it
writes can become markup on the page."
```

---

### Task 5: The Results view, laid out as a laboratory report

**Files:**
- Create: `web/lib/report.ts`, `web/lib/report.test.ts`
- Create: `web/lib/data.ts`
- Create: `web/components/LossChart.tsx`
- Create: `web/app/results/page.tsx`

**Interfaces:**
- Consumes: `EvalFile`, `TrainStats`, `DataReport`, `RunInfo`, `LossPoint`.
- Produces:
  - `lib/report.ts`: `SIGNIFICANCE = 0.05`;
    `flagFor(tuned, base, p): "better" | "worse" | null`;
    `rowsFromEval(benchmark: string, file: EvalFile): ResultRow[]`;
    `subjectRows(file: EvalFile): SubjectRow[]`;
    `lossPoints(curve: {epoch: number; loss: number}[], totalSteps: number): LossPoint[]`;
    `totalSteps(stats: TrainStats): number`;
    `pct(x)`, `change(pp)`, `pValue(p)`
  - `lib/data.ts`: `readData<T>(name: string): T | null`
  - The Results page renders a "Training" heading and a chart whose accessible
    name begins "Training loss" — Task 6 depends on both

- [ ] **Step 1: Write the failing tests**

`web/lib/report.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { EvalFile } from "./types";
import { change, flagFor, lossPoints, pValue, pct, rowsFromEval, subjectRows, totalSteps } from "./report";

const report = (mode: "constrained" | "generative", base: number, tuned: number, p: number) => ({
  mode, n: 1000, base_accuracy: base, tuned_accuracy: tuned,
  base_unparseable: 0, tuned_unparseable: 0,
  mcnemar: { wins: 60, regressions: 30, both_correct: 500, both_wrong: 410, discordant: 90, p_value: p },
  by_subject: { Anatomy: { n: 120, base: 0.5, tuned: 0.6 }, Pharmacology: { n: 300, base: 0.55, tuned: 0.56 } },
});
const file = { test_set: "t", base: "b", adapter: "a", device: "cuda", timing: {}, samples: [],
  reports: { constrained: report("constrained", 0.52, 0.58, 0.002), generative: report("generative", 0.4, 0.41, 0.6) },
} as EvalFile;

describe("flagFor", () => {
  it("flags a significant gain as better and a significant loss as worse", () => {
    expect(flagFor(0.6, 0.5, 0.01)).toBe("better");
    expect(flagFor(0.4, 0.5, 0.01)).toBe("worse");
  });
  it("does not flag a difference chance could produce", () => {
    expect(flagFor(0.6, 0.5, 0.2)).toBeNull();
    expect(flagFor(0.6, 0.5, 0.05)).toBeNull();
  });
});

describe("rowsFromEval", () => {
  it("makes one row per scoring mode with the change in points", () => {
    const rows = rowsFromEval("MedQA-USMLE", file);
    expect(rows.map((r) => r.mode)).toEqual(["constrained", "generative"]);
    expect(rows[0].change).toBeCloseTo(6, 5);
    expect(rows[0].flag).toBe("better");
    expect(rows[1].flag).toBeNull();
  });
});

describe("subjectRows", () => {
  it("lists subjects by how many questions they have", () => {
    expect(subjectRows(file).map((s) => s.subject)).toEqual(["Pharmacology", "Anatomy"]);
  });
});

describe("formatting", () => {
  it("formats percentages, signed changes and p values", () => {
    expect(pct(0.5812)).toBe("58.1%");
    expect(change(6)).toBe("+6.0");
    expect(change(-1.25)).toBe("−1.3");
    expect(pValue(0.0004)).toBe("< 0.001");
    expect(pValue(0.0421)).toBe("0.042");
  });
});

describe("training", () => {
  it("derives the step count from the examples and the effective batch", () => {
    expect(totalSteps({ examples: 17285, batch_size: 8, grad_accum: 4 } as never)).toBe(541);
  });
  it("turns epochs into steps", () => {
    expect(lossPoints([{ epoch: 0.5, loss: 1.8 }], 541)).toEqual([{ step: 271, loss: 1.8 }]);
  });
});
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `cd web && npm test; cd ..` — FAIL, cannot resolve `./report`.

- [ ] **Step 3: Write `web/lib/report.ts`**

```ts
import type { EvalFile, LossPoint, TrainStats } from "./types";

export const SIGNIFICANCE = 0.05;
export type Flag = "better" | "worse" | null;

export interface ResultRow {
  benchmark: string;
  mode: "constrained" | "generative";
  n: number;
  tuned: number;
  base: number;
  change: number;
  p: number;
  flag: Flag;
  unparseable: { base: number; tuned: number };
}

export interface SubjectRow { subject: string; n: number; base: number; tuned: number }

/** A flag only where McNemar says the difference is unlikely to be chance.
 * Everything else is reported as within what chance produces. */
export function flagFor(tuned: number, base: number, p: number): Flag {
  if (p >= SIGNIFICANCE || tuned === base) return null;
  return tuned > base ? "better" : "worse";
}

export function rowsFromEval(benchmark: string, file: EvalFile): ResultRow[] {
  return (["constrained", "generative"] as const).map((mode) => {
    const r = file.reports[mode];
    return {
      benchmark, mode, n: r.n,
      tuned: r.tuned_accuracy, base: r.base_accuracy,
      change: (r.tuned_accuracy - r.base_accuracy) * 100,
      p: r.mcnemar.p_value,
      flag: flagFor(r.tuned_accuracy, r.base_accuracy, r.mcnemar.p_value),
      unparseable: { base: r.base_unparseable, tuned: r.tuned_unparseable },
    };
  });
}

export function subjectRows(file: EvalFile): SubjectRow[] {
  return Object.entries(file.reports.constrained.by_subject)
    .map(([subject, s]) => ({ subject, ...s }))
    .sort((a, b) => b.n - a.n);
}

export function totalSteps(stats: TrainStats): number {
  return Math.ceil(stats.examples / (stats.batch_size * stats.grad_accum));
}

export function lossPoints(curve: { epoch: number; loss: number }[], steps: number): LossPoint[] {
  return curve.map((c) => ({ step: Math.round(c.epoch * steps), loss: c.loss }));
}

export const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
export const change = (pp: number) => `${pp >= 0 ? "+" : "−"}${Math.abs(pp).toFixed(1)}`;
export const pValue = (p: number) => (p < 0.001 ? "< 0.001" : p.toFixed(3));
```

`web/lib/data.ts`:

```ts
import fs from "node:fs";
import path from "node:path";

/** Read a synced result file. Missing is normal -- the evaluation files only
 * exist once the evaluation has run -- so it returns null rather than throwing. */
export function readData<T>(name: string): T | null {
  const file = path.join(process.cwd(), "data", name);
  if (!fs.existsSync(file)) return null;
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
}
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `cd web && npm test; cd ..` — PASS, 8 in `report.test.ts`.

- [ ] **Step 5: Write `web/components/LossChart.tsx`**

The chart follows the dataviz rules: one series so no legend, a 2px line in the
validated data green, a 4px end-dot with a 2px surface ring, 1px solid gridlines,
labels in ink tokens, only the first and last values labelled directly, a
crosshair that snaps to the nearest step with the same readout on keyboard focus,
and every value also reachable in the table.

```tsx
"use client";

import { useId, useMemo, useState } from "react";
import type { LossPoint } from "@/lib/types";

const W = 640;
const H = 260;
const M = { top: 16, right: 56, bottom: 48, left: 44 };
const Y_MIN = 1.5;
const Y_MAX = 3.0;
const Y_TICKS = [1.5, 2.0, 2.5, 3.0];

export function LossChart({ points, steps }: { points: LossPoint[]; steps: number }) {
  const [active, setActive] = useState<number | null>(null);
  const titleId = useId();
  const x = (s: number) => M.left + (s / steps) * (W - M.left - M.right);
  const y = (l: number) => M.top + (1 - (l - Y_MIN) / (Y_MAX - Y_MIN)) * (H - M.top - M.bottom);
  const path = useMemo(
    () => points.map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.loss).toFixed(1)}`).join(""),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [points, steps]);
  const xTicks = Array.from({ length: Math.floor(steps / 100) + 1 }, (_, i) => i * 100);
  const first = points[0];
  const last = points[points.length - 1];
  const a = active === null ? null : points[active];

  function nearest(clientX: number, rect: DOMRect) {
    const svgX = ((clientX - rect.left) / rect.width) * W;
    let best = 0;
    for (let i = 1; i < points.length; i++)
      if (Math.abs(x(points[i].step) - svgX) < Math.abs(x(points[best].step) - svgX)) best = i;
    return best;
  }

  return (
    <figure className="chart">
      <svg
        viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby={titleId} tabIndex={0}
        onPointerMove={(e) => setActive(nearest(e.clientX, e.currentTarget.getBoundingClientRect()))}
        onPointerLeave={() => setActive(null)}
        onFocus={() => setActive((v) => v ?? points.length - 1)}
        onBlur={() => setActive(null)}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") setActive((v) => Math.min((v ?? -1) + 1, points.length - 1));
          if (e.key === "ArrowLeft") setActive((v) => Math.max((v ?? points.length) - 1, 0));
        }}
      >
        <title id={titleId}>
          {`Training loss by step. It falls from ${first.loss.toFixed(2)} to ${last.loss.toFixed(2)}, and levels off after about step 150.`}
        </title>
        {Y_TICKS.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} className="grid" />
            <text x={M.left - 8} y={y(t)} className="tick" textAnchor="end" dominantBaseline="middle">
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={x(t)} y={H - M.bottom + 20} className="tick" textAnchor="middle">{t}</text>
        ))}
        <text x={(M.left + W - M.right) / 2} y={H - 8} className="axis-label" textAnchor="middle">
          Training step
        </text>
        <path d={path} className="line" />
        <circle cx={x(last.step)} cy={y(last.loss)} r={4} className="end-dot" />
        <text x={x(last.step) + 10} y={y(last.loss)} className="direct-label" dominantBaseline="middle">
          {last.loss.toFixed(2)}
        </text>
        <text x={x(first.step) + 10} y={y(first.loss)} className="direct-label" dominantBaseline="middle">
          {first.loss.toFixed(2)}
        </text>
        {a && (
          <g className="crosshair" pointerEvents="none">
            <line x1={x(a.step)} x2={x(a.step)} y1={M.top} y2={H - M.bottom} />
            <circle cx={x(a.step)} cy={y(a.loss)} r={4} className="end-dot" />
          </g>
        )}
      </svg>
      <p className="readout" aria-live="polite">
        {a ? `Step ${a.step}, loss ${a.loss.toFixed(3)}` : " "}
      </p>
      <figcaption>
        Cross-entropy on the answer tokens only, logged every 10 steps. Most of the drop happens in
        the first 150 steps, about 4,800 examples.
      </figcaption>
      <details className="numbers">
        <summary>Show the numbers</summary>
        <div className="table-wrap">
          <table>
            <thead><tr><th className="num">Step</th><th className="num">Loss</th></tr></thead>
            <tbody>
              {points.map((p) => (
                <tr key={p.step}><td className="num">{p.step}</td><td className="num">{p.loss.toFixed(3)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
```

- [ ] **Step 6: Write `web/app/results/page.tsx`**

```tsx
import { LossChart } from "@/components/LossChart";
import { readData } from "@/lib/data";
import { change, lossPoints, pValue, pct, rowsFromEval, subjectRows, totalSteps } from "@/lib/report";
import type { DataReport, EvalFile, RunInfo, TrainStats } from "@/lib/types";

export const metadata = { title: "Results" };

const MODE_LABEL = { constrained: "Answer choice", generative: "Written answer" };
const SOURCE_LABEL: Record<string, string> = {
  medmcqa: "MedMCQA exam questions", medqa: "MedQA-USMLE exam questions",
  medical_o1: "worked clinical reasoning examples", chatdoctor: "patient conversations",
};

export default function ResultsPage() {
  const medqa = readData<EvalFile>("eval_medqa.json");
  const medmcqa = readData<EvalFile>("eval_medmcqa.json");
  const stats = readData<TrainStats>("train_stats.json");
  const data = readData<DataReport>("data_report.json");
  const run = readData<RunInfo>("run.json");
  const curve = readData<{ epoch: number; loss: number }[]>("loss_curve.json");

  const rows = [
    ...(medqa ? rowsFromEval("MedQA-USMLE", medqa) : []),
    ...(medmcqa ? rowsFromEval("MedMCQA", medmcqa) : []),
  ];
  const subjects = medmcqa ? subjectRows(medmcqa) : [];
  const steps = stats ? totalSteps(stats) : 0;

  return (
    <>
      <h1>Results</h1>
      <p className="lede">
        How the fine-tuned model compares with the model it started from, on exam questions it
        never saw during training.
      </p>

      <section className="sheet" aria-labelledby="report-title">
        <h2 id="report-title">Benchmark report</h2>
        <dl className="report-head">
          <dt>Specimen</dt>
          <dd>{run ? `${run.base} with a LoRA adapter (rank ${run.lora_rank}), trained ${new Date(run.trained_on).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" })}` : "Qwen3-4B with a LoRA adapter"}</dd>
          <dt>Reference</dt>
          <dd>Qwen3-4B as released, given the same prompts and the same greedy decoding</dd>
          <dt>Held out</dt>
          <dd>{data ? `${data.holdout_pool.toLocaleString("en-GB")} questions, checked against the training data for duplicates` : "Two exam benchmarks, never used in training"}</dd>
        </dl>

        {rows.length === 0 ? (
          <p>
            Not measured yet. Training finished; the comparison against the base model runs next.
            Until then this page shows only what training measured.
          </p>
        ) : (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Test</th><th className="num">Questions</th><th className="num">Result</th>
                    <th className="num">Reference</th><th className="num">Change</th>
                    <th className="num">p</th><th>Flag</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={`${r.benchmark}-${r.mode}`}>
                      <td>{r.benchmark}<br /><span className="footnote">{MODE_LABEL[r.mode]}</span></td>
                      <td className="num">{r.n.toLocaleString("en-GB")}</td>
                      <td className="num">{pct(r.tuned)}</td>
                      <td className="num">{pct(r.base)}</td>
                      <td className="num">{change(r.change)}</td>
                      <td className="num">{pValue(r.p)}</td>
                      <td>{r.flag === "better" ? <span className="flag-better">Better</span>
                         : r.flag === "worse" ? <span className="flag-worse">Worse</span> : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="footnote">
              A flag means the difference is unlikely to be chance (McNemar&apos;s exact test,
              p &lt; 0.05). A change without a flag is within what chance produces. Answer choice
              compares the model&apos;s scores for A to D directly; written answer lets it reason
              first, then reads the letter it states.
            </p>
          </>
        )}
      </section>

      {subjects.length > 0 && (
        <section aria-labelledby="subjects-title">
          <h2 id="subjects-title">MedMCQA by subject</h2>
          <p className="lede">Answer choice scoring. Small subjects move a lot by chance; read them with their counts.</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Subject</th><th className="num">Questions</th><th className="num">Result</th>
                    <th className="num">Reference</th><th className="num">Change</th></tr>
              </thead>
              <tbody>
                {subjects.map((s) => (
                  <tr key={s.subject}>
                    <td>{s.subject}</td><td className="num">{s.n.toLocaleString("en-GB")}</td>
                    <td className="num">{pct(s.tuned)}</td><td className="num">{pct(s.base)}</td>
                    <td className="num">{change((s.tuned - s.base) * 100)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section aria-labelledby="training-title">
        <h2 id="training-title">Training</h2>
        {stats && data ? (
          <p>
            {stats.examples.toLocaleString("en-GB")} examples, one pass, in{" "}
            {(stats.train_runtime_seconds / 3600).toFixed(1)} hours on one {run?.gpu ?? "GPU"}.{" "}
            {Object.entries(data.mix).sort((a, b) => b[1] - a[1])
              .map(([k, v]) => `${v.toLocaleString("en-GB")} ${SOURCE_LABEL[k] ?? k}`)
              .join(", ")}.
          </p>
        ) : null}
        {curve && stats ? <LossChart points={lossPoints(curve, steps)} steps={steps} /> : (
          <p>The training record is missing. Run <code>npm run dev</code> from <code>web/</code>, which copies it in.</p>
        )}
      </section>
    </>
  );
}
```

- [ ] **Step 7: Build and look at it**

```bash
cd web && npm test && npm run build && cd ..
```

Expected: tests pass; the build lists `/results`. Then start `next start` and
open `/results` in a browser: the report shows its "Not measured yet" state, and
the Training section shows the chart. Confirm by eye that the end label, the
first-value label and the axis ticks do not collide, at 1280px and at 360px.

- [ ] **Step 8: Commit**

```bash
git add web/
git commit -m "Lay the results out as a laboratory report

Each benchmark is a test, the fine-tuned model is the specimen and the
base model is the reference. A flag appears only where McNemar says a
difference is unlikely to be chance, and the page says in words that
an unflagged change is within what chance produces.

Until the evaluation has run, the report says so plainly and shows only
what training measured: the data, the runtime, and the loss curve,
which levels off after about 4,800 examples.

The chart has one series and so no legend, labels only its first and
last values, and puts every value in a table as well as behind the
crosshair."
```

---

### Task 6: The Method view, end-to-end tests, and a README

**Files:**
- Create: `web/app/method/page.tsx`
- Create: `web/playwright.config.ts`, `web/e2e/app.spec.ts`
- Create: `README.md`

**Interfaces:**
- Consumes: everything above; the DOM hooks named in Tasks 4 and 5.
- Produces: `npm run e2e` passing against the mock server.

- [ ] **Step 1: Write the end-to-end tests**

`web/playwright.config.ts`:

```ts
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  use: { baseURL: "http://127.0.0.1:3100" },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
  webServer: [
    {
      command: "cd .. && MOCK_BACKEND=1 PORT=8009 bash serving/start_local.sh",
      url: "http://127.0.0.1:8009/health",
      timeout: 60_000,
    },
    {
      command: "MODEL_URL=http://127.0.0.1:8009 npx next dev -p 3100",
      url: "http://127.0.0.1:3100",
      timeout: 120_000,
    },
  ],
});
```

`web/e2e/app.spec.ts`:

```ts
import { expect, test, type Page } from "@playwright/test";

async function ask(page: Page, text: string) {
  await page.goto("/");
  await page.getByLabel("Your question").fill(text);
  await page.getByRole("button", { name: "Ask", exact: true }).click();
}

test("a question gets a streamed answer", async ({ page }) => {
  await ask(page, "How do beta blockers work?");
  await expect(page.locator(".turn-model .answer")).toContainText("mock backend");
});

test("emergency guidance appears above the answer", async ({ page }) => {
  await ask(page, "My dad's face is drooping and he can't lift his arm");
  const alarm = page.getByRole("alert");
  await expect(alarm).toContainText("stroke");
  const answer = page.locator(".turn-model .answer");
  await expect(answer).toBeVisible();
  const [a, b] = [await alarm.boundingBox(), await answer.boundingBox()];
  expect(a!.y).toBeLessThan(b!.y);
});

test("an exam vignette gets no emergency guidance", async ({ page }) => {
  await ask(page, "A 60-year-old man presents with crushing chest pain radiating to his jaw. " +
                  "Which of the following is the most likely diagnosis?");
  await expect(page.locator(".turn-model .answer")).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("the results page shows what training measured", async ({ page }) => {
  await page.goto("/results");
  await expect(page.getByRole("heading", { name: "Training" })).toBeVisible();
  await expect(page.getByRole("img", { name: /^Training loss/ })).toBeVisible();
});

test("the method page explains the pipeline in order", async ({ page }) => {
  await page.goto("/method");
  await expect(page.locator("ol.pipeline > li")).toHaveCount(5);
});
```

- [ ] **Step 2: Run them and confirm the Method test fails**

```bash
cd web && npx playwright install chromium && npm run e2e; cd ..
```

Expected: the four Ask and Results tests pass; `the method page explains the
pipeline in order` fails because `/method` does not exist yet.

- [ ] **Step 3: Write `web/app/method/page.tsx`**

```tsx
import { readData } from "@/lib/data";
import type { DataReport, TrainStats } from "@/lib/types";

export const metadata = { title: "Method" };

export default function MethodPage() {
  const data = readData<DataReport>("data_report.json");
  const stats = readData<TrainStats>("train_stats.json");
  const n = (x: number | undefined) => (x === undefined ? "–" : x.toLocaleString("en-GB"));

  return (
    <article className="method">
      <h1>Method</h1>
      <p>
        This is Qwen3-4B, fine-tuned on medical exam questions and patient conversations, and
        compared with the model it started from on questions it never saw. It is an educational
        tool. It is not a medical device, it can be wrong, and it cannot examine anyone.
      </p>

      <ol className="pipeline">
        <li>
          <h2>Four sources, filtered</h2>
          <p>
            MedMCQA and MedQA-USMLE supply exam questions, one with written explanations; a set of
            worked clinical reasoning supplies step-by-step thinking; and ChatDoctor supplies
            patient conversations. Half of MedMCQA was dropped before training: a quarter had no
            usable explanation, and a third were marked multiple-choice while carrying a single
            answer, which is its known noisy portion.
          </p>
        </li>
        <li>
          <h2>Held out, and checked</h2>
          <p>
            {n(data?.holdout_pool)} benchmark questions were set aside before any training data
            was chosen. Every training question was normalised and compared against them, and
            against each other; {n(data?.decontaminated_removed)} of{" "}
            {n(data?.decontaminated_from)} were removed. A separate check over 28,000 of the same
            exam questions found none that overlapped the benchmarks, so the removals were
            duplicates within the training data and the check is insurance. It runs on every
            build. It matches exact text after normalising case and punctuation, so it cannot
            catch a question that was paraphrased.
          </p>
        </li>
        <li>
          <h2>Trained on one free GPU</h2>
          <p>
            {n(stats?.examples)} examples, one pass, as a LoRA adapter of rank {stats?.rank ?? "–"} on
            a single Kaggle T4. The loss was computed on the answers only, so the model was never
            trained to reproduce the questions. The maximum length, {stats?.max_seq ?? "–"} tokens,
            came from measuring the data rather than guessing, and a check at step 50 projected the
            run&apos;s length and would have stopped it if it could not finish in time.
          </p>
        </li>
        <li>
          <h2>Scored against the base model</h2>
          <p>
            Both models answer the same prompts with the same greedy decoding. Answer choice
            scoring compares the model&apos;s scores for A to D directly; written answer scoring
            lets it reason and then reads the letter it states, and an answer it never states
            counts as wrong. McNemar&apos;s exact test on the paired results says whether a
            difference could be chance.
          </p>
        </li>
        <li>
          <h2>Screened before it answers</h2>
          <p>
            Every question is checked for signs of an emergency before the model starts writing:
            stroke, heart attack, trouble breathing, anaphylaxis, thoughts of suicide, heavy
            bleeding, and fever in a young baby. Guidance appears above the answer. A pasted exam
            question describing an emergency is recognised as an exam question, so the warning is
            kept for people describing themselves or someone with them. It is a pattern check,
            not a clinician, and it will miss things.
          </p>
        </li>
      </ol>
    </article>
  );
}
```

- [ ] **Step 4: Run everything and confirm it passes**

```bash
cd web && npm test && npm run build && npm run e2e; cd ..
.venv/bin/python -m pytest -q
```

Expected: vitest passes, the build lists `/`, `/results`, `/method`, all five
end-to-end tests pass, and the Python suite passes.

- [ ] **Step 5: Write `README.md`**

```markdown
# Qwen3-4B fine-tuned for medical question answering

A 4-billion-parameter Qwen3 model fine-tuned on medical exam questions and patient
conversations on a free Kaggle T4, compared with the model it started from on two
exam benchmarks it never saw, and served behind a safety screen and a chat
interface.

It is an educational tool, not a medical device. It can be wrong.

## What was trained

17,285 examples in one pass: MedMCQA and MedQA-USMLE exam questions, worked
clinical reasoning, and ChatDoctor patient conversations, about 70% exam and
reasoning to 30% dialogue. A LoRA adapter of rank 32 on Qwen3-4B, 4.8 hours on one
T4. The 5,456 benchmark questions were held out and checked against the training
data before training began. See `results/run1/` for what the run measured.

## Evaluation

See `docs/superpowers/plans/2026-09-24-evaluation.md`. Results appear in
`results/run1/eval_*.json` and on the Results page once the evaluation has run.

## Run it

    # the model server (needs requirements-dev.txt and requirements-smoke.txt,
    # and the adapter in training/outputs/run1)
    bash serving/start_local.sh
    # or canned replies with the real safety screen, no model
    MOCK_BACKEND=1 bash serving/start_local.sh

    cd web && npm install && npm run dev      # http://localhost:3000

## Tests

    .venv/bin/python -m pytest                # Python: data, scoring, safety, server
    cd web && npm test && npm run e2e         # web: parsers, report, end to end

## Layout

    training/   data preparation, training, evaluation, Kaggle notebooks
    evaluation/ the evaluation notebook's Kaggle metadata
    serving/    the model server and the safety screen
    web/        the Next.js app: Ask, Results, Method
    results/    what each run measured
    scripts/    Kaggle uploads, notebook generation, the local smoke test
```

- [ ] **Step 6: Commit**

```bash
git add web/ README.md
git commit -m "Add the Method page, end-to-end tests, and a README

Method walks the pipeline in the order it ran, with its numbers read
from the run's own records, and says what each check cannot do: the
duplicate check matches exact text and misses paraphrase, and the
safety screen is a pattern check that will miss things.

The end-to-end tests run the real page against the real server in mock
mode: an answer streams, emergency guidance renders above it, a pasted
exam vignette raises no alarm, and the Results and Method pages
render."
```

---

## After this plan

The controller renders each page at 1280px and 360px with Playwright, reviews the
screenshots against the design tokens and the frontend-design principles, and fixes
anything that collides, overflows or reads as template. When the evaluation from
Plan 2 finishes, its `eval_medqa.json` and `eval_medmcqa.json` are copied into
`results/run1/`, and the Results page fills in on the next build.

## Self-Review

**Spec coverage (sections 4 and 5).**

| spec requirement | task |
|---|---|
| FastAPI server, GPU or CPU backend by env var | 2 (`ModelBackend` on cuda / mps / cpu via `load_model`) |
| `screen_message()`: emergencies and out-of-scope asks | 1 |
| emergency guidance above the answer, not suppressing it | 2 (screen first), 4 (panel above), 6 (asserted) |
| validated against a benign control set that fires on none | 1 (17 benign questions) |
| Chat: streaming, multi-turn, safety banner, red-flag callout | 4 |
| Benchmark: base vs tuned, both modes, per-subject | 5 |
| Method | 6 |
| proxy route; no token or model address in the browser | 4 |
| `MOCK_BACKEND=1` for building offline | 2, used by 4 and 6 |

Deferred, deliberately: the spec's GGUF backend and ngrok deployment. The reserved
ngrok domain serves the text2sql demo today, and reusing it would take that demo
down; local serving on Apple silicon runs the full-precision model directly.

**Placeholder scan.** No TBD or TODO. Every code step carries its code and every
test step its assertions.

**Type consistency.** `Screen.to_dict()` keys (`emergency`, `out_of_scope`,
`exam_context`; each flag `category`, `heading`, `advice`) match `Screen` and
`ScreenFlag` in `lib/types.ts`. The `/chat` event names `screen`, `token`, `done`,
`error` match `ServerEvent`. `evaluate.py`'s report keys match `EvalReport` and
`EvalFile`. `trim_history` and `system_prompt_for` have the same signatures in
Task 2's tests and implementation. The DOM hooks the end-to-end tests use —
"Your question", the "Ask" button, `.turn-model .answer`, `role="alert"`, a
"Training" heading, an image named "Training loss…", `ol.pipeline > li` — are each
produced by the task that names them.

**Review Focus.** Each of the five lines has its test in the owning task: exam
vignettes and first-person symptoms in Task 1; lay phrasing and curly apostrophes
in Task 1; the warning before the answer in Tasks 2 and 6; the server not running
in Tasks 2 and 4; long pastes and long histories in Task 2.
