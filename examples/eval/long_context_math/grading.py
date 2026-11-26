"""Answer extraction and grading, shared by every script so a cell cannot be scored two ways."""

import re

BOXED = re.compile(r"\\boxed\s*\{")

# The sentence Qwen3 publishes for halting its thinking under a budget, used verbatim for every model.
STOP_THINKING = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n"

# Preference order when several appear; "</think>" is the de-facto convention.
CLOSE_CANDIDATES = ["</think>", "</thought>", "</reasoning>", "<|end_of_thought|>"]


def norm(s: str) -> str:
    s = s.strip().strip("$").strip().rstrip(".")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\(?:text|mathrm|displaystyle)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    return re.sub(r"[{}\s,$\\]", "", s).lower()


def _brace_body(text: str, start: int) -> str | None:
    depth, i, out = 1, start, []
    while i < len(text):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return "".join(out)
        out.append(text[i])
        i += 1
    return None


def last_boxed(text: str) -> str | None:
    matches = list(BOXED.finditer(text))
    return _brace_body(text, matches[-1].end()) if matches else None


def is_correct(text: str, label: str) -> bool:
    b = last_boxed(text)
    return b is not None and norm(b) == norm(label)


def detect_close(finished: list[dict], sample: int = 400, floor: float = 0.5) -> str:
    """The end-of-thinking marker this model actually emits, read from its own finished generations.

    Chat templates are not reliable for this: Instella's template has no think markers while the model
    closes with </think> in every finished sample. Below `floor` the model is treated as having no
    thinking phase and gets an empty marker.
    """
    if not finished:
        return ""
    look = finished[:sample]
    best, rate = "", 0.0
    for cand in CLOSE_CANDIDATES:
        r = sum(cand in f["response"] for f in look) / len(look)
        if r > rate:
            best, rate = cand, r
    return best if rate >= floor else ""


def forcing_suffix(close: str) -> str:
    return STOP_THINKING + (close + "\n\n" if close else "\n")
