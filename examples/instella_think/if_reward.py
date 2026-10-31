#!/usr/bin/env python3
"""Verifiable instruction-following reward for the Instella-MoE Think-RL runs.

Wired in with ``--custom-rm-path if_reward.batched_reward`` (this directory is on the
workers' PYTHONPATH as INSTELLA_DIR, see the runtime_env block in run_async.sh).

Constraints are checked in-process by the vendored IFEvalG registry (./IFEvalG,
Apache-2.0, Google Research via AllenAI open-instruct). Nothing here reaches the
network, which is deliberate: miles' built-in ``--rm-type ifbench``
(miles/rollout/rm_hub/ifbench.py) `git clone`s allenai/IFBench at IMPORT time on every
worker and pip-installs into the running venv. That is a startup-time network
dependency and a mutating install inside a 24-node job, for a checker we can vendor.

WHAT THIS SCORES
Only the answer, never the reasoning. The </think> trace is stripped before checking,
so a model cannot satisfy "write 3 paragraphs" inside its scratchpad. Training uses
PARTIAL credit (fraction of constraints satisfied) and eval uses STRICT all-or-nothing,
matching IFEval's reported metric. The split matters: strict reward on multi-constraint
prompts is mostly 0 early on, which starves GRPO of within-group variance -- and the
dynamic-sampling filter (check_reward_nonzero_std) then discards those groups outright,
so a strict-only reward would silently drop most of the batch.

REWARD SCALE IS 1.0, NOT 10.0
Instella-MoE/rl/examples/if_rl/reward_model.py used REWARD_SCALE=10.0, which only works
alongside its miles_core_changes.patch: that patch redefines "perfect" as
_REWARD_PERFECT=REWARD_SCALE inside miles/utils/metric_utils.py and log_utils.py and
divides eval means by it. We do not carry that patch (it does not apply to this miles),
so a 10x reward here would leave miles' own pass-rate/eval-accuracy metrics reading
"7.28" for 72.8% and would multiply every advantage by 10 against an unchanged
--eps-clip. Keep this at 1.0 unless you also port those hunks.

SILENT ZEROS ARE THE FAILURE MODE
Every path that can return 0.0 for a reason that is NOT "the model disobeyed" is
counted and logged (missing metadata, unknown constraint id, checker exception,
think-gate rejection). If reward is flat 0, read the IF_REWARD line in the worker log
before touching hyperparameters.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
import sys
import threading
from collections import Counter

logger = logging.getLogger(__name__)

# --- vendored checker registry ---------------------------------------------------
try:
    from .IFEvalG import instructions_registry
except ImportError:  # loaded as a top-level module (the usual case under Ray)
    _self_dir = os.path.dirname(os.path.abspath(__file__))
    if _self_dir not in sys.path:
        sys.path.insert(0, _self_dir)
    try:
        from IFEvalG import instructions_registry
    except ImportError as exc:  # pragma: no cover
        instructions_registry = None
        _IMPORT_ERROR = exc
        logger.error(
            "IFEvalG import failed (%s). Every IF reward will be 0.0. "
            "Deps are langdetect, absl-py, immutabledict, nltk (+ punkt_tab).",
            exc,
        )

# --- secondary registry: IFBench's extended constraints ----------------------------
# IFEval-era IFEvalG implements 54 constraint types. The IFBench eval set uses a NEWER
# family (sentence:keyword, words:consonants, ratio:stop_words, format:sub-bullets, ...)
# that IFEvalG does not have at all -- measured: 300/300 IFBench prompts referenced at
# least one unknown id, i.e. that eval set would have scored a flat 0.0 and looked like
# total model failure. IFBench's own INSTRUCTION_DICT is a superset (classic IFEval
# verifiers plus the new ones) exposing the same build_description/check_following
# interface, so it is consulted for anything IFEvalG lacks.
#
# IFEvalG stays PRIMARY on purpose: it is vendored in-tree, so training rewards do not
# depend on an external clone being present, and ids common to both keep identical
# semantics to the recipe this run is reproducing.
_IFBENCH_DIR = os.environ.get("IF_IFBENCH_DIR", "")
ifbench_registry = None
if _IFBENCH_DIR and os.path.isdir(_IFBENCH_DIR):
    try:
        if _IFBENCH_DIR not in sys.path:
            sys.path.insert(0, _IFBENCH_DIR)
        from ifbench import instructions_registry as ifbench_registry  # type: ignore

        logger.info(
            "IFBench registry loaded from %s (%d constraint types)",
            _IFBENCH_DIR,
            len(ifbench_registry.INSTRUCTION_DICT),
        )
    except Exception as exc:
        ifbench_registry = None
        logger.warning(
            "IFBench registry NOT loaded from %s (%s). IFBench-only constraints will "
            "score 0; needs `pip install emoji syllapy`.",
            _IFBENCH_DIR,
            exc,
        )
else:
    logger.warning(
        "IF_IFBENCH_DIR=%r not a directory; IFBench-only constraints will score 0.",
        _IFBENCH_DIR,
    )

# Make the reward DETERMINISTIC. langdetect seeds its detector from the clock unless told
# otherwise, and three checkers depend on it -- language:response_language
# (IFEvalG/instructions.py:181), change_case:english_capital (:1358) and
# change_case:english_lowercase (:1387). Unseeded, the identical response can be scored
# differently on two calls, which in RL is noise injected directly into the group's
# advantages, and it is worst on short answers where detection is least certain. Neither
# IFEvalG nor the upstream Instella reward sets this. Caught by the self-test below, where
# a valid lowercase answer intermittently scored 0.5 instead of 1.0.
try:
    import langdetect

    langdetect.DetectorFactory.seed = 0
except ImportError:
    pass

# --- knobs -----------------------------------------------------------------------
REWARD_SCALE = float(os.environ.get("IF_REWARD_SCALE", "1.0"))
_PARTIAL_CREDIT = os.environ.get("IF_PARTIAL_CREDIT", "1") == "1"
_REQUIRE_THINK = os.environ.get("IF_REQUIRE_THINK", "1") == "1"
_THINK_PREFILL = os.environ.get("IF_THINK_PREFILL", "0") == "1"
_CONCURRENCY = int(os.environ.get("IF_REWARD_CONCURRENCY", "128"))
_LOG_EVERY = int(os.environ.get("IF_LOG_EVERY", "200"))

_SPECIAL_TOKEN_RE = re.compile(r"<｜[^｜]*｜>")


# --- telemetry -------------------------------------------------------------------
class _Stats:
    """Counts the ways a reward can be 0 for a non-model reason."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.c: Counter = Counter()
        self.reward_sum = 0.0
        self.strict_pass = 0
        self.unknown_ids: Counter = Counter()

    def note(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.c[key] += n

    def note_sample(self, reward: float, strict_ok: bool) -> None:
        with self._lock:
            self.c["scored"] += 1
            self.reward_sum += reward
            self.strict_pass += int(strict_ok)
            n = self.c["scored"]
        if _LOG_EVERY > 0 and n % _LOG_EVERY == 0:
            self.dump()

    def dump(self) -> None:
        with self._lock:
            n = max(1, self.c["scored"])
            top = ", ".join(f"{k}={v}" for k, v in self.unknown_ids.most_common(5))
            logger.info(
                "IF_REWARD scored=%d mean_reward=%.4f strict_pass=%.4f | "
                "no_metadata=%d label_fallback=%d think_gate_fail=%d empty_answer=%d "
                "unknown_id=%d checker_error=%d via_ifbench=%d | top_unknown[%s]",
                self.c["scored"],
                self.reward_sum / n,
                self.strict_pass / n,
                self.c["no_metadata"],
                self.c["label_fallback"],
                self.c["think_gate_fail"],
                self.c["empty_answer"],
                self.c["unknown_id"],
                self.c["checker_error"],
                self.c["via_ifbench"],
                top,
            )


STATS = _Stats()


# --- response handling -----------------------------------------------------------
def _clean_answer(response: str) -> str:
    """Return only the post-</think> answer, stripped of tags and special tokens."""
    if "</think>" in response:
        response = response.split("</think>", 1)[-1]
    elif _THINK_PREFILL or response.lstrip().startswith("<think>"):
        # An unclosed reasoning block means there is no answer to grade.
        return ""
    response = response.replace("<answer>", "").replace("</answer>", "")
    return _SPECIAL_TOKEN_RE.sub("", response).strip()


def _has_clean_think(response: str) -> bool:
    if _THINK_PREFILL:
        return "</think>" in response
    return response.lstrip().startswith("<think>") and "</think>" in response


# --- constraint checking ---------------------------------------------------------
def _checker_class(inst_id: str):
    """Vendored IFEvalG first, then IFBench's extended registry for what it lacks."""
    if instructions_registry is not None:
        cls = instructions_registry.INSTRUCTION_DICT.get(inst_id)
        if cls is not None:
            return cls
    if ifbench_registry is not None:
        cls = ifbench_registry.INSTRUCTION_DICT.get(inst_id)
        if cls is not None:
            STATS.note("via_ifbench")
            return cls
    return None


def _coerce_kwargs(kw) -> dict:
    """kwargs arrive as a dict, or as a JSON/py-literal string from parquet."""
    if isinstance(kw, str):
        try:
            kw = json.loads(kw)
        except Exception:
            try:
                kw = ast.literal_eval(kw)
            except Exception:
                return {}
    if not isinstance(kw, dict):
        return {}
    # A None value means "unset"; passing it through would override the checker's
    # own default with None and raise inside build_description.
    return {k: v for k, v in kw.items() if v is not None}


def _check_one(inst_id: str, kw, answer: str, prompt_text: str) -> bool:
    """True iff the answer satisfies this one constraint.

    Distinguishes "the model disobeyed" (False, expected) from "we could not even
    build the checker" (False, but counted as checker_error/unknown_id) -- the second
    kind is a data or wiring bug and must not be read as a model failure.
    """
    cls = _checker_class(inst_id)
    if cls is None:
        STATS.note("unknown_id")
        with STATS._lock:
            STATS.unknown_ids[inst_id] += 1
        return False

    try:
        checker = cls(inst_id)
        kwargs = _coerce_kwargs(kw)

        # Checkers that quote the prompt (combination:repeat_prompt, copy:*,
        # new:copy_span_idx) require prompt_to_repeat and raise ValueError without it.
        # The upstream Instella code tried to handle this with
        #     if "prompt" in checker.get_instruction_args(): build_description(prompt=...)
        # which is dead code twice over: no checker has an arg named "prompt" (they all
        # use "prompt_to_repeat"), and that second build_description call would have
        # dropped the kwargs from the first. So every copy/repeat constraint whose data
        # omitted prompt_to_repeat scored a silent 0 there. Fill it explicitly instead,
        # in ONE build_description call.
        try:
            needed = checker.get_instruction_args_keys() or []
        except Exception:
            needed = []
        if "prompt_to_repeat" in needed and not kwargs.get("prompt_to_repeat"):
            if not prompt_text:
                STATS.note("checker_error")
                return False
            kwargs["prompt_to_repeat"] = prompt_text

        checker.build_description(**kwargs)
        if not answer.strip():
            return False
        return bool(checker.check_following(answer))
    except Exception as exc:
        STATS.note("checker_error")
        logger.debug("IF checker %s failed: %s", inst_id, exc)
        return False


def _loose_variants(answer: str) -> list[str]:
    """The eight response variants of IFEval's official LOOSE metric.

    Verbatim from evaluation_lib.test_instruction_following_loose (allenai/IFBench,
    inherited from Google's IFEval): strip the first line, the last line, both, and/or
    markdown '*' emphasis. An instruction counts as followed if ANY variant satisfies it.

    Why this matters: published IFEval numbers are almost always the LOOSE ones, while a
    training reward wants STRICT. On this model the same responses read 73.2% strict and
    79.7% loose at step 39 -- a 6.5 point gap that is entirely metric definition, because
    a think-style policy tends to wrap its answer in a preamble ("Sure, here you go:")
    that loose scoring strips and strict scoring does not.
    """
    lines = answer.split("\n")
    no_first = "\n".join(lines[1:]).strip()
    no_last = "\n".join(lines[:-1]).strip()
    no_both = "\n".join(lines[1:-1]).strip()
    return [
        answer,
        answer.replace("*", ""),
        no_first,
        no_last,
        no_both,
        no_first.replace("*", ""),
        no_last.replace("*", ""),
        no_both.replace("*", ""),
    ]


def _check_one_loose(inst_id: str, kw, answer: str, prompt_text: str) -> bool:
    """Loose form of _check_one: satisfied by any of the eight variants."""
    return any(
        v.strip() and _check_one(inst_id, kw, v, prompt_text) for v in _loose_variants(answer)
    )


def score_counts(sample) -> tuple[int, int, int] | None:
    """(strict_followed, loose_followed, n_constraints), or None if unscorable.

    Raw counts rather than a per-sample ratio, because the two aggregation levels need
    different things from them: prompt-level accuracy is `followed == n`, while
    instruction-level accuracy is these counts summed over the whole dataset. IFEval
    micro-averages the latter (IFBench/evaluation_lib.py:219-220 accumulates
    instruction_correct/instruction_total across examples), so averaging per-prompt
    fractions would silently over-weight prompts carrying few constraints.
    """
    meta = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    response = getattr(sample, "response", None) or ""
    ids, kwargs_list = _constraints_of(sample, meta)
    if not ids:
        return None
    n = len(ids)
    # A gated or empty response followed nothing; it is NOT unscorable. Its constraints
    # still belong in the instruction-level denominator, or dropping them would make a
    # run that stops emitting </think> look like it improved.
    if _REQUIRE_THINK and "</think>" not in response:
        return 0, 0, n
    answer = _clean_answer(response)
    if not answer:
        return 0, 0, n
    prompt_text = _prompt_text_of(sample, meta)
    strict_n = sum(1 for i, kw in zip(ids, kwargs_list) if _check_one(i, kw, answer, prompt_text))
    loose_n = sum(
        1 for i, kw in zip(ids, kwargs_list) if _check_one_loose(i, kw, answer, prompt_text)
    )
    return strict_n, loose_n, n


def score_strict_and_loose(sample) -> tuple[float, float] | None:
    """(strict, loose) prompt-level scores in [0,1], or None if unscorable.

    Exposed for the eval logger (if_eval_log.py) and for offline analysis of dumped
    rollouts, so both read the SAME checker set the reward uses -- the only difference
    between the two numbers is the loose variant expansion, never the registry.
    """
    got = score_counts(sample)
    if got is None:
        return None
    strict_n, loose_n, n = got
    return float(strict_n == n), float(loose_n == n)


def _prompt_text_of(sample, meta: dict) -> str:
    if meta.get("prompt_text"):
        return str(meta["prompt_text"])
    prompt = getattr(sample, "prompt", None)
    if isinstance(prompt, list):
        for turn in reversed(prompt):
            if isinstance(turn, dict) and turn.get("role") == "user":
                return str(turn.get("content") or "")
        return str(prompt[0].get("content", "")) if prompt else ""
    return str(prompt or "")


def _constraints_of(sample, meta: dict) -> tuple[list, list]:
    """Return (instruction_id_list, kwargs_list), padding kwargs to match."""
    ids = meta.get("instruction_id_list")
    kwargs_list = meta.get("kwargs")

    if not ids:
        # Fallback: some exports stash the constraint spec on `label` instead.
        # prepare_if_data.py always writes metadata, so hitting this means the data
        # was produced some other way -- counted so it cannot pass unnoticed.
        parsed = getattr(sample, "label", None)
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except Exception:
                try:
                    parsed = ast.literal_eval(parsed)
                except Exception:
                    parsed = None
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            spec = parsed[0]
            ids = spec.get("instruction_id") or spec.get("instruction_id_list") or []
            kwargs_list = spec.get("kwargs") or []
            if ids:
                STATS.note("label_fallback")

    if isinstance(ids, str):
        ids = [ids]
    ids = list(ids or [])
    kwargs_list = list(kwargs_list or [])
    if len(kwargs_list) != len(ids):
        kwargs_list = (kwargs_list + [{}] * len(ids))[: len(ids)]
    return ids, kwargs_list


# --- scoring ---------------------------------------------------------------------
def _is_eval(sample, meta: dict, evaluation_kwarg: bool) -> bool:
    """Decide strict vs partial scoring.

    Do NOT trust the `evaluation` kwarg alone. miles threads `evaluation` all the way
    down to generate_and_rm() but then calls the reward WITHOUT it
    (miles/rollout/sglang_rollout.py:354, :363, :396 -- `batched_async_rm(args, samples)`
    and `async_rm(args, sample)`, no evaluation=), so for a --custom-rm-path reward that
    kwarg is always False, on the eval path too. The upstream Instella reward keyed its
    strict/partial split on exactly that kwarg, which means its "strict eval" was silently
    partial-credit on this code path.

    So the split is driven by the DATA instead: prepare_if_data.py stamps
    metadata.split = "eval" into the eval JSONLs, which miles passes through untouched.
    The kwarg is still honoured if a future miles starts forwarding it.
    """
    if evaluation_kwarg:
        return True
    return str(meta.get("split", "")).lower() == "eval"


def _score_one(sample, evaluation: bool = False) -> float:
    meta = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    response = getattr(sample, "response", None) or ""
    evaluation = _is_eval(sample, meta, evaluation)

    ids, kwargs_list = _constraints_of(sample, meta)
    if not ids:
        STATS.note("no_metadata")
        STATS.note_sample(0.0, False)
        return 0.0

    # Think-format gate. Training demands a well-formed <think>...</think>; eval only
    # demands the close, because a prefilled/again-truncated open should not zero a
    # measurement. Set IF_REQUIRE_THINK=0 for a no-think policy -- and do check the
    # smoke run, since if the policy does not emit <think> at all this gate turns
    # every training reward into 0 and the run will look like a broken reward.
    if _REQUIRE_THINK:
        ok = ("</think>" in response) if evaluation else _has_clean_think(response)
        if not ok:
            STATS.note("think_gate_fail")
            STATS.note_sample(0.0, False)
            return 0.0

    answer = _clean_answer(response)
    if not answer:
        STATS.note("empty_answer")
        STATS.note_sample(0.0, False)
        return 0.0

    prompt_text = _prompt_text_of(sample, meta)
    passed = sum(1 for i, kw in zip(ids, kwargs_list) if _check_one(i, kw, answer, prompt_text))

    strict_ok = passed == len(ids)
    partial = _PARTIAL_CREDIT and not evaluation
    raw = (passed / len(ids)) if partial else float(strict_ok)

    STATS.note_sample(raw, strict_ok)
    return raw * REWARD_SCALE


# --- entry point (--custom-rm-path) ----------------------------------------------
async def batched_reward(args, samples, **kwargs):
    """Reward entry point for both call shapes miles uses.

    miles calls this with a single Sample from async_rm() and with a list[Sample]
    from batched_async_rm() (miles/rollout/rm_hub/__init__.py:46,104), so both must
    work. Batch shape is detected by list/tuple rather than isinstance(_, Sample):
    Ray workers can re-import miles.utils.types under a different module identity,
    which makes that isinstance silently False (same hazard overlong_filter.py hit
    with Sample.Status).

    `evaluation` is passed through by miles and selects strict scoring.
    """
    evaluation = bool(kwargs.get("evaluation", False))

    if not isinstance(samples, (list, tuple)):
        return await asyncio.to_thread(_score_one, samples, evaluation)
    if not samples:
        return []

    sem = asyncio.Semaphore(max(1, _CONCURRENCY))

    async def _one(sample):
        async with sem:
            # The checkers are synchronous and some are heavy (nltk tokenization,
            # langdetect), so keep them off the event loop that is also driving
            # generation HTTP traffic.
            return await asyncio.to_thread(_score_one, sample, evaluation)

    return await asyncio.gather(*[_one(s) for s in samples])


# --- self-test -------------------------------------------------------------------
if __name__ == "__main__":
    # Smoke test with no miles/Ray involved:
    #   python3 examples/instella_think/if_reward.py
    class _S:
        def __init__(self, response, metadata, prompt="Write something.", label=None):
            self.response = response
            self.metadata = metadata
            self.prompt = prompt
            self.label = label

    logging.basicConfig(level=logging.INFO)
    if instructions_registry is None:
        raise SystemExit("IFEvalG not importable; install langdetect absl-py immutabledict nltk")

    print(f"registry: {len(instructions_registry.INSTRUCTION_DICT)} constraint types")
    meta = {
        "instruction_id_list": ["change_case:english_lowercase", "punctuation:no_comma"],
        "kwargs": [{}, {}],
        "prompt_text": "write it in lowercase with no commas",
    }
    cases = [
        ("<think>reasoning</think>all lowercase and no commas here", 1.0, "both pass"),
        ("<think>reasoning</think>All Lowercase, but has a comma", 0.0, "both fail"),
        ("<think>reasoning</think>lowercase, with a comma", 0.5, "one of two"),
        ("no think block at all", 0.0, "think gate"),
    ]
    ok = True
    for response, want, why in cases:
        got = asyncio.run(batched_reward(None, [_S(response, meta)]))[0]
        flag = "ok " if abs(got - want) < 1e-6 else "FAIL"
        ok &= flag == "ok "
        print(f"  {flag} train partial={got:.2f} want={want:.2f}  ({why})")

    # Strict eval, both routes. The metadata route is the one that actually fires in
    # production, because miles never forwards its `evaluation` kwarg to the reward.
    half = "<think>r</think>lowercase, comma"
    got = asyncio.run(batched_reward(None, [_S(half, meta)], evaluation=True))[0]
    flag = "ok " if got == 0.0 else "FAIL"
    ok &= flag == "ok "
    print(f"  {flag} eval strict={got:.2f} want=0.00  (via evaluation= kwarg)")

    eval_meta = dict(meta, split="eval")
    got = asyncio.run(batched_reward(None, [_S(half, eval_meta)]))[0]
    flag = "ok " if got == 0.0 else "FAIL"
    ok &= flag == "ok "
    print(f"  {flag} eval strict={got:.2f} want=0.00  (via metadata.split, the real path)")

    # And a single (non-list) sample, the async_rm call shape. The answer is deliberately a
    # long, unmistakably English sentence: change_case:english_lowercase runs langdetect,
    # which is least reliable on short strings.
    _long = "<think>r</think>this is a fully lowercase english sentence written without any commas at all"
    got = asyncio.run(batched_reward(None, _S(_long, meta)))
    flag = "ok " if abs(got - 1.0) < 1e-6 else "FAIL"
    ok &= flag == "ok "
    print(f"  {flag} single-sample call={got:.2f} want=1.00  (async_rm shape)")

    STATS.dump()
    raise SystemExit(0 if ok else 1)
