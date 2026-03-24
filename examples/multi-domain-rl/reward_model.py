"""Multi-domain reward model: math, code, ifbench.

External code used:
- ``grade_answer_verl`` / ``extract_answer``: Miles core
  (``miles.rollout.rm_hub.math_utils``), adapted from DeepScaler
  (``agentica-project/deepscaler``) + Dan Hendrycks' normalization.

- ``IFEvalG/`` constraint checker library: (c) 2024 Google Research Authors
  (Apache 2.0), vendored unmodified from AllenAI's open-instruct
  (``github.com/allenai/open-instruct/.../IFEvalG``).

Original to this example:
- Math fallback heuristics (``lenient_math_reward``): answer-line,
  last-number, last-letter, numeric normalization.  Not in open-instruct.
- Code execution reward (``compute_code_reward``, ``_execute_code_with_tests``,
  ``_run_pytest_tests``): subprocess + partial credit.  NOT from open-instruct
  (whose ``CodeVerifier`` calls an external HTTP API instead).
- IFBench reward wrappers (``_compute_ifeval_reward``,
  ``_check_one_constraint``, ``_enrich_ifbench_metadata``): uses the same
  IFEvalG library as open-instruct's ``IFEvalVerifier`` but adds
  partial-credit toggle, metadata enrichment, and oe-eval prompt-rebuild
  logic.

Usage:
    --custom-rm-path \
        examples.multi-domain-rl.reward_model.batched_reward
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter

# Miles core: math grading from DeepScaler + Dan Hendrycks' normalization.
from miles.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer
from miles.rollout.rm_hub.math_utils import grade_answer_verl
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")
_ASSERT_FUNC_RE = re.compile(r"assert\s+(\w+)\s*\(")
_REPETITION_THRESHOLD = 0.5
_CODE_TIMEOUT = int(os.environ.get("CODE_EXEC_TIMEOUT", "10"))


# ---------------------------------------------------------------------------
# Think-model support: strip <think>...</think> reasoning traces
# ---------------------------------------------------------------------------

def _strip_thinking(response: str) -> str:
    """Strip chain-of-thought <think>...</think> block from a response.

    For think models, the final answer follows the closing </think> tag.
    If no </think> tag is present, the response is returned unchanged.
    """
    if "</think>" in response:
        return response.split("</think>", 1)[-1].strip()
    return response


# ---------------------------------------------------------------------------
# Repetition detection
# ---------------------------------------------------------------------------

def _repetition_ratio(text: str, n: int = 4) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


# ---------------------------------------------------------------------------
# Math reward
# grade_answer_verl / extract_answer: Miles core (from DeepScaler / Hendrycks).
# Fallback heuristics (answer-line, last-number, last-letter, numeric
# normalization) are not from open-instruct.
# ---------------------------------------------------------------------------

def _extract_last_number(text: str) -> str | None:
    matches = re.findall(r"(?<!\w)(-?\d+(?:\.\d+)?)(?!\w)", text)
    return matches[-1] if matches else None


def _extract_last_letter(text: str) -> str | None:
    matches = re.findall(r"(?:^|\s|[:(])([A-E])(?:\s|[.),;:]|$)", text)
    return matches[-1] if matches else None


def _normalize_numeric(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").replace("\\", "")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return None


def _extract_answer_line(text: str) -> str | None:
    matches = _ANSWER_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def lenient_math_reward(response: str, label: str) -> float:
    if not label or not response:
        return 0.0

    label_str = str(label).strip()

    if grade_answer_verl(response, label_str):
        return 1.0

    if "</think>" in response:
        after_think = response.split("</think>")[-1]
        extracted = extract_boxed_answer(after_think)
        if extracted is not None and grade_answer_verl(f"\\boxed{{{extracted}}}", label_str):
            return 1.0

    answer_line = _extract_answer_line(response)
    if answer_line is not None:
        answer_norm = _normalize_numeric(answer_line)
        label_norm = _normalize_numeric(label_str)
        if answer_norm is not None and label_norm is not None and answer_norm == label_norm:
            return 1.0

    if len(label_str) == 1 and label_str.upper() in "ABCDE":
        last_letter = _extract_last_letter(response)
        if last_letter and last_letter.upper() == label_str.upper():
            return 1.0

    label_norm = _normalize_numeric(label_str)
    if label_norm is not None:
        last_num = _extract_last_number(response)
        if last_num is not None:
            last_norm = _normalize_numeric(last_num)
            if last_norm is not None and last_norm == label_norm:
                return 1.0

    return 0.0


# ---------------------------------------------------------------------------
# Code reward  (subprocess + partial credit)
# ---------------------------------------------------------------------------

def _extract_python_code(response: str, entry_point: str | None = None) -> str | None:
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        if entry_point:
            for m in matches:
                if f"def {entry_point}" in m:
                    return m.strip()
        return matches[-1].strip()
    if "def " in response:
        return response.strip()
    return None


def _execute_code_with_tests(code: str, test_cases: list[str]) -> float:
    if not code or not test_cases:
        return 0.0

    test_block = "\n".join(test_cases)
    full_code = f"{code}\n\n{test_block}\n"

    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True, text=True, timeout=_CODE_TIMEOUT,
        )
        if result.returncode == 0:
            return 1.0

        n_passed = 0
        for tc in test_cases:
            single = f"{code}\n\n{tc}\n"
            try:
                r = subprocess.run(
                    [sys.executable, "-c", single],
                    capture_output=True, text=True, timeout=_CODE_TIMEOUT,
                )
                if r.returncode == 0:
                    n_passed += 1
            except (subprocess.TimeoutExpired, Exception):
                pass
        return n_passed / len(test_cases) if test_cases else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception as e:
        logger.debug(f"Code execution error: {e}")
        return 0.0


def _infer_entry_point(test_cases) -> str:
    if isinstance(test_cases, dict):
        return test_cases.get("entry_point", "")
    if isinstance(test_cases, list):
        for tc in test_cases:
            m = _ASSERT_FUNC_RE.search(str(tc))
            if m:
                return m.group(1)
    return ""


_TEST_FUNC_RE = re.compile(r"^def (test_\w+)\s*\(", re.MULTILINE)


def _extract_test_functions(test_str: str) -> list[str]:
    """Split pytest-style test code into individual test function blocks."""
    matches = list(_TEST_FUNC_RE.finditer(test_str))
    if not matches:
        return []
    blocks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(test_str)
        blocks.append(test_str[start:end].rstrip())
    return blocks


def _run_pytest_tests(code: str, test_str: str) -> float:
    """Run pytest-style test functions with partial credit."""
    test_funcs = _extract_test_functions(test_str)
    if not test_funcs:
        full = f"{code}\n\n{test_str}\n"
        try:
            r = subprocess.run(
                [sys.executable, "-c", full],
                capture_output=True, text=True, timeout=_CODE_TIMEOUT,
            )
            return 1.0 if r.returncode == 0 else 0.0
        except (subprocess.TimeoutExpired, Exception):
            return 0.0

    all_code = f"{code}\n\n" + "\n\n".join(test_funcs)
    calls = "\n".join(f"{_TEST_FUNC_RE.search(tf).group(1)}()" for tf in test_funcs)
    full = f"{all_code}\n\n{calls}\n"
    try:
        r = subprocess.run(
            [sys.executable, "-c", full],
            capture_output=True, text=True, timeout=_CODE_TIMEOUT,
        )
        if r.returncode == 0:
            return 1.0
    except (subprocess.TimeoutExpired, Exception):
        pass

    n_passed = 0
    for tf in test_funcs:
        fname = _TEST_FUNC_RE.search(tf).group(1)
        single = f"{code}\n\n{tf}\n\n{fname}()\n"
        try:
            r = subprocess.run(
                [sys.executable, "-c", single],
                capture_output=True, text=True, timeout=_CODE_TIMEOUT,
            )
            if r.returncode == 0:
                n_passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass
    return n_passed / len(test_funcs)


def compute_code_reward(response: str, label: str) -> float:
    response = _strip_thinking(response)
    try:
        test_cases = json.loads(label)
    except (json.JSONDecodeError, TypeError):
        test_cases = None

    entry_point = _infer_entry_point(test_cases)
    code = _extract_python_code(response, entry_point=entry_point or None)
    if code is None:
        return 0.0

    if isinstance(test_cases, dict):
        test_str = test_cases.get("test", "")
        if not test_str:
            return 0.0
        if _TEST_FUNC_RE.search(test_str):
            return _run_pytest_tests(code, test_str)
        if entry_point and "check(" in test_str:
            full_test = f"{code}\n\n{test_str}\ncheck({entry_point})\n"
            try:
                result = subprocess.run(
                    [sys.executable, "-c", full_test],
                    capture_output=True, text=True, timeout=_CODE_TIMEOUT,
                )
                return 1.0 if result.returncode == 0 else 0.0
            except (subprocess.TimeoutExpired, Exception):
                return 0.0
        return _run_pytest_tests(code, test_str)

    if isinstance(test_cases, list):
        return _execute_code_with_tests(code, test_cases)

    return 0.0


# ---------------------------------------------------------------------------
# IFBench reward
# ---------------------------------------------------------------------------

try:
    from .IFEvalG import instructions_registry
except ImportError:
    try:
        _ifeval_dir = os.path.dirname(__file__)
        if _ifeval_dir not in sys.path:
            sys.path.insert(0, _ifeval_dir)
        from IFEvalG import instructions_registry
    except ImportError:
        instructions_registry = None
        logger.warning("IFEvalG not found; ifbench rewards will return 0.0")


def _check_one_constraint(inst_id, kwargs, response, prompt_text=None):
    if inst_id not in instructions_registry.INSTRUCTION_DICT:
        return False
    try:
        if kwargs is None:
            kwargs = {}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        checker = instructions_registry.INSTRUCTION_DICT[inst_id](inst_id)
        checker.build_description(**kwargs)
        args = checker.get_instruction_args()
        if args and "prompt" in args and prompt_text:
            checker.build_description(prompt=prompt_text)
        return bool(response.strip() and checker.check_following(response))
    except Exception:
        return False


def _compute_ifeval_reward(response: str, metadata: dict, partial_credit: bool = False) -> float:
    if instructions_registry is None:
        return 0.0

    instruction_ids = metadata.get("instruction_id_list", [])
    kwargs_list = metadata.get("kwargs", [])
    prompt_text = metadata.get("prompt_text", "")

    if not instruction_ids:
        return 0.0
    if not kwargs_list or len(kwargs_list) != len(instruction_ids):
        kwargs_list = [{}] * len(instruction_ids)

    if not response.strip():
        return 0.0

    passed = 0
    total = len(instruction_ids)
    for inst_id, kw in zip(instruction_ids, kwargs_list):
        if _check_one_constraint(inst_id, kw, response, prompt_text):
            passed += 1
        elif not partial_credit:
            return 0.0

    if partial_credit:
        return passed / total if total > 0 else 0.0
    return 1.0


def _enrich_ifbench_metadata(meta: dict, label, prompt) -> dict:
    if meta.get("instruction_id_list"):
        return meta

    meta = dict(meta)
    parsed = label
    if isinstance(parsed, str):
        try:
            import ast
            parsed = ast.literal_eval(parsed)
        except Exception:
            return meta

    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
        constraint = parsed[0]
        meta["instruction_id_list"] = constraint.get("instruction_id", [])
        meta["kwargs"] = constraint.get("kwargs", [])

    if "prompt_text" not in meta:
        if isinstance(prompt, list):
            meta["prompt_text"] = prompt[0].get("content", "") if prompt else ""
        elif isinstance(prompt, str):
            meta["prompt_text"] = prompt

    return meta


# ---------------------------------------------------------------------------
# Per-sample scoring — routes by metadata.rm_type
# ---------------------------------------------------------------------------

def _score_one(sample: Sample, evaluation: bool = False) -> float:
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (meta.get("rm_type") or "").strip()
    response = sample.response or ""
    label = sample.label

    # Strip thinking traces for repetition check (full response may be
    # dominated by <think> content, use the answer portion only).
    answer_text = _strip_thinking(response) if "</think>" in response else response
    if _repetition_ratio(answer_text) > _REPETITION_THRESHOLD:
        return 0.0

    if rm_type == "ifbench":
        meta = _enrich_ifbench_metadata(meta, label, sample.prompt)
        return _compute_ifeval_reward(_strip_thinking(response), meta, partial_credit=not evaluation)

    if rm_type == "code":
        return compute_code_reward(response, str(label) if label is not None else "")

    if rm_type in ("math", "deepscaler"):
        return lenient_math_reward(response, str(label) if label is not None else "")

    if label is not None:
        return lenient_math_reward(response, str(label))
    return 0.0


# ---------------------------------------------------------------------------
# Entry point (matches Miles custom_rm_path API signature)
# ---------------------------------------------------------------------------

async def batched_reward(args, samples, **kwargs):
    """Synchronous multi-domain reward (no LLM judge needed)."""
    evaluation = kwargs.get("evaluation", False)

    if isinstance(samples, Sample):
        return _score_one(samples, evaluation=evaluation)

    return [_score_one(s, evaluation=evaluation) for s in samples]
