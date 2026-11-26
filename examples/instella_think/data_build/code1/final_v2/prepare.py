"""Stage 1 (no execution): join the dropped final_v1 rows to open-r1/codeforces and build candidate records.

Candidates (final_v1/dropped.csv, mapped to a CF id in decontam/date_check.csv, ref=cf):
  task "statement": S4 non-decorative image rows (822). The statement text (description, input, output, note) is
      replaced by Open-R1's text; our wrapper (prefix/suffix instructions), layout and Examples block are kept.
  task "checker":   S3 multi-answer rows and S_label_differs:refs_follow_label rows (577). The prompt is unchanged;
      Open-R1's generated_checker is attached.
Per row: mapping consistency, statement checks on the new text, test cleaning, merge of Open-R1 official tests,
references (our v6/OLMo refs + accepted CF Python submissions) and wrong submissions (for checker negatives).

    python3 prepare.py  ->  /tmp/v7v2/cands.pkl
"""
import collections
import csv
import glob
import json
import pickle
import re
import sys

import pandas as pd

from v2common import (CUTOFF, FINAL1, MAX_TEST_BYTES, MAX_TESTS_TOTAL, OPENR1, V7, WORK, latex_dollars,
                      prompt_format, split_wrapper)
from analyze import RX, body_of, has_multi, norm_in, parse_samples, tolerance  # noqa: E402
from sandbox import grader_eq  # noqa: E402  (string compare only)
from v2common import load_server_module  # noqa: E402
ces = load_server_module()  # comparison helpers of the current grader (no execution)


def eq_ci(a, b):
    """grader_eq, or equal once YES/NO are case-folded (the grader's case_insensitive option)."""
    return grader_eq(a, b) or ces._lines_equal_with_decimal(a, b, None, True)

PRI = {"v6": 0, "olmo": 1, "v6near": 3}
MAX_CF_OK, MAX_CF_WA = 6, 3
FLOATY = re.compile(r"^-?\d+\.\d+$")
# figure references that make a statement depend on a missing picture (analyze.RX image plus figure/picture forms)
IMG_REF = re.compile(RX["image"].pattern + r"|\b(figure|picture|illustration|drawing)s? (below|above)\b"
                     r"|\b(as )?shown (in|on) the (figure|picture|image|drawing)\b|\bin the (figure|picture)\b", re.I)
TAG = re.compile(r"<image>|\[image\]|!\[[^\]]*\]\(|<img\b", re.I)


def num_tokens(t):
    t = t.replace("\\cdot", " ").replace("·", " ")
    t = re.sub(r"[\^{}]", "", t)
    return {x for x in re.findall(r"\d+", t) if len(x) >= 2}


def openr1_text(o):
    return "\n".join(latex_dollars(o.get(k) or "") for k in ("description", "input_format", "output_format", "note"))


def render_examples(fmt, exs):
    exs = [e for e in exs if e.get("input") is not None]
    if fmt == "dash":
        head = "-----Example-----" if len(exs) == 1 else "-----Examples-----"
        return head + "\n" + "\n\n".join(f"Input\n{e['input'].strip()}\n\nOutput\n{e['output'].strip()}" for e in exs)
    if fmt == "s2cf":
        return ("Demo Input:\n" + repr([e["input"].strip() + "\n" for e in exs]) + "\n\nDemo Output:\n"
                + repr([e["output"].strip() + "\n" for e in exs]))
    head = "Example" if len(exs) == 1 else "Examples"
    return head + "\n\n" + "\n\n\n".join(f"Input\n\n{e['input'].strip()}\n\n\nOutput\n\n{e['output'].strip()}" for e in exs)


def swap_statement(prompt, o):
    """New prompt: same wrapper and layout, Open-R1 text sections, our Examples block kept verbatim."""
    pre, core, suf = split_wrapper(prompt)
    fmt = prompt_format(core)
    desc, inp, out, note = (latex_dollars(o.get(k) or "") for k in ("description", "input_format", "output_format", "note"))
    exs = list(o["examples"]) if o.get("examples") is not None else []
    ex_block, had_note, how = None, False, "sections"
    if fmt == "dash":
        i_in = core.find("\n-----Input-----\n")
        i_out = core.find("\n-----Output-----\n", i_in + 1)
        m_ex = re.compile(r"\n-----(Examples?|Sample[^\n]*?)-----\n").search(core, max(i_out, 0) + 1)
        i_note = core.find("\n-----Note-----\n", (m_ex.start() if m_ex else max(i_out, 0)) + 1)
        had_note = i_note >= 0
        if i_in < 0 or i_out < 0:
            how = "rendered"
        elif m_ex:
            ex_block = core[m_ex.start() + 1:(i_note if had_note else len(core))].strip("\n")
        new = f"{desc}\n\n\n-----Input-----\n\n{inp}\n\n\n-----Output-----\n\n{out}\n\n\n"
        new += (ex_block or render_examples("dash", exs))
        if had_note and note:
            new += f"\n\n\n\n-----Note-----\n\n{note}"
    elif fmt == "s2cf":
        i_pd = core.find("Problem Description:\n")
        i_demo = core.find("\n\nDemo Input:\n")
        i_note = core.find("\n\nNote:\n", max(i_demo, 0))
        head = core[:i_pd] if i_pd >= 0 else ""
        if i_demo >= 0:
            ex_block = core[i_demo + 2:(i_note if i_note >= 0 else len(core))].strip("\n")
        else:
            how = "rendered"
        new = (f"{head}Problem Description:\n{desc}\n\nInput Specification:\n{inp}\n\nOutput Specification:\n{out}\n\n"
               + (ex_block or render_examples("s2cf", exs)) + f"\n\nNote:\n{note or 'none'}")
        had_note = True
    else:
        m_in = re.compile(r"\n\nInput\n\n").search(core)
        m_out = re.compile(r"\n\nOutput\n\n").search(core, m_in.end()) if m_in else None
        m_ex = re.compile(r"\n\n(Examples?|Sample Input|Samples?)\n\n").search(core, m_out.end()) if m_out else None
        notes = [m for m in re.finditer(r"\n\nNote\n\n", core) if m_ex and m.start() > m_ex.start()]
        had_note = bool(notes)
        if fmt != "cc" or not m_in or not m_out:
            how = "rendered"
        if m_ex and how == "sections":
            ex_block = core[m_ex.start() + 2:(notes[-1].start() if notes else len(core))].strip("\n")
        new = f"{desc}\n\nInput\n\n{inp}\n\nOutput\n\n{out}\n\n" + (ex_block or render_examples("cc", exs))
        if had_note and note:
            new += f"\n\nNote\n\n{note}"
    if ex_block is None:
        how = "rendered" if how == "rendered" else "sections_examples_rendered"
    return pre + new.strip("\n") + suf, dict(fmt=fmt, how=how, kept_our_examples=ex_block is not None)


def mapping_check(v, o, task, tests, off_tests):
    """Is our row the Open-R1 problem? Samples, tests, constraint numbers, title and time limit must agree."""
    body = body_of(v["prompt"])
    ours = parse_samples(body)
    exs = [dict(input=e["input"], output=e["output"]) for e in (o["examples"] if o["examples"] is not None else [])]
    ex_in = {norm_in(e["input"]): e["output"] for e in exs}
    s_in = [s for s in ours if norm_in(s["input"]) in ex_in]
    s_match = [s for s in s_in if task == "checker" or eq_ci(s["output"], ex_in[norm_in(s["input"])])]
    s_conflict = len(s_in) - len(s_match)
    offd = {norm_in(t["input"]): t["output"] for t in off_tests}
    t_in = [t for t in tests if norm_in(t["input"]) in offd]
    t_conf = [t for t in t_in if task != "checker" and not eq_ci(t["output"], offd[norm_in(t["input"])])]
    t_case = [t for t in t_in if task != "checker" and not grader_eq(t["output"], offd[norm_in(t["input"])])]
    # our tests vs statement examples too (rows whose statement samples did not parse)
    e_in = [t for t in tests if norm_in(t["input"]) in ex_in]
    e_conf = [t for t in e_in if task != "checker" and not eq_ci(t["output"], ex_in[norm_in(t["input"])])]
    e_case = [t for t in e_in if task != "checker" and not grader_eq(t["output"], ex_in[norm_in(t["input"])])]
    our_nums = num_tokens(re.sub(r"^(Time|Memory) Limit: .*$", "", body, flags=re.M))
    their = num_tokens(openr1_text(o) + "\n" + "\n".join(e["input"] + "\n" + e["output"] for e in exs))
    cont = len(our_nums & their) / len(our_nums) if our_nums else 1.0
    missing = sorted(our_nums - their, key=len)[:8]
    title_ok = tl_ok = None
    m = re.search(r"^Title: (.*)$", v["prompt"], re.M)
    if m:
        title_ok = m.group(1).strip().lower() == str(o["title"] or "").strip().lower()
    m = re.search(r"^Time Limit: ([\d.]+) seconds", v["prompt"], re.M)
    if m and o.get("time_limit") is not None:
        tl_ok = abs(float(m.group(1)) - float(o["time_limit"])) < 1e-6
    agree = bool(s_match) or bool(t_in and not t_conf) or bool(e_in and not e_conf)
    why = []
    if not agree:
        why.append("no_shared_sample_or_test")
    if s_conflict or t_conf or e_conf:
        why.append("sample_or_test_output_differs")
    if len(our_nums) >= 3 and cont < 0.9:
        why.append("constraint_numbers_differ")
    if title_ok is False:
        why.append("title_differs")
    if tl_ok is False:
        why.append("time_limit_differs")
    return dict(ok=not why, why=why, n_our_samples=len(ours), sample_match=len(s_match), sample_conflict=s_conflict,
                test_shared=len(t_in), test_conflict=len(t_conf), ex_test_shared=len(e_in), ex_test_conflict=len(e_conf),
                num_containment=round(cont, 3), n_our_nums=len(our_nums), missing_nums=missing, title_ok=title_ok,
                tl_ok=tl_ok, yes_no_case_differs=bool(t_case or e_case))


def clean_tests(tests, task):
    """Light version of stdio/decide.py cleaning for rows never cleaned (python-literal tests in a stdin set,
    identical repeats; for exact-compare rows a same-input/different-output conflict drops the row; for checker rows
    the first test per input is kept)."""
    from decide import _strlist
    out, notes, seen = [], collections.Counter(), {}
    for t in tests:
        if not isinstance(t, dict):
            notes["non_dict_test"] += 1
            continue
        t = {"input": str(t.get("input", "")), "output": str(t.get("output", ""))}
        if _strlist(t["input"]) and _strlist(t["output"]):
            notes["python_literal_test_dropped"] += 1
            continue
        if len(t["input"]) + len(t["output"]) > MAX_TEST_BYTES:
            notes["gt1MB_test_dropped"] += 1
            continue
        k = norm_in(t["input"])
        if k in seen:
            if task == "checker" or grader_eq(t["output"], seen[k]):
                notes["repeat_dropped"] += 1
                continue
            notes["same_input_conflicting_outputs"] += 1
            continue
        seen[k] = t["output"]
        out.append(t)
    return out, notes


def main():
    C = json.load(open(f"{WORK}/cands.json"))
    D = pickle.load(open(f"{WORK}/cand_rows.pkl", "rb"))
    rows = D["rows"]
    dc = {r["custom_id"]: r for r in csv.DictReader(open(f"{V7}/decontam/date_check.csv"))}
    drop_reason = {r["custom_id"]: r["reason"] for r in csv.DictReader(open(f"{FINAL1}/dropped.csv"))}
    O = pd.read_pickle(f"{WORK}/openr1_default.pkl")
    orec = {r["id"]: r for r in O.to_dict("records")}
    subs = pd.read_parquet("/tmp/datacand/norm/cf_py_subs.parquet")
    want = {dc[c]["ref_id"] for c in C["img"] + C["ma"]}
    subs = subs[subs.problem_id.isin(want)]
    ok_subs, wa_subs = collections.defaultdict(list), collections.defaultdict(list)
    for r in subs.itertuples():
        (ok_subs if r.verdict == "OK" else wa_subs)[r.problem_id].append(r.source)
    out, stats = {}, collections.Counter()
    for task, ids in (("statement", C["img"]), ("checker", C["ma"])):
        for cid in ids:
            v = rows[cid]
            pid = dc[cid]["ref_id"]
            o = orec.get(pid)
            rec = dict(cid=cid, task=task, src=v["src"], pid=pid, v1_reason=drop_reason[cid], row=v["row"],
                       s7_cluster=v["s7_cluster"], metadata=dict(v["metadata"]), orig_prompt=v["prompt"],
                       olmo_passrate=v.get("olmo_passrate"), drop=None, changes=[])
            out[cid] = rec
            if o is None:
                rec["drop"] = "openr1_id_not_found"
                continue
            date = pd.Timestamp(o["contest_start"], unit="s").strftime("%Y-%m-%d") if o["contest_start"] else ""
            checker = o["generated_checker"] or None
            off = [dict(input=t["input"].replace("\r\n", "\n"), output=t["output"].replace("\r\n", "\n")) for t in (o["official_tests"] if o["official_tests"] is not None else [])]
            tests, notes = clean_tests(v["label"], "checker" if (task == "checker" or checker) else "statement")
            rec.update(date=date, title=o["title"], rating=o["rating"], time_limit=o["time_limit"],
                       memory_limit=o["memory_limit"], checker=checker, n_official=len(off), clean_notes=dict(notes),
                       official_complete=bool(o["official_tests_complete"]), n_own_tests=len(tests))
            mc = mapping_check(v, o, "checker" if (task == "checker" or checker) else "statement", tests, off)
            rec["mapping"] = mc
            # --- new prompt / statement-level checks
            if task == "statement":
                newp, info = swap_statement(v["prompt"], o)
                rec["prompt"], rec["swap"] = newp, info
                nonnote = "\n".join(latex_dollars(o.get(k) or "") for k in ("description", "input_format", "output_format"))
                rec["img_ref_nonnote"] = [m.group(0) for m in IMG_REF.finditer(nonnote)][:3]
                rec["img_tag_left"] = bool(TAG.search(newp))
                body = body_of(newp)
            else:
                rec["prompt"] = v["prompt"]
                body = body_of(v["prompt"])
            rec["interactive"] = bool(o["interaction_format"]) or bool(RX["interactive"].search(body))
            rec["file_io"] = o["input_mode"] != "stdio"
            rec["multi_hits"] = has_multi(body)[:2]
            # --- grading route and options
            rec["route"] = "checker" if checker else "stdio"
            opts = {}
            if rec["route"] == "stdio":
                tol = tolerance(body)
                if tol is not None and any(FLOATY.match(tok) for t in tests for tok in t["output"].split()):
                    opts["float_tol"] = tol if isinstance(tol, float) else 1e-6
                if RX["case_stmt"].search(body) or mc["yes_no_case_differs"]:
                    opts["case_insensitive"] = True
            rec["opts"] = opts
            # --- tests: own (cleaned) first, then Open-R1 official tests with new inputs, up to MAX_TESTS_TOTAL
            seen = {norm_in(t["input"]) for t in tests}
            add = []
            for t in off:
                if len(tests) + len(add) >= MAX_TESTS_TOTAL:
                    break
                k = norm_in(t["input"])
                if k in seen or len(t["input"]) + len(t["output"]) > MAX_TEST_BYTES:
                    continue
                seen.add(k)
                add.append(t)
            rec["tests"], rec["official_added"] = tests, add
            # --- references: v6 > OLMo (by Dolci reward) > accepted CF submissions > v6 near-miss
            refs = [dict(origin=r["origin"], code=r["code"], reward=r.get("reward")) for r in v["refs"]]
            refs.sort(key=lambda r: (PRI.get(r["origin"], 2), -(r["reward"] or 0)))
            cf_ok = [dict(origin="cf_ok", code=c) for c in dict.fromkeys(ok_subs.get(pid, []))][:MAX_CF_OK]
            mine = [r for r in refs if r["origin"] != "v6near"]
            rec["refs"] = mine + cf_ok + [r for r in refs if r["origin"] == "v6near"]
            rec["cf_ok"] = cf_ok
            rec["cf_wa"] = [dict(origin="cf_wa", code=c) for c in dict.fromkeys(wa_subs.get(pid, []))][:MAX_CF_WA]
            # --- static drop decision (first failing rule)
            if not mc["ok"]:
                rec["drop"] = "ambiguous_mapping:" + "+".join(mc["why"])
            elif task == "statement" and (rec["img_ref_nonnote"] or rec["img_tag_left"]):
                rec["drop"] = "openr1_statement_still_refers_to_image"
            elif rec["interactive"] or rec["file_io"]:
                rec["drop"] = "interactive_or_file_io"
            elif task == "checker" and not checker:
                rec["drop"] = "openr1_no_generated_checker"
            elif task == "statement" and not checker and rec["multi_hits"]:
                rec["drop"] = "multi_answer_no_checker"
            elif not date or date >= CUTOFF:
                rec["drop"] = "date_after_cutoff"
            elif notes.get("same_input_conflicting_outputs"):
                rec["drop"] = "same_input_conflicting_outputs"
            elif len(tests) + len(add) < 5:
                rec["drop"] = "lt5_tests_before_merge"
            stats[(task, rec["drop"].split(":")[0] if rec["drop"] else "candidate")] += 1
    pickle.dump(out, open(f"{WORK}/cands.pkl", "wb"))
    for k, n in sorted(stats.items(), key=str):
        print(k, n)
    amb = collections.Counter(why for r in out.values() if r.get("mapping") for why in r["mapping"]["why"])
    print("mapping reasons (non-exclusive)", dict(amb))
    print("swap", collections.Counter((r["swap"]["fmt"], r["swap"]["how"]) for r in out.values() if r.get("swap")))
    print("routes", collections.Counter((r["task"], r["route"]) for r in out.values() if not r["drop"]))


if __name__ == "__main__":
    main()
