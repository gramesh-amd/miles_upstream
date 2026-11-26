"""Stdio route, binary-reward re-decide: re-runs stdio/scripts/decide.py on the existing statement analysis
and sandbox results with the partial-credit rule removed.

Patched relative to the v7 stdio pipeline:
  * constant-output floor: no thinning; a row drops only when one constant output passes every test;
  * dedup winner: verified first, then most tests (was most tests first, so an unverified copy could win);
  * every row that survives cleaning is saved before dedup (prededup.pkl: final tests, verifying refs),
    so merge.py can merge duplicate copies' tests into the winner;
  * rows whose images are decorative (stdio_common.decorative_image) go through the normal cleaning instead
    of the image drop. They were never executed, so they come out unverified; verify.py runs their
    references through the grader server;
  * all outputs go to /tmp/v7final/stdio (nothing under v7_data/stdio is written).
Nothing is executed: the verdicts come from stdio/work/results.pkl.

    python stdio_redecide.py
"""
import sys
import types

SD = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/stdio/scripts"
NEW = "/tmp/v7final/stdio"
sys.path.insert(0, SD)

PATCHES = [
    ('open(f"{OUTD}/fixed.jsonl", "w")', f'open("{NEW}/fixed.jsonl", "w")'),
    ('open(f"{OUTD}/dropped.csv", "w", newline="")', f'open("{NEW}/dropped.csv", "w", newline="")'),
    ('open(f"{OUTD}/multi_answer.csv", "w", newline="")', f'open("{NEW}/multi_answer.csv", "w", newline="")'),
    ('open(f"{OUTD}/stats.json", "w")', f'open("{NEW}/stats.json", "w")'),
    ('open(f"{W}/decisions.pkl", "wb")', f'open("{NEW}/decisions.pkl", "wb")'),
    ("        if floor >= 0.5:\n            S[(src, \"const_floor_ge50\")] += 1",
     "        if floor >= 1.0:\n            S[(src, \"const_floor_ge50\")] += 1"),
    ('kept[row] = dict(rec=rec, v=v, n=len(final), verified=verified, n_ver=len(verifiers))',
     'kept[row] = dict(rec=rec, v=v, n=len(final), verified=verified, n_ver=len(verifiers), verifiers=verifiers)'),
    ("    # --- 10. dedup: keep the copy with the most tests (then verified, then more verifying refs) ---\n",
     "    # --- 10. dedup ---\n"
     f"    pickle.dump({{r: dict(rec=k['rec'], verified=k['verified'], n=k['n'], verifiers=k['verifiers'], "
     f"src=k['v']['src'], custom_id=k['v']['custom_id'], cluster=D[r]) for r, k in kept.items()}}, "
     f"open('{NEW}/prededup.pkl', 'wb'))\n"),
    ('rows.sort(key=lambda r: (-kept[r]["n"], -kept[r]["verified"], -kept[r]["n_ver"], r))',
     'rows.sort(key=lambda r: (-kept[r]["verified"], -kept[r]["n"], -kept[r]["n_ver"], r))'),
    ('        if a["image"]:\n            drop(v, "image_placeholder"',
     '        if a["image"] and not decorative_image(v["prompt"]):\n            drop(v, "image_placeholder"'),
    ('from analyze import norm_in\n', 'from analyze import norm_in\nfrom stdio_common import decorative_image\n'),
]

if __name__ == "__main__":
    import os
    os.makedirs(NEW, exist_ok=True)
    src = open(f"{SD}/decide.py").read()
    for old, new in PATCHES:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    mod = types.ModuleType("decide_binary")
    mod.__file__ = f"{SD}/decide.py"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    mod.main()
