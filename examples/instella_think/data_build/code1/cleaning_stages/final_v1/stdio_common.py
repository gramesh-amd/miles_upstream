"""Shared helpers for the final_v1 build."""
import re

V7 = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data"
FINAL = f"{V7}/final_v1"
WORK = "/tmp/v7final"

# Decorative-image heuristic (MERGE_NOTES rule 1; same regexes as /tmp/v7reg/dp_stdio.py and compute.py):
# every image tag sits on its own line or inside the Note/Explanation section, and the statement never
# points at a figure/picture for required information.
TAG = re.compile(r"<image>|\[Image\]|!\[[^\]]*\]\([^)]*\)", re.I)
FIG = re.compile(r"\b(figure|picture|shown below|image below|illustrat|see the image|pictured|the drawing)\b", re.I)


def decorative_image(t):
    if FIG.search(t):
        return False
    note = re.search(r"\n\s*(Note|Explanation)\b", t)
    for m in TAG.finditer(t):
        if note and m.start() > note.start():
            continue
        ls = t.rfind("\n", 0, m.start()) + 1
        le = t.find("\n", m.end())
        le = len(t) if le < 0 else le
        if t[ls:le].strip() != m.group(0).strip():
            return False
    return bool(TAG.search(t))


def src_of(cid):
    if cid.startswith("rlvr_acecoder"):
        return "acecoder"
    if cid.startswith("llama-nemotron"):
        return "nemotron"
    if cid.startswith("klear"):
        return "klear"
    if cid.startswith("synthetic2"):
        return "synthetic2"
    return "?"
