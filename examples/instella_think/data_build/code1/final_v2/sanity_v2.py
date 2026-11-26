"""End-to-end sanity check of the recovered final_v2 rows through code_reward._score_one (training mode, the row's
own label and metadata, the grader server at CODE_EXEC_API_URL).
  * 30 random recovered rows with their verifying reference as the model answer: reward must be 1;
  * the same 30 rows with stubs (empty program, print(0), printing the row's most common expected output):
    reward must be 0 for every stub.

    CODE_EXEC_API_URL=http://127.0.0.1:18988/<token> python3 sanity_v2.py  ->  final_v2/work/sanity.json
"""
import collections
import gzip
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from v2common import FINAL2, jdump

sys.path.insert(0, "/tmp/wt_v7/examples/instella_think")
import code_reward  # noqa: E402


class S:
    def __init__(self, code, row):
        self.response = f"<think>r</think>\n```python\n{code}\n```"
        self.label, self.metadata = row["label"], dict(row["metadata"])


def main():
    rows = [json.loads(l) for l in open(f"{FINAL2}/train.jsonl")]
    rec = [r for r in rows if "openr1_statement" in r["metadata"]["v7_changes"] or r["metadata"].get("checker")]
    refs = json.load(gzip.open(f"{FINAL2}/work/verifying_refs.json.gz", "rt"))
    pick = random.Random(20261118).sample(rec, min(30, len(rec)))

    def one(row):
        cid = row["metadata"]["custom_id"]
        tests = json.loads(row["label"])
        common = collections.Counter(t["output"].strip() for t in tests).most_common(1)[0][0]
        out = dict(custom_id=cid, n=len(tests), checker=bool(row["metadata"].get("checker")),
                   ref=code_reward._score_one(S(refs[cid]["prog"], row)))
        for k, prog in {"empty_program": "pass", "print_0": "print(0)",
                        "print_most_common_output": f"print({common!r})"}.items():
            out[k] = code_reward._score_one(S(prog, row))
        return out

    with ThreadPoolExecutor(8) as ex:
        R = list(ex.map(one, pick))
    summ = dict(rows=len(R), ref_reward_1=sum(r["ref"] == 1.0 for r in R),
                stub_any_reward=sum(any(r[k] > 0 for k in ("empty_program", "print_0", "print_most_common_output")) for r in R),
                checker_rows=sum(r["checker"] for r in R), api_errors=code_reward.STATS.counts["api_error"])
    jdump(dict(summary=summ, rows=R), f"{FINAL2}/work/sanity.json")
    print(json.dumps(summ))
    for r in R:
        if r["ref"] != 1.0 or any(r[k] > 0 for k in ("empty_program", "print_0", "print_most_common_output")):
            print("PROBLEM", r)


if __name__ == "__main__":
    main()
