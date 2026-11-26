import json,sys
D='/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/input/code-train/dolci-code-train.jsonl'
_rows=None
def rows():
    global _rows
    if _rows is None: _rows=[json.loads(l) for l in open(D)]
    return _rows
