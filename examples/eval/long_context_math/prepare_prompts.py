#!/usr/bin/env python3
"""Render one model's benchmark prompts with its own chat template.

    python3 prepare_prompts.py --model qwen3_5_4b

Run wherever transformers and the model's tokenizer are available (typically inside the serving
container); run_cell.py only reads the resulting JSON. Prompt token counts are measured here because
each sample's generation budget is the context minus the prompt.

Prompts are per model: the question text is identical, but each model gets its own turn markers and
template switches (chat_kwargs).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import BENCHES, DATA_DIR, MODELS, OUT_DIR, model_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    a = ap.parse_args()
    from transformers import AutoTokenizer

    m = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(str(model_path(m)), trust_remote_code=True)
    out = []
    for bench, fn in BENCHES:
        for pi, line in enumerate(open(DATA_DIR / f"{fn}.jsonl")):
            d = json.loads(line)
            text = tok.apply_chat_template(d["prompt"], tokenize=False, add_generation_prompt=True, **m.get("chat_kwargs", {}))
            out.append({"bench": bench, "pi": pi, "label": d["label"], "text": text, "ntok": len(tok(text)["input_ids"])})
    dest = OUT_DIR / f"prompts_{a.model}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))
    print(f"{len(out)} prompts -> {dest}   longest {max(o['ntok'] for o in out)} tokens")
    print(f"  sample tail: {out[0]['text'][-90:]!r}")


if __name__ == "__main__":
    main()
