# Merge-step corrections (applied after the fixers finish)

1. **Image-placeholder drops (stdio, about 1,500 rows) are too aggressive.** A hand check of 6 random drops found
   about half decorative: a standalone `<image>`, or one in Note. Restore a row only if all of these hold:
   - every `<image>` / `[Image]` is on its own line or inside the Note section, not inside a sentence where it
     stands for a formula ("fraction <image>", "in form <image>");
   - the statement does not refer to "the figure", "shown below", "the picture" for required information;
   - a reference solution passes all tests (verified).
   Restored rows get `metadata.fixes += ["image_decorative_kept"]`.
2. **Klear near-duplicate drops (1,846, against about 1,385 exact duplicates):** audit 30 random near-dup pairs.
   Keep both copies when the statements differ in constraints or output format.
3. **Reconcile with codebases/REPORT.md:** adopt existing implementations where they match or beat ours.
4. **Decontamination (user policy):**
   - Drop overlap with LCB v5/v6, Codeforces eval, HE-Pro/MBPP-Pro and CRUXEval.
   - Flag LCB v3 overlap (`metadata.lcb_v3_overlap`).
   - Drop confirmed HE+/MBPP+ paraphrases.
