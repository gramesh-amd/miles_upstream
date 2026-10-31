# IFEvalG

IFEval constraint checker taken from AllenAI's open-instruct codebase:
https://github.com/allenai/open-instruct/tree/main/open_instruct/IFEvalG

No modifications were made.

## Usage

Used by `reward_model.py` to verify IFEval/IFBench constraint satisfaction:

```python
from .IFEvalG import instructions_registry

checker = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
checker.build_description(**kwargs)
passed = checker.check_following(response)
```
