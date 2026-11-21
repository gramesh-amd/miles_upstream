import os

# Reward and loss knobs that reach the workers only as env vars (the instella_think launchers'
# RUNTIME_ENV_JSON), so they are invisible in args. Names that can carry credentials are skipped.
_ENV_VAR_NAMES = ("SLURM_JOB_ID",)
_ENV_VAR_PREFIXES = ("CODE_", "IF_", "DRGRPO", "OVERLONG_", "ADAPTIVE_FILTER", "SOLVED_REWARD", "PER_DOMAIN_",
                     "MLA_UNFUSED_ROPE", "SGLANG_USE_AITER")
_ENV_VAR_SECRET_MARKERS = ("URL", "TOKEN", "KEY", "SECRET", "PASSWORD")


def logged_env_vars() -> dict[str, str]:
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if (k in _ENV_VAR_NAMES or k.startswith(_ENV_VAR_PREFIXES))
        and not any(m in k for m in _ENV_VAR_SECRET_MARKERS)
    }
