import logging
import os

import torch

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import MetricEvent

from .base import (
    MilesDashboardBackend,
    MlflowBackend,
    PrometheusBackend,
    TensorboardBackend,
    TrackingBackend,
    TrackingManager,
    WandbBackend,
)
from .ci_history import CiHistoryBackend

# The full registry lives here, not base.py: base must never import a backend
# module (ci_history imports TrackingBackend from base -> circular). This
# module is the tracking entry point and the one place that imports every
# backend, so it owns the registry.
BACKEND_REGISTRY: dict[str, tuple[type[TrackingBackend], str]] = {
    "wandb": (WandbBackend, "use_wandb"),
    "tensorboard": (TensorboardBackend, "use_tensorboard"),
    "mlflow": (MlflowBackend, "use_mlflow"),
    "prometheus": (PrometheusBackend, "use_prometheus"),
    "ci_history": (CiHistoryBackend, "ci_enable_metrics_capture"),
    "miles_dashboard": (MilesDashboardBackend, "use_miles_dashboard"),
}

logger = logging.getLogger(__name__)
_manager = TrackingManager(BACKEND_REGISTRY)


def init_tracking(args, primary: bool = True, **kwargs):
    _manager.init(args, primary=primary, **kwargs)


def define_step_key_metric_group(prefix: str, step_key: str) -> None:
    """Declare a metric group plotted against its own step key (e.g. ``{name}/*`` vs ``{name}/step``).
    Only wandb acts on this; must be called from the primary tracking process or definitions may be lost."""
    _manager.define_step_key_metric_group(prefix, step_key)


# Logged with every metric group (e.g. rollout/slurm_job_id against rollout/step) so that a run
# resumed across Slurm jobs shows where each job took over: the Slurm job id, and the launcher's
# segment number (MILES_SEGMENT, one per launch, so it also separates runs within one job).
_RUN_MARKERS = {
    name: int(value)
    for name, value in (("slurm_job_id", os.environ.get("SLURM_JOB_ID", "")), ("segment", os.environ.get("MILES_SEGMENT", "")))
    if value.isdigit()
}


def log(args, metrics, step_key: str):
    if _RUN_MARKERS and "/" in step_key:
        prefix = step_key.rsplit("/", 1)[0]
        metrics = {**metrics, **{f"{prefix}/{name}": value for name, value in _RUN_MARKERS.items()}}
    step = metrics.get(step_key)
    _manager.log(metrics, step=step, step_key=step_key)

    if is_event_logger_initialized():
        serializable_metrics = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()}
        get_event_logger().log(MetricEvent, {"metrics": serializable_metrics}, print_log=False)


def finish_tracking():
    _manager.finish()
