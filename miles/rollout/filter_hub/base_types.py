import argparse
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from miles.utils.types import Sample


@dataclass
class FilterOutput:
    keep: bool
    reason: str | None = None


DynamicFilterOutput = FilterOutput


def iter_samples(group: list[Sample | list[Sample]]) -> Iterator[Sample]:
    for sample in group:
        if isinstance(sample, list):
            yield from sample
        else:
            yield sample


def call_dynamic_filter(fn, args, samples: list[Sample | list[Sample]], **kwargs):
    if fn is None:
        return FilterOutput(keep=True)

    output = fn(args, samples, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, FilterOutput):
        output = FilterOutput(keep=output)

    return output


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self._unfiltered_reward_sum = 0.0
        self._unfiltered_reward_count = 0
        self._groups_seen = 0

    def on_group_before_dynamic_filter(self, args: argparse.Namespace, group: list) -> None:
        self._groups_seen += 1
        for sample in _iter_group_samples(group):
            if sample.reward is None:
                continue
            if not args.reward_key and isinstance(sample.reward, dict):
                continue
            if (value := sample.get_reward_value(args)) is None:
                continue
            self._unfiltered_reward_sum += float(value)
            self._unfiltered_reward_count += 1

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
        if self._unfiltered_reward_count:
            metrics["rollout/raw_reward_unfiltered"] = self._unfiltered_reward_sum / self._unfiltered_reward_count
        if self._groups_seen:
            # Raw drop counts are not comparable across steps, because the number
            # of groups generated per step varies with oversampling. Fractions
            # are. Under an adaptive difficulty filter these are the actual
            # progress signal: rollout/raw_reward is pinned inside the kept band
            # by construction and goes flat, while "mastered" rising means
            # prompts are graduating out of the pool as the policy improves.
            metrics["rollout/dynamic_filter/groups_seen"] = self._groups_seen
            for reason, count in self._dynamic_filter_drop_reason_count.items():
                metrics[f"rollout/dynamic_filter/frac_{reason}"] = count / self._groups_seen
            # adaptive_difficulty_filter reports one reason per group shape
            # (mastered_6of8, mastered_7of8, ...) so the shift between shapes
            # stays visible. Summing them gives the single headline number:
            # what share of the sampled pool the policy has already outgrown.
            mastered = sum(
                count
                for reason, count in self._dynamic_filter_drop_reason_count.items()
                if reason.startswith("mastered")
            )
            if mastered:
                metrics["rollout/dynamic_filter/frac_mastered"] = mastered / self._groups_seen
            metrics["rollout/dynamic_filter/frac_kept"] = 1.0 - (
                sum(self._dynamic_filter_drop_reason_count.values()) / self._groups_seen
            )
        return metrics


def _iter_group_samples(group: list) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item
