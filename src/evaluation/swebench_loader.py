"""
SWE-bench Lite dataset loader.

Downloads from HuggingFace and exposes tasks as SWEBenchTask dataclasses.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from datasets import load_dataset

DATASET_HF_MAP = {
    "swebench_lite": "princeton-nlp/SWE-bench_Lite",
    "swebench": "princeton-nlp/SWE-bench",
}


@dataclass
class SWEBenchTask:
    """A single SWE-bench task."""
    instance_id: str
    repo: str                   # e.g. "django/django"
    base_commit: str            # commit to checkout before applying changes
    problem_statement: str      # the GitHub issue text
    hints_text: str             # optional extra hints
    test_patch: str             # patch that adds the evaluation tests
    patch: str                  # gold patch (for validation only, not shown to agent)
    fail_to_pass: list[str]     # tests that must pass after the fix
    pass_to_pass: list[str]     # tests that must continue passing
    version: str
    environment_setup_commit: str


class SWEBenchLoader:
    """Loads SWE-bench tasks from HuggingFace datasets."""

    def __init__(self, dataset_name: str = "swebench_lite"):
        hf_name = DATASET_HF_MAP.get(dataset_name, dataset_name)
        self._dataset = load_dataset(hf_name, split="test")
        # Build instance_id → row index for O(1) lookup
        self._index: dict[str, int] = {
            row["instance_id"]: i for i, row in enumerate(self._dataset)
        }

    def get_task(self, instance_id: str) -> SWEBenchTask:
        if instance_id not in self._index:
            raise KeyError(
                f"Task '{instance_id}' not found. "
                f"Dataset has {len(self._index)} tasks."
            )
        return self._parse(self._dataset[self._index[instance_id]])

    def get_all_tasks(self) -> list[SWEBenchTask]:
        return [self._parse(row) for row in self._dataset]

    def sample_task_ids(self, n: int, seed: int = 42) -> list[str]:
        """Return a reproducible random sample of task IDs."""
        ids = list(self._index.keys())
        rng = random.Random(seed)
        return rng.sample(ids, min(n, len(ids)))

    def __len__(self) -> int:
        return len(self._dataset)

    def _parse(self, row: dict) -> SWEBenchTask:
        def parse_list(value) -> list[str]:
            if isinstance(value, str):
                return json.loads(value)
            return list(value)

        return SWEBenchTask(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            hints_text=row.get("hints_text", ""),
            test_patch=row.get("test_patch", ""),
            patch=row.get("patch", ""),
            fail_to_pass=parse_list(row["FAIL_TO_PASS"]),
            pass_to_pass=parse_list(row["PASS_TO_PASS"]),
            version=row.get("version", ""),
            environment_setup_commit=row.get("environment_setup_commit", ""),
        )
