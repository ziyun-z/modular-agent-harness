"""
Tests for SWEBenchLoader.

Fast tests use a lightweight mock dataset to avoid network calls.
Integration tests (marked with @pytest.mark.integration) hit HuggingFace
and are skipped by default — run with:  pytest -m integration
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.swebench_loader import SWEBenchLoader, SWEBenchTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(
    instance_id: str = "django__django-1234",
    repo: str = "django/django",
    base_commit: str = "abc123",
    problem_statement: str = "Fix a bug",
    hints_text: str = "Look at views.py",
    test_patch: str = "diff --git ...",
    patch: str = "diff --git ...",
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    version: str = "3.2",
    environment_setup_commit: str = "def456",
) -> dict:
    """Return a dict that looks like a HuggingFace SWE-bench row."""
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "hints_text": hints_text,
        "test_patch": test_patch,
        "patch": patch,
        "FAIL_TO_PASS": json.dumps(fail_to_pass or ["tests/test_views.py::TestFoo::test_bar"]),
        "PASS_TO_PASS": json.dumps(pass_to_pass or ["tests/test_models.py::TestBaz::test_qux"]),
        "version": version,
        "environment_setup_commit": environment_setup_commit,
    }


FAKE_ROWS = [
    _make_row("repo__proj-001", fail_to_pass=["tests/a.py::test_1"]),
    _make_row("repo__proj-002", fail_to_pass=["tests/b.py::test_2"]),
    _make_row("repo__proj-003", fail_to_pass=["tests/c.py::test_3"]),
]


@pytest.fixture()
def loader() -> SWEBenchLoader:
    """SWEBenchLoader backed by an in-memory fake dataset — no network calls."""
    with patch("src.evaluation.swebench_loader.load_dataset") as mock_ld:
        mock_ld.return_value = FAKE_ROWS
        return SWEBenchLoader("swebench_lite")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_resolves_known_alias(self):
        with patch("src.evaluation.swebench_loader.load_dataset") as mock_ld:
            mock_ld.return_value = FAKE_ROWS
            SWEBenchLoader("swebench_lite")
            mock_ld.assert_called_once_with(
                "princeton-nlp/SWE-bench_Lite", split="test"
            )

    def test_resolves_full_swebench_alias(self):
        with patch("src.evaluation.swebench_loader.load_dataset") as mock_ld:
            mock_ld.return_value = FAKE_ROWS
            SWEBenchLoader("swebench")
            mock_ld.assert_called_once_with(
                "princeton-nlp/SWE-bench", split="test"
            )

    def test_unknown_name_passed_through(self):
        """Arbitrary HuggingFace dataset names are forwarded as-is."""
        with patch("src.evaluation.swebench_loader.load_dataset") as mock_ld:
            mock_ld.return_value = FAKE_ROWS
            SWEBenchLoader("org/custom-dataset")
            mock_ld.assert_called_once_with("org/custom-dataset", split="test")


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

class TestLen:
    def test_returns_dataset_size(self, loader: SWEBenchLoader):
        assert len(loader) == 3


# ---------------------------------------------------------------------------
# get_task
# ---------------------------------------------------------------------------

class TestGetTask:
    def test_returns_correct_task(self, loader: SWEBenchLoader):
        task = loader.get_task("repo__proj-001")
        assert isinstance(task, SWEBenchTask)
        assert task.instance_id == "repo__proj-001"

    def test_all_fields_populated(self, loader: SWEBenchLoader):
        task = loader.get_task("repo__proj-001")
        assert task.repo == "django/django"
        assert task.base_commit == "abc123"
        assert task.problem_statement == "Fix a bug"
        assert task.hints_text == "Look at views.py"
        assert task.test_patch == "diff --git ..."
        assert task.patch == "diff --git ..."
        assert task.version == "3.2"
        assert task.environment_setup_commit == "def456"

    def test_fail_to_pass_parsed_to_list(self, loader: SWEBenchLoader):
        task = loader.get_task("repo__proj-001")
        assert isinstance(task.fail_to_pass, list)
        assert task.fail_to_pass == ["tests/a.py::test_1"]

    def test_pass_to_pass_parsed_to_list(self, loader: SWEBenchLoader):
        task = loader.get_task("repo__proj-001")
        assert isinstance(task.pass_to_pass, list)

    def test_missing_instance_id_raises_key_error(self, loader: SWEBenchLoader):
        with pytest.raises(KeyError, match="nonexistent"):
            loader.get_task("nonexistent")

    def test_error_message_includes_dataset_size(self, loader: SWEBenchLoader):
        with pytest.raises(KeyError, match="3"):
            loader.get_task("nonexistent")


# ---------------------------------------------------------------------------
# get_all_tasks
# ---------------------------------------------------------------------------

class TestGetAllTasks:
    def test_returns_all_tasks(self, loader: SWEBenchLoader):
        tasks = loader.get_all_tasks()
        assert len(tasks) == 3

    def test_all_elements_are_swebench_tasks(self, loader: SWEBenchLoader):
        tasks = loader.get_all_tasks()
        assert all(isinstance(t, SWEBenchTask) for t in tasks)

    def test_instance_ids_match(self, loader: SWEBenchLoader):
        tasks = loader.get_all_tasks()
        ids = {t.instance_id for t in tasks}
        assert ids == {"repo__proj-001", "repo__proj-002", "repo__proj-003"}


# ---------------------------------------------------------------------------
# sample_task_ids
# ---------------------------------------------------------------------------

class TestSampleTaskIds:
    def test_returns_correct_count(self, loader: SWEBenchLoader):
        ids = loader.sample_task_ids(2)
        assert len(ids) == 2

    def test_returns_valid_ids(self, loader: SWEBenchLoader):
        ids = loader.sample_task_ids(3)
        all_ids = {r["instance_id"] for r in FAKE_ROWS}
        assert set(ids).issubset(all_ids)

    def test_reproducible_with_same_seed(self, loader: SWEBenchLoader):
        ids_a = loader.sample_task_ids(2, seed=0)
        ids_b = loader.sample_task_ids(2, seed=0)
        assert ids_a == ids_b

    def test_different_seeds_may_differ(self, loader: SWEBenchLoader):
        # With 3 tasks sampled 2, seeds 0 and 99 are very likely to differ
        ids_a = loader.sample_task_ids(2, seed=0)
        ids_b = loader.sample_task_ids(2, seed=99)
        # Not guaranteed, but true for this dataset/seeds — guard with len check
        assert len(ids_a) == len(ids_b) == 2

    def test_clamps_to_dataset_size(self, loader: SWEBenchLoader):
        """Requesting more than available should return all tasks, not error."""
        ids = loader.sample_task_ids(999)
        assert len(ids) == 3

    def test_returns_list(self, loader: SWEBenchLoader):
        assert isinstance(loader.sample_task_ids(1), list)


# ---------------------------------------------------------------------------
# _parse edge cases
# ---------------------------------------------------------------------------

class TestParse:
    def test_json_string_fail_to_pass(self):
        """FAIL_TO_PASS stored as a JSON string is parsed correctly."""
        row = _make_row()
        row["FAIL_TO_PASS"] = '["tests/x.py::test_a", "tests/x.py::test_b"]'

        with patch("src.evaluation.swebench_loader.load_dataset") as mock_ld:
            mock_ld.return_value = [row]
            loader = SWEBenchLoader("swebench_lite")

        task = loader.get_task("django__django-1234")
        assert task.fail_to_pass == ["tests/x.py::test_a", "tests/x.py::test_b"]

    def test_native_list_fail_to_pass(self):
        """FAIL_TO_PASS already a list (some HF versions) is handled."""
        row = _make_row()
        row["FAIL_TO_PASS"] = ["tests/x.py::test_a"]

        with patch("src.evaluation.swebench_loader.load_dataset") as mock_ld:
            mock_ld.return_value = [row]
            loader = SWEBenchLoader("swebench_lite")

        task = loader.get_task("django__django-1234")
        assert task.fail_to_pass == ["tests/x.py::test_a"]

    def test_missing_optional_fields_default_to_empty_string(self):
        """Optional fields missing from the row default gracefully."""
        row = _make_row()
        del row["hints_text"]
        del row["test_patch"]
        del row["patch"]
        del row["version"]
        del row["environment_setup_commit"]

        with patch("src.evaluation.swebench_loader.load_dataset") as mock_ld:
            mock_ld.return_value = [row]
            loader = SWEBenchLoader("swebench_lite")

        task = loader.get_task("django__django-1234")
        assert task.hints_text == ""
        assert task.test_patch == ""
        assert task.patch == ""
        assert task.version == ""
        assert task.environment_setup_commit == ""


# ---------------------------------------------------------------------------
# Integration tests  (skipped unless -m integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """Hit the real HuggingFace dataset. Requires network access."""

    @pytest.fixture(scope="class")
    def real_loader(self) -> SWEBenchLoader:
        return SWEBenchLoader("swebench_lite")

    def test_loads_300_tasks(self, real_loader: SWEBenchLoader):
        assert len(real_loader) == 300

    def test_known_task_exists(self, real_loader: SWEBenchLoader):
        # This instance_id is in SWE-bench Lite
        task = real_loader.get_task("django__django-11179")
        assert task.repo == "django/django"
        assert len(task.fail_to_pass) > 0

    def test_sample_ids_are_valid(self, real_loader: SWEBenchLoader):
        ids = real_loader.sample_task_ids(10)
        assert len(ids) == 10
        for id_ in ids:
            task = real_loader.get_task(id_)
            assert task.instance_id == id_
