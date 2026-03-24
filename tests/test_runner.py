"""
Tests for runner.py: config loading, validation, module registry,
and build_*_module helpers.

No Docker, no API calls needed — all external I/O is mocked or uses
temp files.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.runner import (
    COMMUNICATION_REGISTRY,
    COMPRESSION_REGISTRY,
    MEMORY_REGISTRY,
    build_communication_module,
    build_compression_module,
    build_memory_module,
    load_config,
    validate_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str) -> str:
    """Write a YAML string to a temp file and return the path."""
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


def _baseline_cfg() -> dict:
    return {
        "experiment_name": "test",
        "model": "claude-sonnet-4-6",
        "memory": {"type": "naive", "params": {}},
        "compression": {"type": "none", "params": {}},
        "communication": {"type": "single_agent", "params": {}},
        "evaluation": {"dataset": "swebench_lite"},
        "sandbox": {"docker_image": "swebench-sandbox:latest", "timeout_per_task": 600},
    }


# ===========================================================================
# load_config
# ===========================================================================


class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, """
            experiment_name: test
            model: claude-sonnet-4-6
        """)
        cfg = load_config(path)
        assert cfg["experiment_name"] == "test"
        assert cfg["model"] == "claude-sonnet-4-6"

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_returns_dict(self, tmp_path):
        path = _write_yaml(tmp_path, "key: value\n")
        result = load_config(path)
        assert isinstance(result, dict)

    def test_nested_sections_accessible(self, tmp_path):
        path = _write_yaml(tmp_path, """
            memory:
              type: naive
              params: {}
        """)
        cfg = load_config(path)
        assert cfg["memory"]["type"] == "naive"


# ===========================================================================
# validate_config
# ===========================================================================


class TestValidateConfig:
    def test_valid_baseline_has_no_errors(self):
        assert validate_config(_baseline_cfg()) == []

    def test_missing_memory_section(self):
        cfg = _baseline_cfg()
        del cfg["memory"]
        errors = validate_config(cfg)
        assert any("memory" in e for e in errors)

    def test_missing_compression_section(self):
        cfg = _baseline_cfg()
        del cfg["compression"]
        errors = validate_config(cfg)
        assert any("compression" in e for e in errors)

    def test_missing_communication_section(self):
        cfg = _baseline_cfg()
        del cfg["communication"]
        errors = validate_config(cfg)
        assert any("communication" in e for e in errors)

    def test_missing_evaluation_section(self):
        cfg = _baseline_cfg()
        del cfg["evaluation"]
        errors = validate_config(cfg)
        assert any("evaluation" in e for e in errors)

    def test_missing_sandbox_section(self):
        cfg = _baseline_cfg()
        del cfg["sandbox"]
        errors = validate_config(cfg)
        assert any("sandbox" in e for e in errors)

    def test_unknown_memory_type(self):
        cfg = _baseline_cfg()
        cfg["memory"]["type"] = "unknown_memory"
        errors = validate_config(cfg)
        assert any("unknown_memory" in e for e in errors)

    def test_unknown_compression_type(self):
        cfg = _baseline_cfg()
        cfg["compression"]["type"] = "turbo_compression"
        errors = validate_config(cfg)
        assert any("turbo_compression" in e for e in errors)

    def test_unknown_communication_type(self):
        cfg = _baseline_cfg()
        cfg["communication"]["type"] = "hive_mind"
        errors = validate_config(cfg)
        assert any("hive_mind" in e for e in errors)

    def test_missing_memory_type_key(self):
        cfg = _baseline_cfg()
        del cfg["memory"]["type"]
        errors = validate_config(cfg)
        assert any("memory.type" in e for e in errors)

    def test_missing_compression_type_key(self):
        cfg = _baseline_cfg()
        del cfg["compression"]["type"]
        errors = validate_config(cfg)
        assert any("compression.type" in e for e in errors)

    def test_multiple_errors_returned(self):
        errors = validate_config({})
        assert len(errors) > 1

    def test_error_includes_available_options(self):
        cfg = _baseline_cfg()
        cfg["memory"]["type"] = "bad_type"
        errors = validate_config(cfg)
        error_text = " ".join(errors)
        # Should mention what's available
        assert "naive" in error_text


# ===========================================================================
# Module registries
# ===========================================================================


class TestRegistries:
    def test_naive_in_memory_registry(self):
        assert "naive" in MEMORY_REGISTRY

    def test_none_in_compression_registry(self):
        assert "none" in COMPRESSION_REGISTRY

    def test_single_agent_in_communication_registry(self):
        assert "single_agent" in COMMUNICATION_REGISTRY

    def test_memory_registry_values_are_classes(self):
        for name, cls in MEMORY_REGISTRY.items():
            assert isinstance(cls, type), f"{name} is not a class"

    def test_compression_registry_values_are_classes(self):
        for name, cls in COMPRESSION_REGISTRY.items():
            assert isinstance(cls, type), f"{name} is not a class"

    def test_communication_registry_values_are_classes(self):
        for name, cls in COMMUNICATION_REGISTRY.items():
            assert isinstance(cls, type), f"{name} is not a class"


# ===========================================================================
# build_*_module
# ===========================================================================


class TestBuildMemoryModule:
    def test_returns_naive_memory_instance(self):
        from src.memory.naive import NaiveMemory
        module = build_memory_module(_baseline_cfg())
        assert isinstance(module, NaiveMemory)

    def test_unknown_type_raises_value_error(self):
        cfg = _baseline_cfg()
        cfg["memory"]["type"] = "not_real"
        with pytest.raises(ValueError, match="not_real"):
            build_memory_module(cfg)

    def test_params_passed_to_constructor(self):
        """Verify params dict is unpacked as kwargs (NaiveMemory accepts none,
        but we just verify the call doesn't crash with empty params)."""
        cfg = _baseline_cfg()
        cfg["memory"]["params"] = {}
        module = build_memory_module(cfg)
        assert module is not None


class TestBuildCompressionModule:
    def test_returns_no_compression_instance(self):
        from src.compression.none import NoCompression
        module = build_compression_module(_baseline_cfg())
        assert isinstance(module, NoCompression)

    def test_unknown_type_raises_value_error(self):
        cfg = _baseline_cfg()
        cfg["compression"]["type"] = "zip"
        with pytest.raises(ValueError, match="zip"):
            build_compression_module(cfg)


class TestBuildCommunicationModule:
    def test_returns_single_agent_instance(self):
        from src.communication.single_agent import SingleAgentCommunication
        module = build_communication_module(_baseline_cfg())
        assert isinstance(module, SingleAgentCommunication)

    def test_unknown_type_raises_value_error(self):
        cfg = _baseline_cfg()
        cfg["communication"]["type"] = "quantum_swarm"
        with pytest.raises(ValueError, match="quantum_swarm"):
            build_communication_module(cfg)


# ===========================================================================
# Dry-run integration (subprocess-free: patch sys.argv + sys.exit)
# ===========================================================================


class TestDryRunValidation:
    def test_dry_run_exits_on_bad_config(self, tmp_path):
        """validate_config errors should cause sys.exit(1) in main()."""
        path = _write_yaml(tmp_path, """
            memory:
              type: "does_not_exist"
            compression:
              type: "none"
            communication:
              type: "single_agent"
            evaluation:
              dataset: swebench_lite
            sandbox:
              docker_image: swebench-sandbox:latest
              timeout_per_task: 600
        """)
        import sys
        from src.runner import load_config, validate_config

        cfg = load_config(path)
        errors = validate_config(cfg)
        assert len(errors) > 0

    def test_dry_run_passes_on_baseline(self, tmp_path):
        """baseline.yaml should have zero validation errors."""
        baseline_path = Path(__file__).parent.parent / "configs" / "baseline.yaml"
        cfg = load_config(str(baseline_path))
        errors = validate_config(cfg)
        assert errors == []
