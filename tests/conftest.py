"""
Pytest configuration.

Integration tests (marked @pytest.mark.integration) require external services
(Docker daemon, Anthropic API key). They are skipped by default and can be
opted into explicitly:

    pytest -m integration           # run only integration tests
    pytest -m "not integration"     # run only unit tests (default)
    pytest                          # skips integration tests automatically
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip integration-marked tests unless -m integration is passed explicitly."""
    # If the user explicitly selected a marker expression, don't interfere.
    marker_expr = config.option.markexpr
    if marker_expr and "integration" in marker_expr:
        return

    skip_integration = pytest.mark.skip(
        reason="Integration test: pass -m integration to run"
    )
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip_integration)
