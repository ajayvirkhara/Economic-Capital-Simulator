"""
Unit tests for econ_capital/config_loader.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from econ_capital.config_loader import merge_with_global


@pytest.fixture
def temp_default_yaml(tmp_path: Path) -> Path:
    """Create temporary default.yaml and monkeypatch PROJECT_ROOT to use it."""
    yaml_path = tmp_path / "default.yaml"

    def _write(content: dict) -> None:
        yaml_path.write_text(json.dumps(content))

    yield yaml_path, _write

    # Cleanup: reload module to reset GLOBAL_DEFAULTS if needed in other tests
    import importlib
    import econ_capital.config_loader

    importlib.reload(econ_capital.config_loader)


def test_global_defaults_loaded_correctly(temp_default_yaml):
    yaml_path, write = temp_default_yaml
    content = {
        "global": {"seed": 999},
        "simulation": {"default_n_paths": 10000},
        "other": "value",
    }
    write(content)

    # Force reload
    from econ_capital import config_loader
    import importlib

    importlib.reload(config_loader)

    assert config_loader.GLOBAL_DEFAULTS == content


def test_global_defaults_empty_when_missing(temp_default_yaml):
    yaml_path, _ = temp_default_yaml  # file doesn't exist

    from econ_capital import config_loader
    import importlib

    importlib.reload(config_loader)

    assert config_loader.GLOBAL_DEFAULTS == {}


@pytest.mark.parametrize(
    "global_defaults, module_config, expected_seed, expected_sim_paths",
    [
        # Global seed promoted, simulation merged
        (
            {"global": {"seed": 42}, "simulation": {"default_n_paths": 5000}},
            {"simulation": {"default_n_paths": 20000}},
            42,
            20000,
        ),
        # No global section → no seed promotion
        (
            {"simulation": {"default_n_paths": 10000}},
            {"simulation": {"var_q": 0.995}},
            None,
            10000,
        ),
        # Empty module config → full global + promoted seed
        (
            {"global": {"seed": 123}, "simulation": {"default_n_paths": 30000}},
            {},
            123,
            30000,
        ),
        # Module adds new simulation keys
        (
            {"global": {"seed": 456}},
            {"simulation": {"var_q": 0.995}},
            456,
            None,
        ),
    ],
)
def test_merge_with_global(
    global_defaults,
    module_config,
    expected_seed,
    expected_sim_paths,
    monkeypatch,
):
    monkeypatch.setattr("econ_capital.config_loader.GLOBAL_DEFAULTS", global_defaults)

    result = merge_with_global(module_config)

    if expected_seed is not None:
        assert result.get("seed") == expected_seed
    else:
        assert "seed" not in result

    if expected_sim_paths is not None:
        assert result.get("simulation", {}).get("default_n_paths") == expected_sim_paths


def test_merge_preserves_other_keys(monkeypatch):
    global_defaults = {"logging": {"level": "INFO"}, "global": {"seed": 789}}
    module_config = {"logging": {"format": "%(message)s"}, "risk": {"horizon": 1}}

    monkeypatch.setattr("econ_capital.config_loader.GLOBAL_DEFAULTS", global_defaults)

    result = merge_with_global(module_config)

    assert result["logging"] == {"level": "INFO", "format": "%(message)s"}
    assert result["risk"] == {"horizon": 1}
    assert result["seed"] == 789
