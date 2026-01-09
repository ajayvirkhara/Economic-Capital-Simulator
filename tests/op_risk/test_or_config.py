import pytest
import importlib
import econ_capital.op_risk.config
from econ_capital.op_risk.config import OpRiskConfig

importlib.reload(econ_capital.op_risk.config)


def test_file_not_found_raises():
    """Non-existent file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        OpRiskConfig("this_file_does_not_exist.yaml")


@pytest.fixture
def valid_config_path(tmp_path):
    config_content = """
op_risk:
  frequency:
    lambda: 5.0
  severity:
    sigma: 1.2
  scenarios: {}
  insurance: {}
  stress_tests: {}
"""
    p = tmp_path / "valid.yaml"
    p.write_text(config_content)
    return str(p)


def test_properties_are_accessible(valid_config_path):
    """Check all @property getters work."""
    cfg = OpRiskConfig(valid_config_path)
    assert cfg.frequency["lambda"] == 5.0
    assert cfg.severity["sigma"] == 1.2
    assert cfg.scenarios == {}
    assert cfg.insurance == {}
    assert cfg.stress_tests == {}


def test_update_modifies_config(valid_config_path):
    """Ensure .update() works."""
    cfg = OpRiskConfig(valid_config_path)
    cfg.update({"frequency": {"lambda": 10.0}})
    assert cfg.frequency["lambda"] == 10.0


def test_validate_passes_with_valid_data(valid_config_path):
    """Validate() does not raise on valid config."""
    cfg = OpRiskConfig(valid_config_path)
    cfg.validate()  # should not raise
