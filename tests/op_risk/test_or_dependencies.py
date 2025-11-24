from typing import get_origin, get_args
import pytest

from econ_capital.op_risk.dependencies import (
    FrequencyModelType,
    SeverityModelType,
    DataIssueError,
    ModelFittingError,
    FrequencyParams,
    SeverityParams,
    UoMConfig,
    DEFAULT_LAMBDA,
    DEFAULT_GPD_THRESHOLD,
    ConfigDict,
)


# ---------------------------------------------------------
# Enums
# ---------------------------------------------------------


def test_frequency_model_type_enum():
    assert FrequencyModelType.POISSON.value == "poisson"
    assert isinstance(FrequencyModelType.POISSON, FrequencyModelType)


def test_severity_model_type_enum():
    assert SeverityModelType.LOGNORMAL_GPD.value == "lognormal_gpd"
    assert isinstance(SeverityModelType.LOGNORMAL_GPD, SeverityModelType)


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------


def test_data_issue_error_inheritance():
    with pytest.raises(DataIssueError):
        raise DataIssueError("test")


def test_model_fitting_error_inheritance():
    with pytest.raises(ModelFittingError):
        raise ModelFittingError("bad fitting")


# ---------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------


def test_frequency_params_dataclass():
    fp = FrequencyParams(model=FrequencyModelType.POISSON, lambda_hat=2.5)
    assert fp.model == FrequencyModelType.POISSON
    assert fp.lambda_hat == 2.5


def test_severity_params_dataclass():
    sp = SeverityParams(
        model=SeverityModelType.LOGNORMAL_GPD,
        lognormal_mu=1.0,
        lognormal_sigma=0.5,
        gpd_xi=0.2,
        gpd_beta=10000,
        threshold=20000,
    )
    assert sp.threshold == 20000
    assert sp.gpd_beta == 10000
    assert sp.model == SeverityModelType.LOGNORMAL_GPD


def test_uom_config_defaults():
    cfg = UoMConfig(
        uom="Payments",
        frequency_model=FrequencyModelType.POISSON,
        severity_model=SeverityModelType.LOGNORMAL_GPD,
        frequency_params=FrequencyParams(
            model=FrequencyModelType.POISSON, lambda_hat=1.0
        ),
        severity_params=SeverityParams(
            model=SeverityModelType.LOGNORMAL_GPD,
            lognormal_mu=1.0,
            lognormal_sigma=0.5,
            gpd_xi=0.1,
            gpd_beta=10000,
            threshold=10000,
        ),
    )
    assert cfg.uom == "Payments"
    assert cfg.insurance_limit > 0
    assert cfg.deductible >= 0


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------


def test_default_lambda_constant():
    assert isinstance(DEFAULT_LAMBDA, float)
    assert DEFAULT_LAMBDA > 0


def test_default_threshold_constant():
    assert isinstance(DEFAULT_GPD_THRESHOLD, (int, float))
    assert DEFAULT_GPD_THRESHOLD > 0


# ---------------------------------------------------------
# Type aliases
# ---------------------------------------------------------


def test_config_dict_alias():
    origin = get_origin(ConfigDict)
    assert origin is dict

    key_type, val_type = get_args(ConfigDict)
    assert key_type is str
    assert val_type is object
