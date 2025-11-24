"""
Dependencies module for operational risk modelling.
Defines enums, parameter dataclasses, defaults, and custom exceptions.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import TypedDict, Optional


# ----------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------


class FrequencyModelType(Enum):
    POISSON = "poisson"
    NEG_BINOMIAL = "negative_binomial"


class SeverityModelType(Enum):
    LOGNORMAL = "lognormal"
    GPD = "gpd"
    LOGNORMAL_GPD = "lognormal_gpd"
    LOGLOGISTIC = "loglogistic"


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------


class DataIssueError(Exception):
    """Raised when operational risk data is missing, invalid, or inconsistent."""


class ModelFittingError(Exception):
    """Raised when parameter estimation for frequency or severity models fails."""


# ----------------------------------------------------------------------
# Dataclasses for model parameters
# ----------------------------------------------------------------------


@dataclass
class FrequencyParams:
    model: FrequencyModelType
    lambda_hat: float
    k: Optional[float] = None
    p: Optional[float] = None


@dataclass
class SeverityParams:
    model: SeverityModelType
    lognormal_mu: Optional[float] = None
    lognormal_sigma: Optional[float] = None
    gpd_xi: Optional[float] = None
    gpd_beta: Optional[float] = None
    threshold: Optional[float] = None


# ----------------------------------------------------------------------
# UoM configuration structure
# ----------------------------------------------------------------------


@dataclass
class UoMConfig:
    uom: str
    frequency_model: "FrequencyModelType"
    severity_model: "SeverityModelType"
    frequency_params: "FrequencyParams"
    severity_params: "SeverityParams"
    insurance_limit: float = 5_000_000
    deductible: float = 100_000


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

DEFAULT_LAMBDA = 1.0
DEFAULT_GPD_THRESHOLD = 10000.0


# ----------------------------------------------------------------------
# Config dictionary type for testing
# ----------------------------------------------------------------------


class ConfigDict(TypedDict, total=False):
    name: str
    frequency_model: str
    severity_model: str
    lambda_: float
    threshold: float
