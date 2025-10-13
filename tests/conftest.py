import logging
import pytest
from econ_capital.utils import setup_logging, set_global_seed


@pytest.fixture(scope="session", autouse=True)
def _init_logging_and_seed():
    """
    Global Pytest fixture that runs once per test session.

    Purpose:
    --------
    - Sets up consistent logging for all test modules
    - Seeds all RNGs for reproducible Monte Carlo results
    """
    setup_logging()
    set_global_seed(42)
    yield  # all tests run after this line
    logging.getLogger(__name__).info("All tests completed successfully.")
