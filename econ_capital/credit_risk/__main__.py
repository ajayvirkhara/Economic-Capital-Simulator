"""
Command-line entry point for Credit Risk module.
Run with:
    python -m econ_capital.credit_risk
"""

from econ_capital.utils import setup_logging, timed_section
from econ_capital.credit_risk.demo_exposure import main as demo_exposure_main


def main():
    setup_logging(level="INFO")
    logger = setup_logging(__name__)
    logger.info("Running Credit Risk demo")
    with timed_section("credit_risk_demo"):
        demo_exposure_main()
    logger.info("Credit Risk simulation finished successfully.")


if __name__ == "__main__":
    main()
