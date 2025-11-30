"""
Setup script for the Economic Capital Simulator package

Purpose:
- package and install the Economic Capital Simulator
- load runtime dependencies automatically from requirements.txt
- expose the library for local development and CI/test runners

Edit metadata (name, version, author, email, url) before publishing
"""

from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent


def load_requirements(fname: str = "requirements.txt"):
    path = ROOT / fname
    if not path.exists():
        return []
    with path.open("r", encoding="utf8") as fh:
        return [
            ln.strip()
            for ln in fh.readlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]


def load_readme(fname: str = "README.md"):
    path = ROOT / fname
    if not path.exists():
        return ""
    return path.read_text(encoding="utf8")


setup(
    name="economic-capital-simulator",
    version="0.1.0",
    description="Economic Capital Simulator: Monte Carlo platform for Market, Credit and Operational risk",
    long_description=load_readme(),
    long_description_content_type="text/markdown",
    author="Ajayvir Khara",
    author_email="ajayvirkhara@hotmail.com",
    url="https://github.com/ajayvirkhara/Economic-Capital-Simulator",
    packages=find_packages(exclude=("tests", "docs", ".venv")),
    install_requires=load_requirements(),
    include_package_data=True,
    license="MIT",
    python_requires=">=3.11",
)
