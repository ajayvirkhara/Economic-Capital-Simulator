from pathlib import Path
import yaml


class OpRiskConfig:
    """Load and provide programmatic access to operational risk parameters"""

    def __init__(self, config_file: str = "config/op_config.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_yaml()

    def _load_yaml(self):
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        with open(self.config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def as_dict(self):
        return self.config.get("op_risk", self.config)

    def update(self, updates: dict):
        target = self.config.get("op_risk", self.config)
        target.update(updates)

    @property
    def frequency(self):
        return self.config["op_risk"]["frequency"]

    @property
    def severity(self):
        return self.config["op_risk"]["severity"]

    @property
    def scenarios(self):
        return self.config["op_risk"]["scenarios"]

    @property
    def insurance(self):
        return self.config["op_risk"]["insurance"]

    @property
    def stress_tests(self):
        return self.config["op_risk"]["stress_tests"]

    def validate(self):
        freq = self.config.get("op_risk", {}).get("frequency", {})
        sev = self.config.get("op_risk", {}).get("severity", {})
        assert freq.get("lambda", 0) > 0, "Frequency lambda must be > 0"
        assert sev.get("sigma", 0) > 0, "Severity sigma must be > 0"
