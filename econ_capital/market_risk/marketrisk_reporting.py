"""
Regulatory-grade reporting for Market Risk Economic Capital.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.table import Table, TableStyleInfo

from .engine import MarketRiskEconomicCapital

from econ_capital.reporting_utils import apply_clean_style, autofit_columns


class MarketRiskReporter:
    def __init__(
        self,
        engine: MarketRiskEconomicCapital,
        results: Dict[str, Any],
        config: Dict[str, Any],
        output_dir: str | Path = "reports",
    ):
        self.engine = engine
        self.results = results
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filename = (
            self.output_dir
            / f"MarketRisk_EC_Report_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        )

        self.colors = self.config.get("reporting_style", {}).get(
            "colors", self._default_colors()
        )
        # Ensure fallback for required keys
        default_colors = self._default_colors()
        if "gold" not in self.colors:
            self.colors["gold"] = default_colors["gold"]
        if "header" not in self.colors:
            self.colors["header"] = default_colors["header"]

    def _default_colors(self):
        """Standard hardcoded colors for reporting style used as a fallback."""
        return {
            "header": "1F4E78",  # Dark Blue/Grey for headers
            "table_header": "DDEBF7",  # Light Blue/Grey fill
            "gold": "FFD700",  # Gold for top ranks
            "white": "FFFFFF",
        }

    def generate_full_report(self) -> Path:
        wb = Workbook()
        wb.remove(wb.active)

        self._create_cover_sheet(wb)
        self._create_summary_sheet(wb)
        self._create_waterfall_sheet(wb)

        wb.save(self.filename)
        print(f"Report generated: {self.filename}")
        return self.filename

    @property
    def n_paths(self) -> int:
        return self.engine.config.get("n_paths", 500_000)  # fallback to default

    @property
    def cov_method(self) -> str:
        return self.engine.config.get("cov_method", "EWMA").upper()

    def _create_cover_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Cover", 0)
        lines = [
            "MARKET RISK",
            "ECONOMIC CAPITAL",
            "REPORT",
            "",
            f"Run Date: {datetime.now():%Y-%m-%d %H:%M}",
            f"10D VaR (99.9%): £{self.results['var_10d_999']:,.0f}",
            f"10D ES (99.9%): £{self.results['es_10d_999']:,.0f}",
        ]
        for i, line in enumerate(lines, 2):
            cell = ws[f"A{i}"]
            cell.value = line
            cell.font = Font(
                size=24 if i <= 4 else 14, bold=True, color=self.colors["header"]
            )
            cell.alignment = Alignment(horizontal="center")
        ws.merge_cells("A2:F4")
        autofit_columns(ws)

    def _create_summary_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Summary", 1)
        ws["A1"] = "Executive Summary"
        ws["A1"].font = Font(size=18, bold=True, color=self.colors["header"])

        data = [
            ("Run Date", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("10D VaR (99.9%)", f"£{self.results['var_10d_999']:,.0f}"),
            ("10D ES (99.9%)", f"£{self.results['es_10d_999']:,.0f}"),
            ("1Y VaR (99.9%)", f"£{self.results['var_1y_999']:,.0f}"),
            ("1Y ES (99.9%)", f"£{self.results['es_1y_999']:,.0f}"),
            ("Number of Paths", self.n_paths),
            ("Covariance Method", self.cov_method),
        ]
        for i, (k, v) in enumerate(data, 3):
            ws[f"A{i}"] = k
            cell = ws[f"B{i}"]
            cell.value = v
            if k == "Number of Paths":
                cell.number_format = "#,##0"
            ws[f"A{i}"].font = Font(bold=True)
        autofit_columns(ws)

    def _create_waterfall_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Waterfall", 3)
        ws["A1"] = "Capital Uplift Waterfall"
        ws["A1"].font = Font(size=16, bold=True, color=self.colors["header"])

        # Prepare Data with Headers
        headers = ["Position", "Capital Contribution (£)"]
        for c, h in enumerate(headers, 1):
            cell = ws.cell(2, c, h)
            cell.fill = PatternFill("solid", self.colors["header"])
            cell.font = Font(color="FFFFFF", bold=True)

        baseline = self.results.get("baseline_capital", 0)
        breakdown = self.results["capital_breakdown"]

        # Starting data at row 3
        ws.cell(3, 1, "Baseline")
        ws.cell(3, 2, baseline).number_format = "£#,##0"

        current_row = 4
        for i, (pos, uplift) in enumerate(breakdown.head(10).items(), 1):
            name_cell = ws.cell(current_row, 1, pos)
            val_cell = ws.cell(current_row, 2, uplift)
            val_cell.number_format = "£#,##0"

            # Apply gold highlight to top 3 contributors
            if i <= 3:
                gold_fill = PatternFill("solid", fgColor=self.colors["gold"])
                name_cell.fill = gold_fill
                val_cell.fill = gold_fill

            current_row += 1

        table_range = f"A2:B{current_row - 1}"
        tab = Table(displayName="WaterfallTable", ref=table_range)
        tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showRowStripes=True
        )
        ws.add_table(tab)

        # Create chart
        chart = BarChart()
        chart.title = "Top Capital Impacts"
        chart.height = 14
        chart.width = 24
        data_ref = Reference(ws, min_col=2, min_row=3, max_row=current_row - 1)
        cats = Reference(ws, min_col=1, min_row=3, max_row=current_row - 1)
        chart.add_data(data_ref, titles_from_data=False)
        chart.set_categories(cats)

        # Calculate number of points to plot
        num_plot_points = 1 + len(breakdown.head(10))

        apply_clean_style(
            chart,
            "Capital Contribution (£)",
            num_points=num_plot_points,
        )

        ws.add_chart(chart, "D2")

        autofit_columns(ws)


def generate_market_risk_report(
    config: Dict[str, Any],
    engine: MarketRiskEconomicCapital,
    results: Dict[str, Any],
    output_dir: str = "reports",
) -> Path:
    return MarketRiskReporter(
        engine, results, config, output_dir
    ).generate_full_report()
