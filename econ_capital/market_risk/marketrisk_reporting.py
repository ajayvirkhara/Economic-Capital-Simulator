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
        self._create_top10_sheet(wb)

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

    def _create_waterfall_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Waterfall", 3)
        ws["A1"] = "Capital Uplift Waterfall"
        ws["A1"].font = Font(size=16, bold=True, color=self.colors["header"])

        baseline = self.results.get("baseline_capital", 0)
        breakdown = self.results["capital_breakdown"]
        top10_positions = breakdown.head(10).index.tolist()

        data = [["Baseline", baseline]]
        for pos in top10_positions:
            uplift = breakdown.get(pos, 0)
            data.append([pos, uplift])

        for r, (name, val) in enumerate(data, 2):
            ws[f"A{r}"] = name
            ws[f"B{r}"] = val
            ws[f"B{r}"].number_format = "£#,##0"

        chart = BarChart()
        chart.title = "Top 10 Capital Impacts"
        chart.y_axis.title = "Capital (£)"
        chart.height = 14
        chart.width = 24
        data_ref = Reference(ws, min_col=2, min_row=1, max_row=len(data))
        cats = Reference(ws, min_col=1, min_row=2, max_row=len(data))
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats)
        ws.add_chart(chart, "D2")

    def _create_top10_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Top 10 Positions", 4)
        ws["A1"] = "Top 10 Capital Contributors"
        ws["A1"].font = Font(size=16, bold=True, color=self.colors["header"])

        headers = ["Rank", "Position", "Capital Contribution (£)"]
        for c, h in enumerate(headers, 1):
            cell = ws.cell(1, c, h)
            cell.fill = PatternFill("solid", self.colors["header"])
            cell.font = Font(color="FFFFFF", bold=True)

        breakdown = self.results["capital_breakdown"]
        for i, (pos, contrib) in enumerate(breakdown.head(10).items(), 1):
            ws.cell(i + 1, 1, i)
            ws.cell(i + 1, 2, pos)
            ws.cell(i + 1, 3, f"£{contrib:,.0f}").number_format = "£#,##0"
            if i <= 3:
                for c in range(1, 4):
                    ws.cell(i + 1, c).fill = PatternFill("solid", self.colors["gold"])

        if len(breakdown) > 0:
            last_row = min(len(breakdown), 10) + 1
            table_range = f"A1:C{last_row}"
            tab = Table(displayName="Top10Positions", ref=table_range)
            tab.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2", showRowStripes=True
            )
            ws.add_table(tab)
        else:
            ws["A2"] = "No capital breakdown available"


def generate_market_risk_report(
    config: Dict[str, Any],
    engine: MarketRiskEconomicCapital,
    results: Dict[str, Any],
    output_dir: str = "reports",
) -> Path:
    return MarketRiskReporter(
        engine, results, config, output_dir
    ).generate_full_report()
