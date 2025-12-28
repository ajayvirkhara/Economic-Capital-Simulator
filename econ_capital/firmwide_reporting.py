"""
Firm-Wide Economic Capital Report — Regulatory-grade Excel output
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from scipy.stats import norm

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


class FirmWideECReporter:
    def __init__(
        self,
        aggregated_results: Dict[str, Any],
        output_dir: str | Path = "reports",
    ):
        self.results = aggregated_results
        self.output_dir = Path(output_dir) / "firmwide"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = self.output_dir / f"FirmWide_EC_Report_{timestamp}.xlsx"

        # Color scheme
        self.colors = {
            "header": "1F4E78",
            "table_header": "DDEBF7",
            "gold": "FFD700",
            "light_gold": "FFF2CC",
            "white": "FFFFFF",
        }

    def generate_report(self) -> Path:
        wb = Workbook()
        wb.remove(wb.active)

        self._create_cover_sheet(wb)
        self._create_summary_sheet(wb)
        self._create_risk_contributions_sheet(wb)
        self._create_marginal_waterfall_sheet(wb)

        wb.save(self.filename)
        print("\nFirm-wide Economic Capital report generated:")
        print(f"   {self.filename}")
        return self.filename

    def _create_cover_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Cover", 0)

        lines = [
            "FIRM-WIDE",
            "ECONOMIC CAPITAL",
            "REPORT",
            "",
            f"Run Date: {datetime.now():%Y-%m-%d %H:%M}",
            f"Total Economic Capital (99.9%): £{self.results['EC_total']:,.0f}",
            f"Diversification Benefit: £{self.results['diversification_benefit']:,.0f}",
        ]

        for i, line in enumerate(lines, 2):
            cell = ws[f"A{i}"]
            cell.value = line
            cell.font = Font(
                size=28 if i <= 3 else 16,
                bold=True,
                color=self.colors["header"],
            )
            cell.alignment = Alignment(horizontal="center", vertical="center")

        ws.merge_cells("A2:G6")

    def _create_summary_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Executive Summary", 1)
        ws["A1"] = "Firm-Wide Economic Capital Summary"
        ws["A1"].font = Font(size=18, bold=True, color=self.colors["header"])

        data = [
            ("Run Timestamp", self.results["run_timestamp"]),
            ("Total Expected Loss (EL)", f"£{self.results['EL_total']:,.0f}"),
            ("Portfolio Unexpected Loss (UL)", f"£{self.results['UL_portfolio']:,.0f}"),
            ("Total Economic Capital (99.9%)", f"£{self.results['EC_total']:,.0f}"),
            (
                "Diversification Benefit",
                f"£{self.results['diversification_benefit']:,.0f}",
            ),
            (
                "Standalone Sum (No Diversification)",
                f"£{self.results['EC_total'] + self.results['diversification_benefit']:,.0f}",
            ),
        ]

        for i, (label, value) in enumerate(data, 3):
            ws[f"A{i}"] = label
            ws[f"B{i}"] = value
            ws[f"A{i}"].font = Font(bold=True)
            ws[f"B{i}"].number_format = (
                "#,##0" if isinstance(value, str) and "£" in value else ""
            )

    def _create_risk_contributions_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Risk Contributions", 2)
        ws["A1"] = "Individual Risk Contributions"
        ws["A1"].font = Font(size=16, bold=True, color=self.colors["header"])

        df = pd.DataFrame(self.results["individual_risks"]).T
        df = df[["EL", "UL"]]
        df["Total EC (Standalone)"] = df["EL"] + norm.ppf(0.999) * df["UL"]
        df = df.applymap(lambda x: f"£{x:,.0f}" if isinstance(x, (int, float)) else x)

        headers = [
            "Risk Type",
            "Expected Loss (EL)",
            "Unexpected Loss (UL)",
            "Standalone EC",
        ]
        for c, h in enumerate(headers, 1):
            cell = ws.cell(3, c, h)
            cell.fill = PatternFill("solid", self.colors["header"])
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")

        for r, (risk, row) in enumerate(df.iterrows(), 4):
            ws.cell(r, 1, risk)
            for c, val in enumerate(row, 2):
                ws.cell(r, c, val)

        # Format as table
        last_row = len(df) + 3
        last_col = get_column_letter(len(headers))
        tab = Table(displayName="RiskContributions", ref=f"A3:{last_col}{last_row}")
        tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium9", showRowStripes=True
        )
        ws.add_table(tab)

    def _create_marginal_waterfall_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Marginal Contributions", 3)
        ws["A1"] = "Marginal Economic Capital Contributions"
        ws["A1"].font = Font(size=16, bold=True, color=self.colors["header"])

        marginal = self.results["marginal_contributions"]
        df = (
            pd.Series(marginal).sort_values(ascending=False).to_frame("Marginal EC (£)")
        )

        # Write data
        ws["A3"] = "Risk Type"
        ws["B3"] = "Marginal Contribution"
        for cell in ws[3]:
            cell.fill = PatternFill("solid", self.colors["header"])
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")

        for r, (risk, contrib) in enumerate(df.itertuples(), 4):
            ws.cell(r, 1, risk)
            ws.cell(r, 2, contrib)
            ws.cell(r, 2).number_format = "£#,##0"
            if r <= 6:  # Top 3 gold highlight
                for c in [1, 2]:
                    ws.cell(r, c).fill = PatternFill(
                        "solid",
                        self.colors["gold"] if r <= 5 else self.colors["light_gold"],
                    )

        # Bar chart
        chart = BarChart()
        chart.type = "col"
        chart.style = 10
        chart.title = "Marginal Economic Capital by Risk Type"
        chart.y_axis.title = "Contribution (£)"
        chart.x_axis.title = "Risk Type"

        data = Reference(ws, min_col=2, min_row=3, max_row=len(df) + 3)
        cats = Reference(ws, min_col=1, min_row=4, max_row=len(df) + 3)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        chart.height = 12
        chart.width = 20
        ws.add_chart(chart, "D2")

        # Table
        last_row = len(df) + 3
        tab = Table(displayName="MarginalContributions", ref=f"A3:B{last_row}")
        tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showRowStripes=True
        )
        ws.add_table(tab)


def generate_firmwide_ec_report(
    aggregated_results: Dict[str, Any],
    output_dir: str | Path = "reports",
) -> Path:
    """Convenience function to generate the firm-wide report."""
    return FirmWideECReporter(aggregated_results, output_dir).generate_report()
