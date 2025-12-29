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
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.layout import Layout, ManualLayout

from econ_capital.aggregate import aggregate_economic_capital


class FirmWideECReporter:
    def __init__(
        self,
        aggregated_results: Dict[str, Any],
        output_dir: str | Path = "reports",
    ):
        self.results = aggregated_results
        self.market_details = self.results.get("market_details", {})
        self.credit_details = self.results.get("credit_details", {})
        self.op_details = self.results.get("op_details", {})
        self.correlations = self.results.get("correlations", {})
        self.output_dir = Path(output_dir) / "firmwide"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp  # Added for unique table names
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
        self._create_detailed_market_sheet(wb)
        self._create_detailed_credit_sheet(wb)
        self._create_detailed_oprisk_sheet(wb)
        self._create_correlation_matrix_sheet(wb)
        self._create_sensitivity_analysis_sheet(wb)

        wb.save(self.filename)
        wb.close()  # Added to prevent potential file corruption
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

        standalone_per_risk = {
            risk: vals["EL"] + norm.ppf(0.999) * vals["UL"]
            for risk, vals in self.results["individual_risks"].items()
        }
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
            ("Standalone Market EC", f"£{standalone_per_risk.get('Market', 0):,.0f}"),
            ("Standalone Credit EC", f"£{standalone_per_risk.get('Credit', 0):,.0f}"),
            ("Standalone OpRisk EC", f"£{standalone_per_risk.get('OpRisk', 0):,.0f}"),
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
        df = df.map(lambda x: f"£{x:,.0f}" if isinstance(x, (int, float)) else x)

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
        tab = Table(
            displayName=f"RiskContributions_{self.timestamp}",
            ref=f"A3:{last_col}{last_row}",
        )
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
        tab = Table(
            displayName=f"MarginalContributions_{self.timestamp}", ref=f"A3:B{last_row}"
        )
        tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showRowStripes=True
        )
        ws.add_table(tab)

    def _create_detailed_market_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Detailed Market Risk", 4)
        ws["A1"] = "Detailed Market Risk Breakdown"
        ws["A1"].font = Font(size=16, bold=True, color=self.colors["header"])

        # Top 10 contributors (assuming 'capital_breakdown' in market_details)
        breakdown = self.market_details.get("capital_breakdown", pd.Series())
        if breakdown.empty:
            ws["A3"] = "No market risk details available."
            return

        headers = ["Rank", "Position", "Capital Contribution (£)"]
        for c, h in enumerate(headers, 1):
            cell = ws.cell(3, c, h)
            cell.fill = PatternFill("solid", self.colors["header"])
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")

        top_n = breakdown.head(10)
        num_items = len(top_n)
        for i, (pos, contrib) in enumerate(top_n.items(), 4):
            ws.cell(i, 1, i - 3)
            ws.cell(i, 2, pos)
            ws.cell(i, 3, f"£{contrib:,.0f}").number_format = "£#,##0"
            if i - 3 <= 3:  # Highlight top 3
                for c in range(1, 4):
                    ws.cell(i, c).fill = PatternFill("solid", self.colors["gold"])

        # Waterfall chart
        chart = BarChart()
        chart.title = "Top 10 Market Capital Impacts"
        chart.y_axis.title = "Capital (£)"
        chart.height = 12
        chart.width = 18
        max_row = 3 + num_items
        data = Reference(ws, min_col=3, min_row=3, max_row=max_row)
        cats = Reference(ws, min_col=2, min_row=4, max_row=max_row)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        ws.add_chart(chart, "E2")

        # Table
        if len(breakdown) > 0:
            last_row = 3 + num_items
            tab = Table(
                displayName=f"MarketTop10_{self.timestamp}", ref=f"A3:C{last_row}"
            )
            tab.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2", showRowStripes=True
            )
            ws.add_table(tab)

    def _create_detailed_credit_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Detailed Credit Risk", 5)
        ws["A1"] = "Detailed Credit Risk Portfolio"
        ws["A1"].font = Font(size=16, bold=True, color=self.colors["header"])

        # Full portfolio table (assuming 'full_data' in credit_details as pd.DataFrame)
        df = self.credit_details.get("full_data", pd.DataFrame())
        if df.empty:
            ws["A3"] = "No credit risk details available."
            return

        cols = ["name", "Sector", "EAD", "PD", "LGD", "EL", "UL", "EC_Marginal"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        # Write headers
        for c_idx, col_name in enumerate(cols, 1):
            cell = ws.cell(row=3, column=c_idx, value=col_name)
            cell.fill = PatternFill("solid", self.colors["header"])
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")

        # Write data rows
        for r_idx, row in enumerate(df.itertuples(index=False), 4):
            for c_idx, val in enumerate(row, 1):
                cell = ws.cell(row=r_idx, column=c_idx, value=val)
                header = cols[c_idx - 1]
                if isinstance(val, (int, float)):
                    if header == "PD":
                        cell.number_format = "0.00%"
                    elif header == "LGD":
                        cell.number_format = "0.00"
                    elif header in ["EAD", "EL", "UL", "EC_Marginal"]:
                        cell.number_format = "£#,##0"

        # Table
        last_row = len(df) + 3
        last_col = get_column_letter(len(cols))
        tab = Table(
            displayName=f"CreditPortfolio_{self.timestamp}",
            ref=f"A3:{last_col}{last_row}",
        )
        tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showRowStripes=True
        )
        ws.add_table(tab)

    def _create_detailed_oprisk_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Detailed Op Risk", 6)
        ws["A1"] = "Detailed Op Risk Scenarios"
        ws["A1"].font = Font(size=16, bold=True, color=self.colors["header"])

        raw_results = (
            self.op_details.get("results", [])
            if isinstance(self.op_details, dict)
            else []
        )
        expert_details = (
            self.op_details.get("expert_scenario_details", [])
            if isinstance(self.op_details, dict)
            else []
        )

        if not raw_results:
            ws["A3"] = "No op risk details available."
            return

        # Sort and take top 10
        top10 = sorted(
            raw_results, key=lambda x: getattr(x, "uplift_factor", 0), reverse=True
        )[:10]

        headers = ["Rank", "ScenarioID", "Description", "Uplift ×", "Stressed Capital"]
        for c, h in enumerate(headers, 1):
            cell = ws.cell(3, c, h)
            cell.fill = PatternFill("solid", self.colors["header"])
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")

        # Write data rows
        for i, r in enumerate(top10, 4):
            ws.cell(i, 1, int(i - 3))
            ws.cell(i, 2, str(getattr(r, "name", "N/A") or "N/A"))
            ws.cell(i, 3, str(getattr(r, "description", "---") or "---"))

            # Uplift as raw float
            uplift_cell = ws.cell(i, 4, float(getattr(r, "uplift_factor", 0.0)))
            uplift_cell.number_format = "0.00"

            # Capital as raw float
            cap_cell = ws.cell(i, 5, float(getattr(r, "capital_stressed", 0.0)))
            cap_cell.number_format = "#,##0"

            if i - 3 <= 3:  # Top 3 highlighting
                for c in range(1, 6):
                    ws.cell(i, c).fill = PatternFill("solid", self.colors["gold"])

        # Table definition with strictly unique name and checked range
        last_row = len(top10) + 3
        if len(top10) > 0:
            tab = Table(
                displayName=f"OpRiskTop10_{self.timestamp}", ref=f"A3:E{last_row}"
            )

            # Style
            style = TableStyleInfo(
                name="TableStyleMedium2",
                showRowStripes=True,
            )
            tab.tableStyleInfo = style

            # Add to worksheet
            ws.add_table(tab)

        # Expert scenarios
        if expert_details:
            start_expert = last_row + 4
            ws.cell(start_expert, 1, "Expert Judgment Scenarios").font = Font(
                size=14, bold=True, color=self.colors["header"]
            )

            expert_headers = ["Scenario", "Probability", "Impact (£)", "Annual EL (£)"]
            for c, h in enumerate(expert_headers, 1):
                cell = ws.cell(start_expert + 1, c, h)
                cell.fill = PatternFill("solid", self.colors["header"])
                cell.font = Font(color="FFFFFF", bold=True)

            for i, d in enumerate(expert_details, start_expert + 2):
                ws.cell(i, 1, d.get("name", "N/A"))
                ws.cell(i, 2, f"{d.get('probability', 0.0):.1%}")
                ws.cell(i, 3, d.get("impact", 0.0)).number_format = '"£"#,##0'
                ws.cell(i, 4, d.get("annual_el", 0.0)).number_format = '"£"#,##0'

            # Add table for expert scenarios
            expert_last_row = start_expert + 1 + len(expert_details)
            expert_tab = Table(
                displayName=f"ExpertScenarios_{self.timestamp}",
                ref=f"A{start_expert + 1}:D{expert_last_row}",
            )
            expert_tab.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2", showRowStripes=True
            )
            ws.add_table(expert_tab)

    def _create_correlation_matrix_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Correlation Matrix", 7)
        ws["A1"] = "Inter-Risk Correlation Matrix"
        ws["A1"].font = Font(size=16, bold=True, color=self.colors["header"])

        risk_types = list(self.correlations.keys())
        if not risk_types:
            ws["A3"] = "No correlation data available."
            return

        # Headers
        ws.cell(3, 1, "Risk_Type").font = Font(color="FFFFFF", bold=True)
        ws.cell(3, 1).fill = PatternFill("solid", self.colors["header"])

        for c, risk in enumerate(risk_types, 2):
            cell = ws.cell(3, c, risk)
            cell.fill = PatternFill("solid", self.colors["header"])
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")

        # Write data
        for r, rt1 in enumerate(risk_types, 4):
            ws.cell(r, 1, str(rt1)).font = Font(bold=True)
            for c, rt2 in enumerate(risk_types, 2):
                corr = float(self.correlations.get(rt1, {}).get(rt2, 0.0))
                cell = ws.cell(r, c, corr)
                cell.number_format = "0.00"
                cell.alignment = Alignment(horizontal="center")

        last_row = len(risk_types) + 3
        last_col = get_column_letter(len(risk_types) + 1)

        # Use a unique ID and sanitised names
        tab = Table(
            displayName=f"CorrMatrixTable_{self.timestamp}",
            ref=f"A3:{last_col}{last_row}",
            id=700,
        )

        tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium9", showRowStripes=True
        )

        # Use a try-except block to ensure the script finishes even if Excel's XML engine is finicky
        try:
            ws.add_table(tab)
        except Exception:
            pass

    def _create_sensitivity_analysis_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Sensitivity Analysis", 8)
        ws["A1"] = "Sensitivity to Confidence Level"
        ws["A1"].font = Font(size=16, bold=True, color=self.colors["header"])

        levels = [0.99, 0.995, 0.999]

        # Write headers
        ws.cell(3, 1, "Confidence Level").font = Font(color="FFFFFF", bold=True)
        ws.cell(3, 2, "EC Total (£)").font = Font(color="FFFFFF", bold=True)
        for c in [1, 2]:
            ws.cell(3, c).fill = PatternFill("solid", self.colors["header"])

        # Write data rows
        for i, level in enumerate(levels, 4):
            _, _, ec, _, _ = aggregate_economic_capital(
                self.results["individual_risks"], confidence_level=level
            )
            ws.cell(i, 1, f"{level * 100:.1f}%")
            cell_b = ws.cell(i, 2, ec)
            cell_b.number_format = '"£"#,##0'

        # Table
        last_row = len(levels) + 3
        tab = Table(displayName=f"Sensitivity_{self.timestamp}", ref=f"A3:B{last_row}")
        tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showRowStripes=True
        )
        ws.add_table(tab)

        # Chart
        chart = BarChart()
        chart.type = "col"
        chart.style = 10
        chart.title = "EC Sensitivity to Confidence Level"
        chart.y_axis.title = "Economic Capital (£)"
        chart.x_axis.title = "Confidence Level"
        chart.height = 12
        chart.width = 18

        # Data and Categories
        data = Reference(ws, min_col=2, min_row=4, max_row=last_row)
        cats = Reference(ws, min_col=1, min_row=4, max_row=last_row)
        chart.add_data(data, titles_from_data=False)
        chart.set_categories(cats)

        # Axis Labels and Lines
        chart.y_axis.title = "Economic Capital (£)"
        chart.x_axis.title = "Confidence Level"
        chart.x_axis.delete = False  # Explicitly show X-axis labels
        chart.y_axis.delete = False
        chart.layout = Layout(manualLayout=ManualLayout(x=0.15, y=0.1, w=0.75, h=0.75))

        # Add Tick Marks and Gridlines for better readability
        chart.y_axis.majorGridlines = None
        chart.y_axis.majorTickMark = "out"
        chart.x_axis.tickLblPos = "low"

        # Fix Data Labels
        chart.dataLabels = DataLabelList()
        chart.dataLabels.showVal = True  # Show the £ value
        chart.dataLabels.showCatName = False  # Hide the "99.0%" text in the label
        chart.dataLabels.showSerName = False  # Hide the "Series1" text
        chart.dataLabels.showLegendKey = False  # Removes coloured squares
        chart.dataLabels.position = "outEnd"  # Place label above the bar

        # Final polish
        chart.legend = None  # Legend is unnecessary since labels are on the X-axis
        chart.varyColors = True

        ws.add_chart(chart, "D2")


def generate_firmwide_ec_report(
    aggregated_results: Dict[str, Any],
    output_dir: str | Path = "reports",
) -> Path:
    """Convenience function to generate the firm-wide report."""
    return FirmWideECReporter(aggregated_results, output_dir).generate_report()
