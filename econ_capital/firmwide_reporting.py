"""
Firm-Wide Economic Capital Report — Regulatory-grade Excel output
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from scipy.stats import norm, t

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from econ_capital.reporting_utils import apply_clean_style, autofit_columns
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

        # Protect sheet for regulatory compliance
        ws.protection.sheet = True
        ws.protection.password = "ec_report"

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
        autofit_columns(ws)

    def _create_summary_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Executive Summary", 1)

        # Protect sheet for regulatory compliance
        ws.protection.sheet = True
        ws.protection.password = "ec_report"

        ws["A1"] = "Firm-Wide Economic Capital Summary"
        ws["A1"].font = Font(size=18, bold=True, color=self.colors["header"])

        copula_df = 7.0  # Match aggregate.py
        t_quantile = t.ppf(0.999, copula_df)
        standalone_per_risk = {
            risk: vals["EL"] + t_quantile * vals["UL"]
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
            if label != "Run Timestamp":
                ws[f"B{i}"].number_format = '"£"#,##0'
        ws.append(
            [
                "Note",
                "Marginal contributions are post-diversification and do not sum exactly to total EC",
            ]
        )
        autofit_columns(ws)

    def _create_risk_contributions_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Risk Contributions", 2)

        # Protect sheet for regulatory compliance
        ws.protection.sheet = True
        ws.protection.password = "ec_report"

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
                cell = ws.cell(r, c, val)
                cell.number_format = '"£"#,##0'

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
        autofit_columns(ws)

    def _create_marginal_waterfall_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Marginal Contributions", 3)

        # Protect sheet for regulatory compliance
        ws.protection.sheet = True
        ws.protection.password = "ec_report"

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
            ws.cell(r, 2).number_format = '"£"#,##0'
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
        chart.x_axis.title = "Risk Type"

        # Data
        data = Reference(ws, min_col=2, min_row=4, max_row=3 + len(df))
        cats = Reference(ws, min_col=1, min_row=4, max_row=3 + len(df))
        chart.add_data(data, titles_from_data=False)
        if chart.series:
            chart.series[0].name = ""  # Explicitly set series name to empty string
        chart.set_categories(cats)

        apply_clean_style(chart, "Marginal EC (£)")
        ws.add_chart(chart, "E2")

        # Table
        last_row = len(df) + 3
        tab = Table(
            displayName=f"MarginalContributions_{self.timestamp}", ref=f"A3:B{last_row}"
        )
        tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showRowStripes=True
        )
        ws.add_table(tab)
        autofit_columns(ws)

    def _create_detailed_market_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Detailed Market Risk", 4)

        # Protect sheet for regulatory compliance
        ws.protection.sheet = True
        ws.protection.password = "ec_report"

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

        top_n = breakdown
        num_items = len(top_n)

        # Define variables for row tracking
        start_top10 = 4
        last_row = start_top10 + num_items - 1

        for i, (pos, contrib) in enumerate(top_n.items(), start_top10):
            ws.cell(i, 1, i - 3)
            ws.cell(i, 2, pos)
            cell_val = ws.cell(i, 3, float(contrib))
            cell_val.number_format = '"£"#,##0'

            if i - 3 <= 3:  # Highlight top 3
                for c in range(1, 4):
                    ws.cell(i, c).fill = PatternFill("solid", self.colors["gold"])

        # Waterfall chart
        chart = BarChart()
        chart.type = "col"
        chart.style = 10
        chart.title = "Top Market Capital Impacts"
        chart.x_axis.title = "Position"

        # Data and categories
        data = Reference(ws, min_col=3, min_row=start_top10, max_row=last_row)
        cats = Reference(ws, min_col=2, min_row=start_top10, max_row=last_row)
        chart.add_data(data, titles_from_data=False)
        chart.set_categories(cats)

        # Y-axis formatting
        chart.y_axis.number_format = '"£"#,##0'

        apply_clean_style(chart, "Capital Contribution (£)")
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
        note_row = ws.max_row + 2  # 2 rows below the last data row
        ws.cell(row=note_row, column=1).value = (
            "Note: Sum of detailed market positions is standalone (pre-diversification). "
            "Marginal contribution reflects post-diversification effect in the firm-wide portfolio."
        )
        ws.cell(row=note_row, column=1).font = Font(italic=True, color="555555")
        ws.merge_cells(
            start_row=note_row,
            start_column=1,
            end_row=note_row,
            end_column=ws.max_column,
        )
        autofit_columns(ws)

    def _create_detailed_credit_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Detailed Credit Risk", 5)

        # Protect sheet for regulatory compliance
        ws.protection.sheet = True
        ws.protection.password = "ec_report"

        ws["A1"] = "Detailed Credit Risk Portfolio"
        ws["A1"].font = Font(size=16, bold=True, color=self.colors["header"])

        # Full portfolio table
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
                        cell.number_format = '"£"#,##0'

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
        autofit_columns(ws)

    def _create_detailed_oprisk_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Detailed Op Risk", 6)

        # Protect sheet for regulatory compliance
        ws.protection.sheet = True
        ws.protection.password = "ec_report"

        ws["A1"] = "Detailed Op Risk Scenarios"
        ws["A1"].font = Font(size=16, bold=True, color=self.colors["header"])

        raw_results = (
            self.op_details.get("stress_test_results", [])
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

        headers = [
            "Rank",
            "ScenarioID",
            "Description",
            "Baseline Cap",
            "Stressed Capital",
            "Uplift",
        ]
        for c, h in enumerate(headers, 1):
            cell = ws.cell(3, c, h)
            cell.fill = PatternFill("solid", self.colors["header"])
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")
            cell.number_format = '"£"#,##0'

        # Write data rows
        for i, r in enumerate(top10, 4):
            ws.cell(i, 1, int(i - 3))
            ws.cell(i, 2, str(getattr(r, "name", "N/A") or "N/A"))
            ws.cell(i, 3, str(getattr(r, "description", "---") or "---"))

            # Baseline capital
            base_cap = float(getattr(r, "capital_base", 0.0))
            ws.cell(i, 4, base_cap).number_format = '"£"#,##0'

            # Stressed capital
            stressed_cap = float(getattr(r, "capital_stressed", 0.0))
            ws.cell(i, 5, stressed_cap).number_format = '"£"#,##0'

            # Uplift as raw float
            uplift = float(getattr(r, "uplift_factor", 0.0))
            ws.cell(i, 6, uplift).number_format = "0.00x"

            if i - 3 <= 3:  # Top 3 highlighting
                for c in range(1, 7):
                    ws.cell(i, c).fill = PatternFill("solid", self.colors["gold"])

        # Table definition with strictly unique name and checked range
        last_row = len(top10) + 3
        if len(top10) > 0:
            tab = Table(
                displayName=f"OpRiskTop10_{self.timestamp}", ref=f"A3:F{last_row}"
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
        autofit_columns(ws)

    def _create_correlation_matrix_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Correlation Matrix", 7)

        # Protect sheet for regulatory compliance
        ws.protection.sheet = True
        ws.protection.password = "ec_report"

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
        autofit_columns(ws)

    def _create_sensitivity_analysis_sheet(self, wb: Workbook):
        ws = wb.create_sheet("Sensitivity Analysis", 8)

        # Protect sheet for regulatory compliance
        ws.protection.sheet = True
        ws.protection.password = "ec_report"

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
            _, _, EC_total, _, _ = aggregate_economic_capital(
                self.results["individual_risks"],
                confidence_level=level,
                copula_df=7.0,
            )

            ws.cell(i, 1, f"{level * 100:.1f}%")
            cell_b = ws.cell(i, 2, EC_total)
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
        chart.x_axis.title = "Confidence Level"
        chart.height = 12
        chart.width = 18

        # Data and Categories
        data = Reference(ws, min_col=2, min_row=4, max_row=last_row)
        cats = Reference(ws, min_col=1, min_row=4, max_row=last_row)
        chart.add_data(data, titles_from_data=False)
        chart.set_categories(cats)

        # X-axis Labels
        chart.x_axis.title = "Confidence Level"

        apply_clean_style(chart, "Economic Capital (£)")

        ws.add_chart(chart, "E2")
        autofit_columns(ws)


def generate_firmwide_ec_report(
    aggregated_results: Dict[str, Any],
    output_dir: str | Path = "reports",
) -> Path:
    """Convenience function to generate the firm-wide report."""
    return FirmWideECReporter(aggregated_results, output_dir).generate_report()
