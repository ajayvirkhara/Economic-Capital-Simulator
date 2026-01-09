"""
Shared reporting utilities used across all risk reporting modules.
"""

from openpyxl.chart.layout import Layout, ManualLayout
from openpyxl.chart.label import DataLabelList
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.chart.series import DataPoint


def apply_clean_style(chart: BarChart, y_title: str, num_points: int = 11) -> None:
    """Apply consistent clean styling to bar charts across all reports."""
    chart.y_axis.title = y_title
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.y_axis.number_format = '"£"#,##0,,"M"'
    chart.legend = None  # Remove legend
    chart.varyColors = True  # Different colors for bars

    # Clean layout
    chart.layout = Layout(manualLayout=ManualLayout(x=0.1, y=0.05, w=0.85, h=0.70))

    # Clean gridlines
    chart.y_axis.majorGridlines = None
    chart.y_axis.majorTickMark = "out"
    chart.x_axis.tickLblPos = "low"

    # Rotate bar labels to prevent collision
    chart.x_axis.tickLblPos = "low"
    chart.x_axis.textRotation = -45

    # Color palette
    palette = [
        "F06292",  # Soft Pink
        "E67E22",  # Orange
        "BCAE3E",  # Olive Gold
        "7CB342",  # Leaf Green
        "27AE60",  # Emerald
        "26A69A",  # Teal
        "2980B9",  # Ocean Blue
        "5D9CEC",  # Sky Blue
        "BA68C8",  # Amethyst
        "9575CD",  # Deep Purple
    ]

    if chart.series:
        s = chart.series[0]

        for i in range(num_points):
            pt = DataPoint(idx=i)

            # Select color from palette (loops back to start if i > palette length)
            color_hex = palette[i % len(palette)]

            # Disable white-fill behaviour for negative bars
            pt.invertIfNegative = False

            # Apply solid fill and remove default black outline
            pt.graphicalProperties.solidFill = color_hex
            pt.graphicalProperties.ln = None

            s.dPt.append(pt)

        # Do not override any existing formatting
        if s.dLbls is None:
            s.dLbls = DataLabelList()

            # Data labels formatting
            s.dLbls.showSourceLinked = False  # Severs the link for the whole series
            s.dLbls.showVal = True
            s.dLbls.showCatName = False
            s.dLbls.showLegendKey = False
            s.dLbls.showSerName = False
            s.dLbls.numFmt = '"£"#,##0,,"M"'
            s.dLbls.position = "outEnd"
            s.dLbls.showLeaderLines = True


def autofit_columns(ws: Worksheet) -> None:
    """Automatically adjust column widths in an openpyxl worksheet for better readability."""
    for column in ws.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)
        for cell in column:
            cell.alignment = Alignment(wrapText=True, vertical="top")
            try:
                # Check if cell has value and update max_length
                val_str = str(cell.value) if cell.value is not None else ""
                if len(val_str) > max_length:
                    max_length = len(val_str)
            except Exception:
                pass

        # Limit max column width to 50 to prevent Excel rendering errors
        adjusted_width = min(max_length + 2, 50)
        ws.column_dimensions[column_letter].width = adjusted_width
