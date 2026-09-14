"""Build the evidence-backed technical report from paper/manuscript.md."""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "manuscript.md"
OUTPUT = (
    ROOT
    / "paper"
    / "output"
    / "pdf"
    / "Do_Published_HumanEval_Rankings_Survive_Local_Deployment.pdf"
)

NAVY = colors.HexColor("#18324A")
BLUE = colors.HexColor("#2F6B91")
PALE = colors.HexColor("#EAF1F6")
INK = colors.HexColor("#17202A")
MUTED = colors.HexColor("#52616B")
RULE = colors.HexColor("#BCCAD4")


def inline_markup(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(
        r"(https://[^\s<]+)",
        lambda match: f'<link href="{match.group(1)}" color="#2F6B91">{match.group(1)}</link>',
        text,
    )
    return text


def parse_blocks(source: str) -> list[tuple[str, object]]:
    lines = source.splitlines()
    blocks: list[tuple[str, object]] = []
    paragraph: list[str] = []
    index = 0

    def flush() -> None:
        if paragraph:
            blocks.append(("paragraph", " ".join(part.strip() for part in paragraph)))
            paragraph.clear()

    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            flush()
            index += 1
            continue
        if line.startswith("|"):
            flush()
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            if len(rows) > 1 and all(re.fullmatch(r":?-+:?", cell) for cell in rows[1]):
                rows.pop(1)
            blocks.append(("table", rows))
            continue
        if line.startswith("- "):
            flush()
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append(lines[index][2:].strip())
                index += 1
            blocks.append(("bullets", items))
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush()
            blocks.append((f"heading{len(heading.group(1))}", heading.group(2)))
            index += 1
            continue
        paragraph.append(line[:-2] if line.endswith("  ") else line)
        index += 1
    flush()
    return blocks


class PaperDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str) -> None:
        super().__init__(
            filename,
            pagesize=letter,
            leftMargin=0.78 * inch,
            rightMargin=0.78 * inch,
            topMargin=0.72 * inch,
            bottomMargin=0.68 * inch,
            title="Do Published HumanEval Rankings Survive Local Deployment?",
            author="Jaiveer Bassi",
            subject="Five-model HumanEval ranking reproduction on a consumer GPU",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates(PageTemplate(id="paper", frames=[frame], onPage=self.draw_page))

    @staticmethod
    def draw_page(canvas, doc) -> None:
        canvas.saveState()
        width, height = letter
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(0.78 * inch, height - 0.48 * inch, width - 0.78 * inch, height - 0.48 * inch)
        canvas.setFont("Helvetica", 7.8)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.78 * inch, height - 0.39 * inch, "LOCAL DEPLOYMENT OF CODE MODELS")
        canvas.drawRightString(
            width - 0.78 * inch, height - 0.39 * inch, "TECHNICAL REPORT  |  2026"
        )
        canvas.line(0.78 * inch, 0.47 * inch, width - 0.78 * inch, 0.47 * inch)
        canvas.drawString(0.78 * inch, 0.31 * inch, "Bassi")
        canvas.drawRightString(width - 0.78 * inch, 0.31 * inch, f"{doc.page}")
        canvas.restoreState()


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PaperTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=25,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=7,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Heading2"],
            fontName="Helvetica",
            fontSize=13,
            leading=16,
            textColor=BLUE,
            alignment=TA_LEFT,
            spaceAfter=18,
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            spaceAfter=2,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.1,
            leading=12.4,
            textColor=INK,
            leftIndent=14,
            rightIndent=14,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=16,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.4,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.1,
            leading=12.2,
            textColor=INK,
            leftIndent=16,
            firstLineIndent=-9,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=MUTED,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10.8,
            textColor=INK,
            leftIndent=14,
            firstLineIndent=-14,
            spaceAfter=5,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=8.5,
            textColor=colors.white,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=8.8,
            textColor=INK,
        ),
        "table_code": ParagraphStyle(
            "TableCode",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=5.4,
            leading=7,
            textColor=INK,
        ),
    }


def result_chart() -> Drawing:
    labels = ["Qwen\n0.5B", "StarCoder2\n3B", "DeepSeek\n1.3B", "Qwen\n1.5B", "Qwen\n3B"]
    published = [28.0, 31.7, 34.8, 43.9, 52.4]
    local = [23.8, 1.8, 34.1, 39.0, 52.4]
    drawing = Drawing(455, 165)
    chart = VerticalBarChart()
    chart.x = 45
    chart.y = 30
    chart.height = 100
    chart.width = 385
    chart.data = [published, local]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.dy = -3
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 60
    chart.valueAxis.valueStep = 10
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.gridStrokeColor = colors.HexColor("#DDE5EA")
    chart.valueAxis.gridStrokeWidth = 0.4
    chart.bars[0].fillColor = colors.HexColor("#9DB5C5")
    chart.bars[1].fillColor = BLUE
    chart.bars.strokeColor = None
    chart.barSpacing = 1.5
    chart.groupSpacing = 6
    drawing.add(chart)
    drawing.add(
        String(
            45, 148, "HumanEval pass@1 (%)", fontName="Helvetica-Bold", fontSize=8.5, fillColor=NAVY
        )
    )
    drawing.add(String(300, 148, "Published", fontName="Helvetica", fontSize=7.5, fillColor=MUTED))
    drawing.add(String(370, 148, "Local", fontName="Helvetica", fontSize=7.5, fillColor=MUTED))
    drawing.add(Rect(285, 150, 7, 7, fillColor=colors.HexColor("#9DB5C5"), strokeColor=None))
    drawing.add(Rect(358, 150, 7, 7, fillColor=BLUE, strokeColor=None))
    return drawing


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    blocks = parse_blocks(SOURCE.read_text(encoding="utf-8"))
    style = styles()
    story = []
    title_seen = subtitle_seen = False
    in_references = False
    result_chart_added = False

    for kind, content in blocks:
        if kind == "heading1" and not title_seen:
            story.append(Spacer(1, 0.34 * inch))
            story.append(Paragraph(inline_markup(str(content)), style["title"]))
            title_seen = True
            continue
        if kind == "heading2" and not subtitle_seen:
            story.append(Paragraph(inline_markup(str(content)), style["subtitle"]))
            subtitle_seen = True
            continue
        if kind == "heading2":
            in_references = str(content) == "References"
            story.append(Paragraph(inline_markup(str(content)), style["h1"]))
            continue
        if kind == "heading3":
            story.append(Paragraph(inline_markup(str(content)), style["h2"]))
            continue
        if kind == "paragraph":
            text = str(content)
            if text.startswith("**Jaiveer Bassi**") or text.startswith("**Technical report"):
                story.append(Paragraph(inline_markup(text).replace("  ", "<br/>"), style["meta"]))
            elif text.startswith("**Keywords:**"):
                story.append(Paragraph(inline_markup(text), style["meta"]))
                story.append(
                    HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=8, spaceAfter=8)
                )
            elif text.startswith("**Table "):
                story.append(Paragraph(inline_markup(text), style["caption"]))
                if text.startswith("**Table 2") and not result_chart_added:
                    story.append(
                        KeepTogether(
                            [
                                result_chart(),
                                Paragraph(
                                    "<b>Figure 1.</b> Published and local HumanEval pass@1. "
                                    "The StarCoder2-3B divergence determines the "
                                    "rank-reproduction outcome.",
                                    style["caption"],
                                ),
                            ]
                        )
                    )
                    result_chart_added = True
            elif in_references and re.match(r"\[\d+\]", text):
                story.append(Paragraph(inline_markup(text), style["reference"]))
            elif story and any(
                isinstance(item, Paragraph)
                and item.style.name == "H1"
                and item.getPlainText() == "Abstract"
                for item in story[-2:]
            ):
                story.append(Paragraph(inline_markup(text), style["abstract"]))
            else:
                story.append(Paragraph(inline_markup(text), style["body"]))
            continue
        if kind == "bullets":
            for item in content:
                story.append(Paragraph("- " + inline_markup(str(item)), style["bullet"]))
            continue
        if kind == "table":
            rows = content
            rendered = []
            for row_index, row in enumerate(rows):
                rendered_row = []
                for column_index, cell in enumerate(row):
                    cell_style = style["table_header"] if row_index == 0 else style["table_cell"]
                    if len(row) == 3 and row_index > 0 and column_index == 2:
                        cell_style = style["table_code"]
                    rendered_row.append(Paragraph(inline_markup(cell), cell_style))
                rendered.append(rendered_row)
            if len(rows[0]) == 3:
                widths = [1.35 * inch, 2.35 * inch, 2.55 * inch]
            else:
                widths = [1.75 * inch, 1.12 * inch, 1.35 * inch, 1.12 * inch, 1.35 * inch]
            table = Table(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                        ("LEADING", (0, 0), (-1, -1), 9),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(table)

    doc = PaperDocTemplate(str(OUTPUT))

    def invariant_canvas(*args, **kwargs):
        kwargs["invariant"] = 1
        return canvas.Canvas(*args, **kwargs)

    doc.build(story, canvasmaker=invariant_canvas)
    print(OUTPUT)


if __name__ == "__main__":
    build()
