#!/usr/bin/env python3
"""Build the dated procedure amendment and append it to the first 81 PDF pages.

Documentation-only tool; requires ReportLab and pypdf. Repeated runs replace
the generated amendment while preserving the existing protocol page order.
"""
from pathlib import Path
import re
import tempfile
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]


def inline(value):
    value = escape(value)
    def link(match):
        target = match.group(2)
        if not target.startswith('https://'):
            path = (ROOT / 'docs' / target).resolve().relative_to(ROOT)
            target = 'https://github.com/AJUNNYC/inducing_persistent_textual_chain_of_thought_illegibility/blob/main/' + path.as_posix()
        return f'<link href="{target}" color="#235a90">{match.group(1)}</link>'
    value = re.sub(r'\[([^\]]+)\]\([^)]+\)', link, value)
    value = re.sub(r'`([^`]+)`', r'<font face="Courier">\1</font>', value)
    return re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', value)


def main():
    source = (ROOT / 'docs/procedure_update.md').read_text()
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 13
    styles['BodyText'].wordWrap = 'CJK'  # Wrap long immutable hashes safely.
    story = [Paragraph('H.11 Approved procedure update', styles['Title'])]
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith('|'):
            rows = []
            while index < len(lines) and lines[index].startswith('|'):
                row = lines[index]
                if not re.fullmatch(r'[| :\-]+', row):
                    rows.append([Paragraph(inline(c.strip()), styles['BodyText'])
                                 for c in row.strip('|').split('|')])
                index += 1
            table = Table(rows, colWidths=[1.9 * inch, 4.5 * inch], repeatRows=1)
            table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5eef9')),
                ('GRID', (0, 0), (-1, -1), .3, colors.lightgrey),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.extend([table, Spacer(1, 10)])
            continue
        style = styles['Heading2'] if line.startswith('## ') else styles['Heading1'] if line.startswith('# ') else styles['BodyText']
        story.extend([Paragraph(inline(line.lstrip('# ')), style), Spacer(1, 8)])
        index += 1

    def footer(canvas, document):
        canvas.setFont('Helvetica', 8)
        canvas.drawString(40, 25, 'Procedure amendment | 11 October 2026 | H.11')
        canvas.drawRightString(570, 25, str(document.page))

    amendment = ROOT / 'spec/procedure_update.pdf'
    SimpleDocTemplate(str(amendment), title='H.11 Approved procedure update',
                      author='Research implementation documentation',
                      leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40).build(
                          story, onFirstPage=footer, onLaterPages=footer)
    active = ROOT / 'spec/experiment.pdf'
    reader = PdfReader(active)
    if len(reader.pages) < 81:
        raise ValueError('Expected the complete 81-page protocol before H.11')
    if len(reader.pages) > 81 and 'H.11 Approved procedure update' not in reader.pages[81].extract_text():
        raise ValueError('Existing later pages are not this generated amendment')
    writer = PdfWriter()
    for page in reader.pages[:81]:
        writer.add_page(page)
    for page in PdfReader(amendment).pages:
        writer.add_page(page)
    if reader.metadata:
        writer.add_metadata({k: str(v) for k, v in reader.metadata.items() if v is not None})
    with tempfile.NamedTemporaryFile(dir=active.parent, suffix='.pdf', delete=False) as output:
        temporary = Path(output.name)
        writer.write(output)
    temporary.replace(active)
    print(f'Built {len(writer.pages)}-page active protocol and standalone H.11 amendment')


if __name__ == '__main__':
    main()
