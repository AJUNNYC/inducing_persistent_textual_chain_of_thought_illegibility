#!/usr/bin/env python3
"""Integrate editable procedure text into existing PDF sections, without appendices.

Requires PyMuPDF. Page positions are fixed to preserve protocol page citations.
The original historical PDF is never modified. Repeated builds are idempotent
at the content level; the page-indexed Markdown and provenance are refreshed.
"""
import hashlib
import json
from pathlib import Path
import tempfile

import fitz

ROOT = Path(__file__).resolve().parents[1]


def replace(page, rect, text, *, maximum=10):
    rect = fitz.Rect(rect)
    # Check layout on a disposable page before removing any existing content.
    fontsize = maximum
    while fontsize >= 8:
        probe = fitz.open()
        test = probe.new_page(width=page.rect.width, height=page.rect.height)
        remaining = test.insert_textbox(rect, text, fontsize=fontsize,
                                       fontname='helv', lineheight=1.12)
        probe.close()
        if remaining >= 0:
            break
        fontsize -= .25
    else:
        raise ValueError('Updated section does not fit; adjust its source text')
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions(images=0, graphics=0)
    remaining = page.insert_textbox(rect, text, fontsize=fontsize,
                                   fontname='helv', lineheight=1.12)
    if remaining < 0:
        raise ValueError('Replacement layout differed from the checked layout')
    return fontsize


def main():
    spec = json.loads((ROOT/'spec/protocol_page_updates.json').read_text())
    active = ROOT/'spec/experiment.pdf'
    doc = fitz.open(active)
    if len(doc) < spec['page_count']:
        raise ValueError('The complete protocol must exist before editing sections')
    if len(doc) > spec['page_count']:
        doc.delete_pages(spec['page_count'], len(doc)-1)
    for patch in spec['patches']:
        size = replace(doc[patch['page']-1], patch['rect'], patch['text'])
        print(f"Page {patch['page']}: {patch['subject']} ({size:g} pt)")
    # Appendix H is part of the existing protocol; its header now names its role.
    for page in doc:
        for rect in page.search_for('CoT Illegibility — Implementation Addendum'):
            replace(page, [rect.x0, rect.y0, rect.x1+15, rect.y1+3],
                    'CoT Illegibility - Implementation decisions', maximum=9)
    metadata = doc.metadata
    metadata['subject'] = 'Integrated procedure edition: 11 October 2026'
    doc.set_metadata(metadata)
    with tempfile.NamedTemporaryFile(dir=active.parent, suffix='.pdf', delete=False) as output:
        temporary = Path(output.name)
    doc.save(temporary, garbage=4, deflate=True)
    doc.close()
    temporary.replace(active)
    doc = fitz.open(active)
    lines = ['# protocol_reference.md', '',
             'Complete page-indexed extraction of the active 81-page protocol. '
             'The approved procedure changes are integrated into their relevant PDF sections. '
             'Extraction may distort equations and tables; inspect the linked PDF page for exact layout. '
             'The historical source remains in `spec/archive/experiment_original.pdf`.', '']
    for index, page in enumerate(doc):
        lines.extend([f'## page_{index+1:03d}',
                      f'[View PDF page {index+1}](../spec/experiment.pdf#page={index+1})', '',
                      '```text', page.get_text().rstrip(), '```', ''])
    (ROOT/'docs/protocol_reference.md').write_text('\n'.join(lines))
    provenance_path = ROOT/'spec/source_provenance.json'
    provenance = json.loads(provenance_path.read_text())
    provenance['pdf'].update(pages=len(doc), sha256=hashlib.sha256(active.read_bytes()).hexdigest())
    provenance['active_procedure_edition'] = spec['edition']
    provenance['source_policy'] = ('Historical scientific source and approved user procedure decisions; '
                                   'active changes integrated directly into existing sections.')
    provenance['procedure_update_source'] = 'spec/protocol_page_updates.json'
    provenance['validated_live_scope'] = ['vLLM generation benchmark', 'whole-batch Drive integrity',
                                        'evaluation panel preparation', 'shared RL question selection']
    provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'Built {len(doc)}-page integrated protocol; refreshed extraction and provenance')
    doc.close()


if __name__ == '__main__':
    main()
