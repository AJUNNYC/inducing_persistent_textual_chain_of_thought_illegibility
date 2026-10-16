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
    metadata['subject'] = 'Integrated procedure edition: ' + spec['edition']
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
    provenance['procedure_update_sha256'] = hashlib.sha256((ROOT/'spec/protocol_page_updates.json').read_bytes()).hexdigest()
    provenance['validated_live_scope'] = ['vLLM generation benchmark', 'whole-batch Drive integrity',
                                        'evaluation panel preparation', 'shared RL question selection',
                                        'joint accepted-pair validation and exact-deficit refills',
                                        'all forty SFT dataset conditions',
                                        'independent full Drive byte readback and native record validation',
                                        'historical 3072-token untouched 14B GPQA Diamond baseline',
                                        'ten completed 14B ROT13 fits: reports/manifests byte-verified and final payload metadata matched',
                                        'separate 14B ROT13 level-10 extra epoch: 625 additional updates and 26 Drive-verified artifacts',
                                        'new level-10 GPQA immutable protocol and observed inference; official results pending',
                                        'private completion-biased 42-request snapshot and bounded review; observational backup distinct from native completion',
                                        'same fixed42 exact identity/native decoding and provisional frozen-scorer replay; native official results pending']
    provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'Built {len(doc)}-page integrated protocol; refreshed extraction and provenance')
    doc.close()


if __name__ == '__main__':
    main()
