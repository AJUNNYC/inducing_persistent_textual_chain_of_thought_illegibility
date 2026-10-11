"""Approved PDF editions remain checked; altered sources and archives fail."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from operations.validation import validate_protocol_sources


class ProtocolProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name in ('spec/experiment.pdf','spec/archive/experiment_original.pdf',
                     'spec/protocol_page_updates.json','spec/source_provenance.json'):
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)

    def test_recorded_integrated_edition_passes(self):
        self.assertEqual(validate_protocol_sources(self.root),[])

    def test_unrecorded_active_pdf_change_fails(self):
        with (self.root/'spec/experiment.pdf').open('ab') as f:f.write(b'changed')
        self.assertIn('Active PDF differs from recorded provenance',validate_protocol_sources(self.root))

    def test_unrecorded_editable_source_change_fails(self):
        with (self.root/'spec/protocol_page_updates.json').open('ab') as f:f.write(b' ')
        self.assertIn('Integrated PDF source differs from recorded provenance',validate_protocol_sources(self.root))

    def test_archive_change_fails_even_if_declared_hash_is_updated(self):
        archive=self.root/'spec/archive/experiment_original.pdf'
        with archive.open('ab') as f:f.write(b'changed')
        p=self.root/'spec/source_provenance.json';data=json.loads(p.read_text())
        data['original_pdf']['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
        p.write_text(json.dumps(data))
        self.assertIn('Historical PDF changed: spec/archive/experiment_original.pdf',validate_protocol_sources(self.root))


if __name__=='__main__':
    unittest.main()
