"""Validate scalable assignments and prevent ambiguous/overlapping work."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from coordinator_topology import DEFAULT_TOPOLOGY, load_topology
from coordinate_verification_shard import run_shard


class TopologyTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads(DEFAULT_TOPOLOGY.read_text())

    def load(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'topology.json'
            path.write_text(json.dumps(self.config))
            return load_topology(path)

    def test_three_pods_and_model_names_are_configuration_driven(self):
        self.config['pods'].append({'id':'third','credential_file':'/private/third.json','adapter':'/local/third.py'})
        self.config['generation_workers'].append({'pod':'third','model':'other_model','python':'/venv/python','config':'/config/other.yaml'})
        self.assertEqual(len(self.load()['generation_workers']),3)

    def test_primary_need_not_be_first(self):
        self.config['pods'].reverse()
        self.assertEqual(self.load()['primary_pod'],'primary')

    def test_unknown_pod_rejected(self):
        self.config['generation_workers'][0]['pod']='missing'
        with self.assertRaises(ValueError):self.load()

    def test_duplicate_model_rejected(self):
        self.config['generation_workers'][1]['model']='7b'
        with self.assertRaises(ValueError):self.load()

    def test_overlapping_ranges_rejected(self):
        other=copy.deepcopy(self.config['verification_shards'][0]);other['id']='duplicate_range'
        self.config['verification_shards'].append(other)
        with self.assertRaises(ValueError):self.load()

    def test_disjoint_ranges_accepted(self):
        original=self.config['verification_shards'][0];original['stop']=4096
        other=copy.deepcopy(original);other.update(id='next_range',start=4096,stop=8192)
        self.config['verification_shards'].append(other)
        self.assertEqual(len(self.load()['verification_shards']),2)

    def test_multiple_chunks_use_configured_range_and_unique_transfer_paths(self):
        payload=b'archive';digest=hashlib.sha256(payload).hexdigest()
        class Source:
            index=0
            def execute(self,code):
                compile(code,'remote','exec')
                start=256+self.index*128;self.index+=1
                return {'alive':True,'ready':{'start':start,'stop':start+128,'archive':'/transfer.zip','sha256':digest}}
            def download(self,path,target,expected):
                if expected!=digest:raise AssertionError('Wrong digest')
                target.write_bytes(payload)
        class Primary:
            paths=[]
            def put(self,path,content):self.paths.append(path)
            def execute(self,code):
                compile(code,'remote','exec')
                return {'status':'IMPORTED'}
        shard={'id':'custom_shard','start':256,'stop':512,'chunk_size':128,'round':2,
               'data_root':'/work/data','state_file':'/work/state.json'}
        primary=Primary()
        with tempfile.TemporaryDirectory() as root:
            run_shard(Source(),primary,shard,root)
        self.assertEqual(len(primary.paths),2)
        self.assertTrue(all('verification_custom_shard_part_' in p for p in primary.paths))


if __name__=='__main__':
    unittest.main()
