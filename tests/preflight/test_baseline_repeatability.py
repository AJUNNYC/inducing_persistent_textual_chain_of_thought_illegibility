"""Preflight consumes persisted live evidence, never generates a fake pass."""
import json
import os
import pytest


def test_live_baseline_repeatability_evidence():
    path=os.environ.get('BASELINE_REPEATABILITY_REPORT')
    if not path:
        pytest.skip('BLOCKED: live two-model seeds42/43/44 repeatability report unavailable')
    with open(path) as f: report=json.load(f)
    assert report['status']=='passed' and report['ready_for_training'] is True
    assert report['evidence_type']=='live_baseline_diagnostics'
    assert len(report['models'])==2
    assert all(sum(d['panel_count'] for d in m['domains'].values())==1233 for m in report['models'])
