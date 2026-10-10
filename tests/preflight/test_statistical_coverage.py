"""Only the full, prospectively frozen Monte Carlo evidence clears this gate."""
import json
import os
import pytest
from data_generation_collection.evaluation_data.protocol.analysis_plan.statistical_validation import implementation_hash, SCENARIOS
from shared_procedures.identity import content_hash


def test_full_statistical_coverage_evidence():
    path=os.environ.get('STATISTICAL_COVERAGE_REPORT')
    if not path:
        pytest.skip('BLOCKED: full seven-scenario100-repetition10000-draw validation report unavailable')
    with open(path) as f: report=json.load(f)
    digest=report.pop('report_hash')
    assert content_hash(report)==digest
    assert report['status']=='passed' and report['full_validation'] is True
    assert report['draws']==10000 and report['repetitions']==100
    assert report['implementation_hash']==implementation_hash()
    assert [s['scenario'] for s in report['scenarios']]==list(SCENARIOS)
    assert all(s['joint_coverage']['one_sided_wilson_lower']>=.95 and s['diagnostics']['structural_failures']==0 for s in report['scenarios'])
