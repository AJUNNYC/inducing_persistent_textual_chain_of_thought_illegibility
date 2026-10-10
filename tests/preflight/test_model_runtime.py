"""Real model_runtime acceptance; missing resources are explicitly skipped, never passed."""
from pathlib import Path
import os
import pytest
from shared_procedures.configuration import load_configuration,configuration_hash
from shared_procedures.preflight import load_preflight_evidence,require_preflights

def test_live_model_runtime_evidence():
    path=os.getenv('ILLEGIBILITY_PREFLIGHT_EVIDENCE')
    if not path:
        pytest.skip('Required live model_runtime evidence unavailable; affected launch blocked')
    config=load_configuration(Path(__file__).resolve().parents[2]/'config.yaml')
    report=load_preflight_evidence(path,config)
    require_preflights(report,['model_runtime'],expected_configuration_hash=configuration_hash(config))
