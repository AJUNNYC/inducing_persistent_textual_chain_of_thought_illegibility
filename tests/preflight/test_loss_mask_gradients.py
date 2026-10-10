"""Real loss_mask_gradients acceptance; missing resources are explicitly skipped, never passed."""
from pathlib import Path
import os
import pytest
from shared_procedures.configuration import load_configuration,configuration_hash
from shared_procedures.preflight import load_preflight_evidence,require_preflights

def test_live_loss_mask_gradients_evidence():
    path=os.getenv('ILLEGIBILITY_PREFLIGHT_EVIDENCE')
    if not path:
        pytest.skip('Required live loss_mask_gradients evidence unavailable; affected launch blocked')
    config=load_configuration(Path(__file__).resolve().parents[2]/'config.yaml')
    report=load_preflight_evidence(path,config)
    require_preflights(report,['loss_mask_gradients'],expected_configuration_hash=configuration_hash(config))
