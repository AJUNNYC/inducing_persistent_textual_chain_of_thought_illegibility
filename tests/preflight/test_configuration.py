from pathlib import Path
from shared_procedures.configuration import load_configuration,sft_matrix

def test_fixed_configuration():
    config=load_configuration(Path(__file__).resolve().parents[2]/'config.yaml')
    assert len(sft_matrix(config))==40 and sum(config['evaluation']['capability_counts'].values())==1233
