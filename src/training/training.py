"""Shared first-stage delegation, with independently selectable training branches."""
from pathlib import Path
from training.sft.sft_training import sft_training
from training.rl_grpo.rl_training import run_rl_training


def train(*, config, sft_datasets=None, rl_manifest=None, output_root, judge=None,
          preflight_report, runtime_identity, families=('rot13','kazitext'),
          include_rl=True, rl_model_sizes=('7b','14b')):
    result = {}
    if families:
        if sft_datasets is None:
            raise ValueError('Selected SFT branches require verified condition datasets')
        result['sft'] = sft_training(config,sft_datasets,Path(output_root)/'sft',
                                   families=families,preflight_report=preflight_report)
    if include_rl:
        if rl_manifest is None or judge is None:
            raise ValueError('Selected RL trajectories require frozen questions and the reward client')
        result['rl'] = run_rl_training(rl_manifest,Path(output_root)/'rl',judge=judge,
            model_revisions={v['id']:v['revision'] for v in config['models'].values()},
            tokenizer_revisions={v['id']:v['tokenizer_revision'] for v in config['models'].values()},
            model_sizes=rl_model_sizes,runtime_identity=runtime_identity,preflight_evidence=preflight_report)
    if not result:
        raise ValueError('At least one training branch must be selected')
    return result
