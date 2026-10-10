"""Export frozen analysis records and their actual curve coordinates."""
from __future__ import annotations
import json
from pathlib import Path
from shared_procedures.storage import atomic_write_json


def generate_report(analysis: dict, destination: str | Path) -> list[Path]:
    root=Path(destination)
    root.mkdir(parents=True,exist_ok=True)
    data=root/'results.json'
    atomic_write_json(data,analysis)
    document=root/'results.md'
    lines=['# First-stage results','',
           '**Simulation fixture; no empirical experiment was performed.**' if analysis.get('simulation') else 'All missing and indeterminate observations are preserved.',
           '',f"Construction success: `{analysis.get('construction_success')}`. Confirmatory authorization: `{analysis.get('confirmatory_authorized')}`.",
           '',str(analysis.get('claim_limits','')),'', '| State | Domain | Capability | Qualified opacity, full denominator |','|---|---|---:|---:|']
    def display(value): return 'indeterminate' if value is None else f'{value:.4f}'
    for state in analysis.get('state_summaries',[]):
        for domain,row in state.get('domains',{}).items():
            lines.append(f"| {state['state_id']} | {domain} | {display(row.get('accuracy'))} | {display(row.get('qualified_fraction_full'))} |")
    windows=analysis.get('criteria',{}).get('windows',[])
    if windows:
        lines.extend(['','| Window | Status |','|---|---|'])
        for row in windows:
            lines.append(f"| {row.get('window_id',row.get('id','unknown'))} | {row.get('status','unreported')} |")
    document.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=[data,document]
    curves=analysis.get('curves',[])
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for curve in curves:
            rows=curve['levels']
            if not rows: continue
            figure,axes=plt.subplots(1,3,figsize=(15,4))
            for domain in ('gsm8k','mmlu','arc','mbpp'):
                for axis,key in zip(axes,('accuracy','confirmed_qualified_fraction_full','primary_mean_observed')):
                    # NaN creates a visible gap for unknown evidence; never interpolate.
                    values=[r['domains'][domain].get(key) for r in rows]
                    if not any(v is not None for v in values): continue
                    axis.plot([r['prescribed_strength'] for r in rows],[float('nan') if v is None else v for v in values],marker='o',label=domain)
            for axis,label in zip(axes,('Capability accuracy','Qualified opacity, fixed denominator','Observed primary mean score')):
                axis.set_ylabel(label);axis.set_xlabel('Prescribed level / alpha / RL step')
                if axis.lines: axis.legend()
            title=f"{curve['branch']} {curve['model']}"
            if analysis.get('simulation'): title+=' — simulation fixture'
            figure.suptitle(title);figure.tight_layout()
            path=root/f"{curve['branch']}_{curve['model']}_curves.png"
            figure.savefig(path,dpi=180);plt.close(figure);files.append(path)
    return files
