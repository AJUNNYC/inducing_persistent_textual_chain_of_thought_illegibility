"""Reproduce failed frozen repetitions without rerunning or replacing validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shared_procedures.identity import content_hash
from shared_procedures.storage import atomic_write_json
from data_generation_collection.evaluation_data.protocol.analysis_plan import statistical_validation as validation
from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import create_analysis_plan
from data_generation_collection.evaluation_data.protocol.analysis_plan.simultaneous_bounds.simultaneous_bounds import compute_simultaneous_bounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    frozen = json.loads(Path(args.plan).read_text())
    report = json.loads(Path(args.report).read_text())
    if frozen['simulation_plan_hash'] != content_hash({k:v for k,v in frozen.items() if k != 'simulation_plan_hash'}):
        raise ValueError('Simulation plan integrity mismatch')
    if report['report_hash'] != content_hash({k:v for k,v in report.items() if k != 'report_hash'}):
        raise ValueError('Validation report integrity mismatch')
    if frozen['simulation_plan_hash'] != report['simulation_plan_hash']:
        raise ValueError('Report and simulation plan differ')
    if frozen['implementation_hash'] != report['implementation_hash'] or frozen['implementation_hash'] != validation.implementation_hash():
        raise ValueError('Current analysis implementation differs from frozen validation')
    diagnostics = []
    for scenario in report['scenarios']:
        failures = [run for run in scenario['runs'] if not run['joint_covered']]
        if not failures:
            continue
        population = validation.generate_finite_population(frozen, scenario['scenario'])
        truth = validation.population_truth(population)
        for run in failures:
            expected_seed = validation.derive_seed(frozen['seed'], frozen['namespace'], scenario['scenario'], run['repetition'], 'sample')
            if run['seed'] != expected_seed:
                raise ValueError('Repetition seed differs from frozen sequence')
            panel, leg, records = validation.sample_population(population, run['seed'])
            plan = create_analysis_plan(panel, leg)
            bounds = compute_simultaneous_bounds(records, plan, draws=frozen['draws'], retain_replicates=True)
            rows = []
            violations = []
            for j, gate in enumerate(bounds['gates']):
                if np.isfinite(truth[j]) and gate['lower_bound'] > truth[j] + 1e-12:
                    violation = {'gate_id':gate['gate_id'], 'lower_bound':gate['lower_bound'], 'population_truth':float(truth[j])}
                    violations.append(violation)
                    rows.append({**gate, 'population_truth':float(truth[j]),
                                 'observed_denominator':bounds['observed_denominators'][j],
                                 'bootstrap_quantiles':np.quantile(bounds['replicates'][:,j], [0,.01,.05,.5,.95,.99,1]).tolist()})
            if violations != run['violations'] or bounds['critical_value'] != run['critical_value']:
                raise ValueError('Frozen failure did not reproduce exactly')
            diagnostics.append({'scenario':scenario['scenario'], 'repetition':run['repetition'],
                                'critical_value':bounds['critical_value'], 'violations':rows})
            print(json.dumps({'scenario':scenario['scenario'], 'repetition':run['repetition'], 'reproduced_violations':len(rows)}), flush=True)
    atomic_write_json(args.output, {'status':'REPRODUCED', 'source_report_hash':report['report_hash'],
                                   'implementation_hash':validation.implementation_hash(),
                                   'validation_status':report['status'], 'diagnostics':diagnostics,
                                   'scope':'Failure diagnostics only; does not change or certify coverage.'})


if __name__ == '__main__':
    main()
