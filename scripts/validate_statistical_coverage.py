"""Run the full frozen statistical validation against existing evaluation IDs."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shared_procedures.storage import atomic_write_json
from data_generation_collection.evaluation_data.protocol.analysis_plan.prospective_analysis_plan import create_analysis_plan
from data_generation_collection.evaluation_data.protocol.analysis_plan.statistical_validation import run_statistical_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capability-panel', required=True)
    parser.add_argument('--legibility-subset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    plan = create_analysis_plan(json.loads(Path(args.capability_panel).read_text()),
                                json.loads(Path(args.legibility_subset).read_text()))
    root = Path(args.output_dir)
    atomic_write_json(root/'analysis_plan.json', plan)
    report = run_statistical_validation(plan, output_path=root/'report.json',
                                       workers=args.workers, resume=args.resume)
    print(json.dumps({'status':report['status'], 'ready_for_confirmatory_launch':report['ready_for_confirmatory_launch'],
                      'report_hash':report['report_hash']}), flush=True)
    return 0 if report['ready_for_confirmatory_launch'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
