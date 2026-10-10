"""Print and persist actual readiness; a blocked mandatory gate exits nonzero."""
from __future__ import annotations
import argparse
import json
from shared_procedures.configuration import load_configuration
from shared_procedures.preflight import run_preflights

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config.yaml')
    parser.add_argument('--output-dir',default='artifacts/preflight/readiness')
    parser.add_argument('--execute-live',action='store_true',help='Execute available authorized real model/API/storage checks')
    parser.add_argument('--scope',choices=('generation','training'),default='training')
    args=parser.parse_args()
    report=run_preflights(load_configuration(args.config),output_dir=args.output_dir,execute_live=args.execute_live,scope=args.scope)
    print(json.dumps(report,indent=2))
    return 0 if report['ready'] else 2
if __name__=='__main__':raise SystemExit(main())
