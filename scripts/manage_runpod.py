#!/usr/bin/env python3
"""Run the external billing controller, inspect, stop or explicitly authorized resume."""
import argparse
import json
from operations.runpod import RunPodClient
from shared_procedures.storage import read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('watch','inspect','stop','resume'))
    parser.add_argument('--pod-id',required=True)
    parser.add_argument('--authorization')
    parser.add_argument('--state',default='artifacts/state/runpod_lifecycle.json')
    args=parser.parse_args();client=RunPodClient()
    if args.action=='inspect': result=client.inspect_pod(args.pod_id)
    elif args.action=='stop': result=client.stop_pod(args.pod_id)
    else:
        if not args.authorization: parser.error('--authorization is required')
        authorization=read_json(args.authorization)
        result=client.guarded_lifecycle(args.pod_id,authorization,args.state) if args.action=='watch' else client.resume_pod(args.pod_id,authorization)
    print(json.dumps(result,indent=2))
    return 2 if isinstance(result,dict) and result.get('status')=='STOP_UNCONFIRMED' else 0

if __name__=='__main__': raise SystemExit(main())
