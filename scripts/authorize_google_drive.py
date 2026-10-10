#!/usr/bin/env python3
"""Local desktop browser consent; prints no client secrets or token values.

Install google-auth-oauthlib,google-api-python-client and requests on this
computer; no model/CUDA environment is needed. Use the existing desktop client.
"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from operations.drive import authorize,drive_service,DriveStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client',required=True,help='Protected existing desktop-client JSON path')
    parser.add_argument('--token',required=True,help='Protected refresh-token JSON destination')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    authorize(args.client,args.token,args.port)
    store=DriveStore(drive_service(args.token))
    folder=store.create_project_folder()
    print('OAuth consent and restricted account/folder verification completed.')
    print('Protected token file:',Path(args.token).resolve())
    print('Canonical project folder ID:',folder)
    return 0

if __name__=='__main__':raise SystemExit(main())
