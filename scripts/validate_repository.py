#!/usr/bin/env python3
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from operations.validation import validate_repository
def main():
    result=validate_repository(Path(__file__).resolve().parents[1])
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='VERIFIED' else 1

if __name__=='__main__':
    raise SystemExit(main())
