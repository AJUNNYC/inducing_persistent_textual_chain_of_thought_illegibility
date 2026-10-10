#!/usr/bin/env python3
"""Resolve code/lock and actual auxiliary identities before any scientific output."""
from pathlib import Path
import argparse
import subprocess
import os
import sys
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from shared_procedures.configuration import load_configuration,configuration_hash,validate_configuration
from shared_procedures.identity import content_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config.yaml')
    parser.add_argument('--output',required=True)
    parser.add_argument('--resolve-native-assets',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    commit=os.environ.get('SOURCE_CODE_COMMIT')
    if (root/'.git').exists():
        dirty=subprocess.run(['git','status','--porcelain','--untracked-files=normal'],cwd=root,capture_output=True,text=True,check=True)
        if dirty.stdout.strip():raise RuntimeError('Commit the intended source/documentation before freezing a production runtime')
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        if commit and commit!=actual:raise RuntimeError('Build/source commit mismatch')
        commit=actual
    if not commit or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise RuntimeError('Immutable committed source identity is required')
    config=load_configuration(args.config)
    config['runtime']['implementation_revision']=commit
    config['runtime']['dependency_lock_hash']=content_hash((root/'requirements.lock').read_bytes())
    if args.resolve_native_assets:
        from data_generation_collection.sft_data.kazitext.kazitext_data_construction import provision_kazitext_assets
        cache=Path(config['runtime']['cache_root'])
        assets=provision_kazitext_assets(cache/'kazitext_auxiliary',source_dir=cache/'kazitext')
        config['operations'].update(udpipe_path=assets['udpipe_path'],aspell_dictionary_files=assets['aspell_dictionary_files'])
        config['interventions']['kazitext']['aspell_dictionary_provenance']={'files':assets['aspell']['identities'],'settings':assets['aspell']['settings']}
        if config['interventions']['kazitext']['udpipe_sha256']!=assets['udpipe']['sha256']:
            raise RuntimeError('Resolved UDPipe differs from the controlling identity')
    validate_configuration(config)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise RuntimeError('Refusing to overwrite an existing runtime configuration')
    output.write_text(yaml.safe_dump(config,sort_keys=False))
    print('Runtime configuration:',output.resolve())
    print('Scientific configuration SHA256:',configuration_hash(config))
    print('Spending authorization and real live preflight evidence remain separate requirements.')
    return 0

if __name__=='__main__':raise SystemExit(main())
