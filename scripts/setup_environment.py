#!/usr/bin/env python3
"""One repeatable entrypoint: install a locked CPU or CUDA environment."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('cpu','gpu','gpu-published'), default='cpu')
    parser.add_argument('--venv', default='.venv')
    parser.add_argument('--provision', action='store_true')
    parser.add_argument('--drive', action='store_true')
    parser.add_argument('--skip-tests', action='store_true', help='Skip development tests; live scientific gates remain mandatory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environment = Path(args.venv).resolve()
    if not (environment / 'bin/python').exists():
        subprocess.run([sys.executable,'-m','venv',str(environment)],check=True)
    python = str(environment / 'bin/python')
    lock = root / ('requirements_cpu.lock' if args.profile=='cpu' else 'requirements.lock')
    subprocess.run([python,'-m','pip','install','--require-hashes','-r',str(lock)],check=True,cwd=root)
    if args.profile == 'gpu-published':
        # pip cannot hash-lock VCS dependencies. Preserve the entire existing
        # index lock and install only the supplemental immutable commit, without
        # resolving dependencies or installing the incompatible upstream extra.
        subprocess.run([python,'-m','pip','install','--no-deps','--no-build-isolation',
                        '-r',str(root / 'requirements_sft_published.lock')],check=True,cwd=root)
    subprocess.run([python,'-m','pip','install','--no-deps','--no-build-isolation','-e',str(root)],check=True,cwd=root)
    subprocess.run([python,'scripts/validate_repository.py'],check=True,cwd=root)
    subprocess.run([python,'-m','experiment_overview.experiment_overview','hardware'],check=True,cwd=root)
    if not args.skip_tests:
        subprocess.run([python,'-m','pytest','tests/unit','tests/integration','-q'],check=True,cwd=root)
    if args.provision:
        subprocess.run([python,'-m','experiment_overview.experiment_overview','provision'],check=True,cwd=root)
    if args.drive:
        subprocess.run([python,'-m','experiment_overview.experiment_overview','drive-init'],check=True,cwd=root)
    print('Installation and requested setup operations completed; scientific launch gates are checked separately.')

if __name__=='__main__': main()
