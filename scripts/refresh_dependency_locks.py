#!/usr/bin/env python3
"""Resolve pinned manifests with hashes; handle official legacy Aspell archive."""
import subprocess
import tempfile
from pathlib import Path
import tomllib

ROOT=Path(__file__).resolve().parents[1]
ASPELL_HASH='2042910e6418e5f387f5b424d1d914032ed4ce904ea195b870db55bcb31cb38d'

def main():
    source=(ROOT/'pyproject.toml').read_text()
    project=tomllib.loads(source)['project']
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'pyproject.toml'
        path.write_text(source.replace(', "aspell-python-py3==1.15"',''))
        subprocess.run(['uv','pip','compile',str(path),'--extra','gpu','--extra','test','--extra','kazitext','--generate-hashes','--no-header','--output-file',str(ROOT/'requirements.lock'),'--python-version','3.12'],check=True)
        with (ROOT/'requirements.lock').open('a') as lock:
            lock.write('\n# Official verified legacy source archive; installed using pip.\naspell-python-py3==1.15 \\\n    --hash=sha256:'+ASPELL_HASH+'\n')
        cpu=Path(temporary)/'cpu.in'
        deps=project['dependencies']+project['optional-dependencies']['gpu']+project['optional-dependencies']['test']
        cpu.write_text('\n'.join(d for d in deps if not d.startswith(('bitsandbytes','trl')))+'\n')
        subprocess.run(['uv','pip','compile',str(cpu),'--torch-backend','cpu','--emit-index-url','--generate-hashes','--no-header','--output-file',str(ROOT/'requirements_cpu.lock'),'--python-version','3.12'],check=True)
        lock=ROOT/'requirements_cpu.lock'
        lock.write_text(lock.read_text().replace('--index-url https://pypi.org/simple','--index-url https://pypi.org/simple\n--extra-index-url https://download.pytorch.org/whl/cpu',1))

if __name__=='__main__': main()
