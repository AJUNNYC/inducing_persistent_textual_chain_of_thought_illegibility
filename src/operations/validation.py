"""Repeatable static inventory, science traceability, and source integrity checks."""
from __future__ import annotations
import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

EXCLUDED={'.git','.venv','.pytest_cache','__pycache__','node_modules','.cache','build','dist'}
ARCHIVED_PDF_SHA256='db5e7dcd68b7632f2c222468a02affabe8526bedc48ef2b7139c31077343a704'


def validate_protocol_sources(root: Path) -> list[str]:
    """Bind the editable active edition to its source; retain the archive pin."""
    errors=[]
    archive=root/'spec/archive/experiment_original.pdf'
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVED_PDF_SHA256:
        errors.append('Historical PDF changed: spec/archive/experiment_original.pdf')
    provenance=json.loads((root/'spec/source_provenance.json').read_text())
    active=root/'spec/experiment.pdf'
    if provenance['pdf']['path']!='spec/experiment.pdf' or not active.is_file() or hashlib.sha256(active.read_bytes()).hexdigest()!=provenance['pdf']['sha256']:
        errors.append('Active PDF differs from recorded provenance')
    source=root/'spec/protocol_page_updates.json'
    if (provenance.get('procedure_update_source')!='spec/protocol_page_updates.json'
            or not source.is_file()
            or hashlib.sha256(source.read_bytes()).hexdigest()!=provenance.get('procedure_update_sha256')):
        errors.append('Integrated PDF source differs from recorded provenance')
    return errors


def tracked_inventory(root: Path) -> list[str]:
    if (root/'.git').exists():
        process=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root,capture_output=True,check=True)
        return sorted({p for p in process.stdout.decode().split('\0') if p and (root/p).is_file()})
    # Published source images intentionally omit Git internals. Inspect the
    # shipped files instead, retaining static runtime markers only.
    paths=[]
    for directory,children,files in os.walk(root):
        children[:]=[p for p in children if p not in EXCLUDED and not p.endswith('.egg-info')]
        relative=Path(directory).relative_to(root)
        for name in files:
            path=relative/name
            if path.parts[0] in ('data','artifacts') and name!='.gitkeep': continue
            if name.startswith('.env') and name!='.env.example': continue
            if path.as_posix()=='tools/mermaid/package-lock.json': continue
            paths.append(path.as_posix())
    return sorted(paths)


def validate_repository(root: str | Path) -> dict:
    root=Path(root)
    manifest=json.loads((root/'spec/repository_manifest.json').read_text())
    errors=[]
    expected={e['path']:e for e in manifest['entries']}
    # Ignore runtime data/artifacts, but retain tracked markers and inventories.
    files=tracked_inventory(root)
    for path in files:
        # Preserve the already frozen generation dependency filename.
        if path!='requirements-vllm.txt' and any(not re.fullmatch(r'[a-z0-9_.]+',part) for part in Path(path).parts): errors.append(f'Noncanonical tracked filename: {path}')
        if path not in expected: errors.append(f'Tracked path absent from canonical manifest: {path}')
    for path,e in expected.items():
        target=root/path
        if not target.exists() or (e['kind']=='directory')!=target.is_dir(): errors.append(f'Manifest path/type missing: {path}')
        diagram=root/e['diagram']
        if not diagram.is_file() or e['node'] not in diagram.read_text() or e['display_path'] not in diagram.read_text(): errors.append(f'Diagram mapping missing: {path}')
    for module in manifest['modules']:
        for key in ('code','doc','diagram','test'):
            if not (root/module[key]).is_file(): errors.append(f'Module path missing: {module[key]}')
        source=(root/module['code']).read_text()
        if not source.strip(): errors.append(f'Empty executable module: {module["code"]}')
        elif not any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for node in ast.parse(source).body): errors.append(f'Executable module has no implementation: {module["code"]}')
        testsource=(root/module['test']).read_text()
        if not testsource.strip(): errors.append(f'Empty test module: {module["test"]}')
        elif module.get('test_kind')=='executable_static_validation_command':
            if not any(isinstance(node,ast.FunctionDef) and node.name=='main' for node in ast.parse(testsource).body):
                errors.append(f'Validation command has no executable main: {module["test"]}')
        elif not any(isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name.startswith(('test','Test')) for node in ast.parse(testsource).body): errors.append(f'Test module has no executable test: {module["test"]}')
    link_count=0
    for path in files:
        target=root/path
        if target.suffix=='.py':
            try: ast.parse(target.read_text())
            except SyntaxError: errors.append(f'Invalid Python syntax: {path}')
        if target.suffix!='.md': continue
        for link in re.findall(r'\]\(([^\s)]+)(?:\s+[^)]*)?\)',target.read_text()):
            if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:',link) or link.startswith('#'): continue
            if not (target.parent/unquote(link.split('#')[0])).exists(): errors.append(f'Broken local link in {path}: {link}')
            link_count+=1
    errors.extend(validate_protocol_sources(root))
    req=json.loads((root/'spec/requirements_manifest.json').read_text())['requirements']
    canonical=json.loads((root/'spec/module_manifest.json').read_text())['modules']
    if manifest['modules']!=canonical: errors.append('Repository and module manifests disagree')
    if len({m['code'] for m in canonical})!=len(canonical): errors.append('Duplicate module source paths')
    if len({q['id'] for q in req})!=len(req): errors.append('Duplicate requirement IDs')
    for q in req:
        for key in ('modules','tests'):
            for path in q[key]:
                if not (root/path).is_file(): errors.append(f'Unresolved requirement path: {path}')
    tracked_count=len(subprocess.run(['git','ls-files','-z'],cwd=root,capture_output=True,check=True).stdout.split(b'\0'))-1 if (root/'.git').exists() else None
    return {'status':'FAILED' if errors else 'VERIFIED','intended_static_files':len(files),'tracked_files':tracked_count,'inventory_source':'git' if tracked_count is not None else 'image_filesystem','manifest_entries':len(expected),
            'modules':len(manifest['modules']),'requirements':len(req),'local_markdown_links':link_count,'errors':errors,
            'scope':'static integrity only; run unit/integration/live preflights and native Mermaid parser separately'}
