#!/usr/bin/env python3
"""Synchronize intended static paths and diagram mappings after source changes.

Uses Git's tracked/nonignored inventory; never adds ignored secrets or runtime data.
"""
from pathlib import Path
import hashlib,json,subprocess

ROOT=Path(__file__).resolve().parents[1]
DIAGRAM='spec/mermaid/repository/implementation_extensions.mmd'


def main():
    (ROOT/DIAGRAM).touch(exist_ok=True)
    inventory=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT,capture_output=True,check=True).stdout.decode().split('\0')
    files=sorted({p for p in inventory if p and (ROOT/p).is_file()})
    directories={'.'}
    for file in files:
        directories.update(str(p) for p in Path(file).parents if str(p)!='.')
    manifest=json.loads((ROOT/'spec/repository_manifest.json').read_text())
    entries={e['path']:e for e in manifest['entries'] if e['path'] in set(files)|directories}
    for path in sorted(set(files)|directories):
        if path not in entries:
            directory=path in directories
            entries[path]={'path':path,'kind':'directory' if directory else 'implementation_extension','entry_type':'directory' if directory else 'file','owner':'coordinator','source':'User implementation assignment; operational H9 conventions','display_path':path+'/' if directory else path,'diagram':DIAGRAM,'node':'p_'+hashlib.sha256(path.encode()).hexdigest()[:14]}
        if path in files:
            entries[path]['blank']=(ROOT/path).stat().st_size==0
            if not entries[path]['blank'] and entries[path]['kind'].startswith('blank_'):entries[path]['kind']=entries[path]['kind'].replace('blank_','implemented_',1)
    lines=['---','title: "Implementation extension path inventory"','---','flowchart TD','    EXT["Operational implementations, deployment and validation evidence"]']
    for path,e in sorted(entries.items()):
        if e['diagram']==DIAGRAM:
            lines.extend([f'    {e["node"]}["{e["display_path"]}"]',f'    EXT --> {e["node"]}'])
    (ROOT/DIAGRAM).write_text('\n'.join(lines)+'\n')
    manifest['entries']=[entries[p] for p in sorted(entries)]
    manifest['files']=len(files);manifest['directories_including_root']=len(directories)
    manifest['scope']='Complete intended tracked source/documentation tree; runtime datasets/artifacts,credentials and dependency caches excluded.'
    canonical=json.loads((ROOT/'spec/module_manifest.json').read_text())['modules']
    if len({m['code'] for m in canonical})!=len(canonical):
        raise RuntimeError('Canonical module manifest contains duplicate source paths')
    manifest['modules']=canonical
    (ROOT/'spec/repository_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Complete static repository hierarchy','','Generated from the intended source/documentation manifest. Runtime caches,private credentials,large datasets/checkpoints and Git internals are excluded. Populated scientific implementations and executed fixture tests are distinguished from blocked live preflights in [validation_report.md](validation_report.md).','','```text']
    for p,e in sorted(entries.items()):
        if p=='.':lines.append(ROOT.name+'/');continue
        lines.append('  '*len(Path(p).parts)+Path(p).name+('/' if e['kind']=='directory' else ''))
    lines.append('```')
    (ROOT/'docs/repository_layout.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'files':len(files),'directories':len(directories),'modules':len(manifest['modules'])}))

if __name__=='__main__':main()
