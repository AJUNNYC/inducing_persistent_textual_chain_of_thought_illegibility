from pathlib import Path
import sqlite3,json
r=Path('/opt/runpod_work/tla6hi5owr4prx')
for level in [1,3,5,7,8,9,10]:
 for epoch in [1,2]:
  fit=r/(f'artifacts/sft_14b_campaign_9de8813/training/sft/kazitext/14b/level_{level:02d}' if epoch==1 else f'training_oct28/kazitext_level{level:02d}_epoch2')
  result=json.loads((fit/'fit_result.json').read_text());assert result['status']=='complete'
  db=r/'artifacts/sft_14b_campaign_9de8813/state/artifact_registry.sqlite3' if epoch==1 else fit/'drive_registry.sqlite3'
  c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
  rows=c.execute('select local_path,sync_status,record from artifacts').fetchall();c.close()
  records={path:(status,json.loads(raw)) for path,status,raw in rows}
  required=[fit/'fit_result.json',*list((fit/'checkpoint_000625').glob('*')),*[p for p in (fit/'final_adapter').rglob('*') if p.is_file()]]
  missing=[]
  for p in required:
   rec=records.get(str(p))
   if not rec or rec[0]!='VERIFIED' or not rec[1].get('drive_file_id'):missing.append(p.name)
  print('LEVEL',level,'EPOCH',epoch,'REQUIRED',len(required),'UNVERIFIED',missing,flush=True)
