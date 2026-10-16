"""Independent recomputation for the matched numerical contract diagnostic."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path('/home/aimldl/LH-VLN/runs/balanced100_matched_contract_20261013_v1')
protocol=json.loads((ROOT/'protocol_frozen.json').read_text())
status=json.loads((ROOT/'status.json').read_text())
assert status['phase']=='completed'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for path,digest in protocol['dependencies'].items():assert sha(Path(path))==digest,path
for path,digest in json.loads((ROOT/'source_frame_hashes.json').read_text()).items():assert sha(Path(path))==digest,path
matched=json.loads((ROOT/'training/matched_verification.json').read_text())
assert matched['passed']
a,b=(matched['arms'][arm] for arm in ('historical_contract','repaired_contract'))
assert a['initialization_sha256']==b['initialization_sha256']
assert a['draws_sha256']==b['draws_sha256']
for arm,item in matched['arms'].items():
    assert item['final']['step']==4000
    assert item['checkpoint_sha256']==sha(ROOT/'training'/(arm+'.pt'))
summary=json.loads((ROOT/'summary.json').read_text())
paired={}
for split in ('train','dev','balanced100'):
    cohorts={}
    for arm in ('historical_contract','repaired_contract','random'):
        folder=ROOT/'evaluation'/split/arm
        rows=[json.loads(line) for line in (folder/'traces.jsonl').read_text().splitlines()]
        indices=list(range(100)) if split=='balanced100' else [r['index'] for r in protocol['cohort'][split]]
        assert [r['index'] for r in rows]==indices
        for row in rows:
            assert row['status']=='completed'
            assert len(row['steps'])==len(row['actions'])<=150
            assert len(row['successes'])==row['target_count']
            stage=0
            for i,step in enumerate(row['steps']):
                assert step['t']==i and step['active_target']==stage
                if step['action']=='stop':
                    distance=step['distances_after'][stage]
                    if distance is None:assert not step['active_target_success_after']
                    elif abs(distance-1)>1e-6:assert step['active_target_success_after']==(distance<1)
                    assert row['successes'][stage]==step['active_target_success_after']
                    stage+=1
        u=[[int(v) for v in r['successes']] if r['reference_path_length'] is not None else [0]*len(r['successes']) for r in rows]
        recomputed={'episodes':len(rows),'chains':sum(all(v) for v in u),'stage_success':sum(map(sum,u)),'stage_count':sum(map(len,u)),'ISR_macro':sum(sum(v)/len(v) for v in u)/len(u),'CSR_local_macro':sum(sum(v[j]*(1+(1 if j==0 else v[j-1])) for j in range(len(v)))/(2*len(v)) for v in u)/len(u)}
        saved=json.loads((folder/'metrics.json').read_text())
        assert all(math.isclose(saved[k],v,abs_tol=1e-12) for k,v in recomputed.items())
        assert saved==summary[split][arm]
        cohorts[arm]={r['index']:r for r in rows}
    pairs=[]
    for i in indices:
        old,new=(cohorts[arm][i] for arm in ('historical_contract','repaired_contract'))
        assert old['start_state']==new['start_state']==cohorts['random'][i]['start_state']
        assert old['reference_path_length']==new['reference_path_length']==cohorts['random'][i]['reference_path_length']
        valid=old['reference_path_length'] is not None
        pairs.append({'index':i,'old_chain':bool(valid and all(old['successes'])),'new_chain':bool(valid and all(new['successes'])),'old_stages':sum(old['successes']) if valid else 0,'new_stages':sum(new['successes']) if valid else 0})
    paired[split]={'old_only_chains':sum(p['old_chain'] and not p['new_chain'] for p in pairs),'new_only_chains':sum(p['new_chain'] and not p['old_chain'] for p in pairs),'stage_gain_tasks':sum(p['new_stages']>p['old_stages'] for p in pairs),'stage_loss_tasks':sum(p['new_stages']<p['old_stages'] for p in pairs),'rows':pairs}
result={'passed':True,'episodes_verified':324,'matched_training':True,'paired':paired,'summary':summary,'scope':'one-seed numerical diagnostic; not human grounding or unseen-scene validation'}
(ROOT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
table=['| Cohort | Arm | Chain sukses | Stage sukses | ISR macro (%) | CSR local (%) |','|---|---|---:|---:|---:|---:|']
for split,arms in summary.items():
    for arm,metrics in arms.items():
        table.append(f"| {split} | {arm} | {metrics['chains']}/{metrics['episodes']} | {metrics['stage_success']}/{metrics['stage_count']} | {100*metrics['ISR_macro']:.2f} | {100*metrics['CSR_local_macro']:.2f} |")
report='\n'.join(['# Matched contract: hasil terverifikasi','', '## Charter','Uji paket alignment observasi–aksi dan yaw runtime pada data, arsitektur, seed, sampling, dan budget yang sama. Diagnosis numerik satu seed; grounding held tidak diubah.','', '## Canonical sources','protocol_frozen.json; training/matched_verification.json; evaluation/*/*/traces.jsonl; independent_verification.json.','', '## Findings ledger','F1: inisialisasi dan seluruh draws identik; kedua arm4000updates. F2: seluruh324episode diverifikasi dari trace. F3: tabel di bawah dihitung dengan reference invalid tetap nol.','']+table+['','## Scope filter','Kontrak historis di sini merupakan model baru pada cohort4task, bukan checkpoint historis300trajectory. Seleksi checkpoint final dibuat sama; efek snapshot bug tidak diuji.','', '## Object filter','Train/development dan balanced100 sudah dipakai pengembangan. Keberhasilan numerik tidak memvalidasi grounding manusia. Teacher privileged56/100 pada audit sebelumnya bukan learned SR.','', '## Conclusions','Hasil F1–F3 mendukung perbandingan numerik kedua arm pada cohort ini. Efek alignment dan yaw belum dipisahkan; klaim peningkatan generalisasi membutuhkan seed berulang dan evaluasi yang tidak dipakai memilih model.','', '## Next steps / Risks','Gunakan perbandingan train/dev/balanced100 dan pasangan task dalam independent_verification.json untuk memilih intervensi berikutnya. Data tambahan perlu gate demonstrasi/grounding; recovery hanya pada train. Tidak ada perubahan paper/Tabel4 atau checkpoint lama.',''])
(ROOT/'result_report.md').write_text(report)
(ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'},indent=2)+'\n')
print(json.dumps({'passed':True,'summary':summary,'paired':{k:{a:v for a,v in p.items() if a!='rows'} for k,p in paired.items()}},indent=2))
