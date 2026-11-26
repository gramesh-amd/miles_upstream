import json,csv,gzip,ast,collections
V7='/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data'
ids=[json.loads(l) for l in open(f'{V7}/input/row_ids.jsonl')]
rows=[]
for l,i in zip(open(f'{V7}/input/code-train/dolci-code-train.jsonl'),ids):
    r=json.loads(l); lab=json.loads(r['label'])
    stdio=isinstance(lab,list) and bool(lab) and isinstance(lab[0],dict)
    rows.append(dict(row=i['row'],cid=i['custom_id'],src=i['src'],olmo=i['olmo_passrate'],route0='stdio' if stdio else 'function',ntests0=len(lab) if isinstance(lab,list) else None))
cid_count=collections.Counter(r['cid'] for r in rows)
pidmap={x['custom_id_full']:x['problem_id'] for x in csv.DictReader(open(f'{V7}/../v6/rollout_analysis/eval_transfer/v6_pid_to_custom_id.csv'))}
pref={}
for l in gzip.open(f'{V7}/../v7_regrade/problems.jsonl.gz','rt'):
    d=json.loads(l); md=d['metadata']; md=md if isinstance(md,dict) else ast.literal_eval(md)
    pref[d['problem_id']]=(md.get('prefilter') or {}).get('n_correct')
solve=json.load(open('/it-share-prj2-1/data/goramesh-instella-rl/out/v7_improve/data_audit/v6_solve.json'))
train={}
for l in open(f'{V7}/final_v2/train.jsonl'):
    m=json.loads(l)['metadata']; train[m['custom_id']]=m
d1={r['custom_id']:r['reason'] for r in csv.DictReader(open(f'{V7}/final_v1/dropped.csv'))}
d2={r['custom_id']:r['reason'] for r in csv.DictReader(open(f'{V7}/final_v2/dropped_v2.csv'))}
out=[]
for r in rows:
    c=r['cid']; p=pidmap.get(c)
    s=solve.get(p) if p else None
    r.update(pid=p or '',v6_pref=pref.get(p,'') if p else '',
             v6_n=s[0] if s else 0, v6_mean=round(s[1]/s[0],4) if s else '', v6_solve=round(s[2]/s[0],4) if s else '',
             v6_ever=int(s[2]>0) if s else '', v6_groups=s[6] if s else 0,
             in_v2=int(c in train), route=train[c]['route'] if c in train else r['route0'],
             v1_reason=d1.get(c,''), v2_reason=d2.get(c,''))
    out.append(r)
with open('/it-share-prj2-1/data/goramesh-instella-rl/out/v7_improve/data_audit/master.csv','w') as f:
    w=csv.DictWriter(f,list(out[0].keys())); w.writeheader(); w.writerows(out)
print(len(out),'dup cids',sum(v>1 for v in cid_count.values()),'in_v2',sum(r['in_v2'] for r in out),'in_v6',sum(bool(r['pid']) for r in out))
