import json,collections,csv,statistics as st
D='/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/final_v3_candidate'
rows=[json.loads(l) for l in open(f'{D}/train.jsonl')]
u=[r for r in rows if 'oversample_of' not in r['metadata']]
c=collections.Counter((r['metadata']['src'],r['metadata']['route']) for r in u)
cd=collections.Counter((r['metadata']['src'],r['metadata']['route']) for r in rows)
ch=collections.Counter(x for r in u for x in r['metadata'].get('v7_changes',[]) if x.startswith('v3') or x=='openr1_checker')
pref=collections.Counter((r['metadata']['route'],r['metadata'].get('v6_prefilter_n_correct')) for r in rows)
nt={rt:[len(json.loads(r['label'])) for r in u if r['metadata']['route']==rt] for rt in ('function','stdio')}
keys=collections.Counter(tuple(sorted(r['metadata'].keys()))==tuple(sorted(r['metadata'].keys())) for r in u)
v2keys=set(json.loads(open('/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/final_v2/train.jsonl').readline())['metadata'])
missing=collections.Counter(k for r in u for k in ('rm_type','split','n_tests','custom_id','src','route','v7_changes','verified') if k not in r['metadata'])
nchk=sum(1 for r in u if r['metadata'].get('checker'))
ntest_ok=sum(1 for r in u if r['metadata']['n_tests']==len(json.loads(r['label'])))
dates=[r['metadata'].get('cf_date') for r in u if r['metadata'].get('cf_date')]
out=dict(rows_written=len(rows),unique=len(u),oversample_copies=len(rows)-len(u),unique_by_src_route={f'{a}|{b}':v for (a,b),v in sorted(c.items())},
 draws_by_src_route={f'{a}|{b}':v for (a,b),v in sorted(cd.items())},v3_changes=dict(ch),checker_rows=nchk,
 prefilter_draws={f'{a}|{b}':v for (a,b),v in sorted(pref.items(),key=str)},
 tests={rt:dict(mean=round(st.mean(v),1),median=st.median(v),min=min(v),max=max(v)) for rt,v in nt.items()},
 missing_required_keys=dict(missing),n_tests_consistent=ntest_ok,max_cf_date=max(dates))
print(json.dumps(out,indent=1))
s=json.load(open(f'{D}/stats.json')); s['final']=out; json.dump(s,open(f'{D}/stats.json','w'),indent=1)
d=list(csv.DictReader(open(f'{D}/dropped_v3.csv')))
print(collections.Counter((r['reason'].split(':')[0] if not r['reason'].startswith('slow') else 'slow_reference_gt4s',r['src'],r['route']) for r in d))
