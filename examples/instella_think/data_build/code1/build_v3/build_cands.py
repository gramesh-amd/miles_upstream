"""Step 4: stdio rows dropped by final_v1/final_v2 that may be fine (decorative image placeholders, multi-answer
statements, few tests), with their tests and candidate reference solutions. Writes $CODE1_WORK/restore_cands.jsonl.gz."""
import json,gzip,pickle,collections,os,pandas as pd
ROOT=os.environ.get('INSTELLA_RL_ROOT','/it-share-prj2-1/data/goramesh-instella-rl')
WORK=os.environ.get('CODE1_WORK',f'{ROOT}/out/v7_improve/data_audit')
V7=f'{ROOT}/out/v7_data'
d=pd.read_csv(f'{WORK}/master2.csv',keep_default_na=False)
CATS={'S4_image_placeholder_tag_not_decorative':'image','S4_image_placeholder_phrase_not_decorative':'image',
 'v2:openr1_statement_still_refers_to_image':'image','v2:openr1_no_generated_checker':'multi_noR1checker',
 'S_lt3_tests_unaugmentable':'few_tests','P6_lt5_tests':'few_tests','v2:lt5_tests_before_merge':'few_tests',
 'S3_multi_answer_no_checker':'multi_nomap','v2:multi_answer_no_checker':'multi_nomap','S_non_english_statement':'non_english'}
c=d[(d.route0=='stdio')&(d.in_v2==0)&(d.cluster_kept==0)&d.cat.isin(CATS)]
want={int(r.row):dict(cat=r.cat,group=CATS[r.cat],cid=r.cid,src=r.src,pid=r.pid) for r in c.itertuples()}
drop=set(open(f'{V7}/decontam/drop_list.txt').read().split())
A=pickle.load(open(f'{V7}/stdio/work/analysis.pkl','rb'))
out=[];stat=collections.Counter()
for l in gzip.open(f'{V7}/stdio/work/rows.jsonl.gz','rt'):
    v=json.loads(l); r=int(v['row'])
    if r not in want: continue
    w=want[r]
    if w['cid'] in drop: stat['decontam']+=1; continue
    if v['kind']!='stdio' or not v['label'] or not isinstance(v['label'][0],dict): stat['nonstdio']+=1; continue
    tests=[];seen=set()
    for t in list(v['label'])+list(v.get('full_tests') or []):
        if not isinstance(t,dict): continue
        k=' '.join(str(t['input']).split())
        if k in seen: continue
        if len(str(t['input']))+len(str(t['output']))>1_000_000: continue
        seen.add(k); tests.append({'input':str(t['input']),'output':str(t['output'])})
    tests=tests[:32]
    refs=[x for x in v['refs'] if x['origin']=='v6']+[x for x in v['refs'] if x['origin']=='olmo']+[x for x in v['refs'] if x['origin']=='v6near']
    a=A.get(r,{})
    opts={}
    if a.get('tol') is not None: opts['float_tol']=a['tol'] if isinstance(a['tol'],float) else 1e-6
    if a.get('case_stmt'): opts['case_insensitive']=True
    out.append(dict(w,row=r,prompt=v['prompt'],metadata=v['metadata'],tests=tests,n_label=len(v['label']),refs=refs[:8],opts=opts,
                    multi_hits=a.get('multi_hits'),image_tags=a.get('image_tags')))
    stat[w['group']]+=1
with gzip.open(f'{WORK}/restore_cands.jsonl.gz','wt') as g:
    for o in out: g.write(json.dumps(o,ensure_ascii=False)+'\n')
print(stat,len(out))
print(collections.Counter((o['group'],min(len(o['tests']),6)) for o in out))
print(collections.Counter((o['group'],len(o['refs'])>0) for o in out))
