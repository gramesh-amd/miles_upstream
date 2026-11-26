"""Verify restore candidates through the sandboxed grader (training options, 5 s/test). All code runs in the server."""
import json,gzip,sys,time,collections
sys.path.insert(0,'/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/final_v1/scripts')
import gclient as gc
from concurrent.futures import ThreadPoolExecutor
C=[json.loads(l) for l in gzip.open(sys.argv[1],'rt')]
def job(c):
    t0=time.time(); out=dict(cid=c['cid'],group=c['group'],n_tests=len(c['tests']))
    tried=[]
    for j,ref in enumerate(c['refs']):
        res=gc.run('stdio',ref['code'],c['tests'],c['opts'],stop_on_fail=True)
        if res is None: out['why']='grader_unreachable'; break
        tried.append(ref['origin'])
        if all(res):
            out.update(verified=True,ref=j,origin=ref['origin'])
            if c['opts']:
                ex=gc.run('stdio',ref['code'],c['tests'],{})
                out['exact_pass']=bool(ex) and all(ex)
            break
    else:
        out.update(verified=False,why='no_reference_passes')
    # how many distinct refs pass (multi-answer evidence) -- only for multi groups, up to 4 more refs
    if c['group'].startswith('multi') and out.get('verified'):
        npass=0;nrun=0
        for k,ref in enumerate(c['refs']):
            if k==out['ref'] or nrun>=4: continue
            res=gc.run('stdio',ref['code'],c['tests'],c['opts']); nrun+=1
            if res and all(res): npass+=1
        out.update(other_refs_run=nrun,other_refs_pass=npass)
    out['tried']=tried; out['secs']=round(time.time()-t0,1)
    return out
t0=time.time(); res=[]
with ThreadPoolExecutor(int(sys.argv[3])) as ex:
    for i,o in enumerate(ex.map(job,C)):
        res.append(o)
        if (i+1)%100==0: print(i+1,f'{time.time()-t0:.0f}s',flush=True)
with open(sys.argv[2],'w') as f:
    for o in res: f.write(json.dumps(o)+'\n')
print(collections.Counter((o['group'],o.get('verified',False)) for o in res))
