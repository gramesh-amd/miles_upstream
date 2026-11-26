import json,pickle,sys,tempfile,collections
sys.path.insert(0,'/tmp/wt_lux/examples/instella_think'); sys.path.insert(0,'.')
import code_exec_server as ces
from concurrent.futures import ThreadPoolExecutor
fails=set(json.load(open("/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/function/verify_grader.json"))["ref_fail_rows"])
fin=pickle.load(open('/tmp/v7fn/final.pkl','rb'))['final']
A={a['row']:a for a in pickle.load(open('/tmp/v7fn/audit.pkl','rb'))}
rows={r['row']:r for r in pickle.load(open('/tmp/v7fn/rows.pkl','rb'))}
runner=ces.Runner(12,2048,sys.executable,tempfile.mkdtemp(prefix='v7rf-'))
import sandbox
def chk(d):
    a=A[d['row']]; ok=[i for i,x in enumerate(a['refs']) if x['prog']=='ok']
    T=d['tests_final']
    cand=[i for i in ok if sandbox.passvec(sandbox.run(a['ref_codes'][i]+d.get('verify_suffix',''),T))==[1]*len(T)]
    good=None; vecs=[]
    for i in cand[:4]:
        r=runner.assert_tests(a['ref_codes'][i]+d.get('verify_suffix',''),T,5.0,name='__main__')
        vecs.append(''.join(map(str,r)))
        if all(r): good=i; break
    return d['row'],good,vecs
res={}
with ThreadPoolExecutor(4) as ex:
    for row,good,vecs in ex.map(chk,[d for d in fin if d['row'] in fails]):
        res[row]=(good,vecs)
print('fixed by another ref',sum(g is not None for g,_ in res.values()),'of',len(res))
import os
prev=pickle.load(open('/tmp/v7fn/recheck_fail.pkl','rb')) if os.path.exists('/tmp/v7fn/recheck_fail.pkl') else {}
prev.update(res)
pickle.dump(prev,open('/tmp/v7fn/recheck_fail.pkl','wb'))
for row,(g,v) in list(res.items()):
    if g is None: print(rows[row]['custom_id'], v[:2], rows[row]['prompt'][:120].replace('\n',' '))
