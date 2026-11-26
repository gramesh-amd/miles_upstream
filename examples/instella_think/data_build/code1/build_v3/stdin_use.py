import gzip,json,glob,re,collections,csv,sys
from multiprocessing import Pool
import pandas as pd
d=pd.read_csv('master2.csv',keep_default_na=False)
d=d[(d.pid!='')&(d.route0=='stdio')]
P={r.pid:r.src for r in d.itertuples()}
def one(fn):
    c=collections.Counter()
    for l in gzip.open(fn,'rt'):
        x=json.loads(l); s=P.get(x['problem_id'])
        if not s: continue
        resp=x['response']; ans=resp.split('</think>')[-1] if '</think>' in resp else ''
        m=re.findall(r'```(?:python|py)?\n(.*?)```',ans,re.S)
        code=m[-1] if m else ''
        reads=bool(re.search(r'input\(|sys\.stdin|stdin\.read|open\(0\)',code))
        func_only=bool(code) and not reads
        c[s,'n']+=1; c[s,'has_code']+=bool(code); c[s,'reads_stdin']+=reads; c[s,'code_no_stdin']+=func_only
        c[s,'no_stdin_reward>0']+=func_only and x['reward']>0
    return c
if __name__=='__main__':
    fs=sorted(glob.glob('/it-share-prj2-1/data/goramesh-instella-rl/out/v7_regrade/samples/step_*.jsonl.gz'))[::4]
    tot=collections.Counter()
    with Pool(60) as p:
        for c in p.imap_unordered(one,fs): tot.update(c)
    print(sorted(tot.items()))
