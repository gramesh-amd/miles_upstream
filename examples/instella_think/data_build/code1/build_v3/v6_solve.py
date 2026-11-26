"""Step 1: per-problem stats from the v6 training rollouts (n, reward sum, solved, truncated, first/last step,
groups).  python3 v6_solve.py <workers> $CODE1_WORK/v6_solve.json"""
import gzip,json,glob,re,collections,sys,os
from multiprocessing import Pool
ROOT=os.environ.get('INSTELLA_RL_ROOT','/it-share-prj2-1/data/goramesh-instella-rl')
S=f'{ROOT}/out/v7_regrade/samples/'
pid_re=re.compile(r'"problem_id":\s*"([^"]+)"'); rw_re=re.compile(r'"reward":\s*"?([-0-9.eE]+)')
st_re=re.compile(r'"step":\s*"?(\d+)'); tr_re=re.compile(r'"truncated":\s*"?(\w+)')
def one(fn):
    out=collections.defaultdict(lambda:[0,0.0,0,0,None,None,0])  # n,sum,nsolved,ntrunc,firststep,laststep,ngroups
    groups=set()
    with gzip.open(fn,'rt') as f:
        for line in f:
            p=pid_re.search(line).group(1); r=float(rw_re.search(line).group(1)); s=int(st_re.search(line).group(1))
            t=tr_re.search(line); t=t and t.group(1) in ('true','True')
            o=out[p]; o[0]+=1;o[1]+=r;o[2]+=r>=1.0-1e-9;o[3]+=bool(t)
            o[4]=s if o[4] is None else min(o[4],s); o[5]=s if o[5] is None else max(o[5],s)
            groups.add(p)
    for p in groups: out[p][6]+=1
    return dict(out)
if __name__=='__main__':
    fs=sorted(glob.glob(S+'step_*.jsonl.gz'))
    agg={}
    with Pool(int(sys.argv[1])) as pool:
        for d in pool.imap_unordered(one,fs):
            for p,o in d.items():
                a=agg.get(p)
                if a is None: agg[p]=list(o)
                else:
                    a[0]+=o[0];a[1]+=o[1];a[2]+=o[2];a[3]+=o[3];a[4]=min(a[4],o[4]);a[5]=max(a[5],o[5]);a[6]+=o[6]
    json.dump(agg,open(sys.argv[2],'w'))
    print(len(agg),sum(a[0] for a in agg.values()))
