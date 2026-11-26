import pandas as pd, pickle, sys, re
W="work"
c=pd.read_parquet(f'{W}/candidates.parquet'); ev=pickle.load(open(f'{W}/eval.pkl','rb')); tr=pickle.load(open(f'{W}/train.pkl','rb'))
es=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 260
g=c[c.eval_set==es]
def cl(s): 
    s=re.sub(r'\s+',' ',s); return s
for eid, gg in g.groupby('eval_id', sort=False):
    e=ev[gg.ei.iloc[0]]
    # for pro sets show the new part text
    parts={ev[i]['part']:ev[i]['text'] for i in gg.ei}
    print(f"\n### {eid} [{','.join(sorted(parts))}] EVAL: {cl(' || '.join(parts.values()))[:n+150]}")
    for r in gg.sort_values('tfidf',ascending=False).itertuples():
        print(f"  - {r.custom_id} ({r.src}) [{r.why}] tf={r.tfidf:.2f}: {cl(tr[r.ti]['text'])[:n]}")
