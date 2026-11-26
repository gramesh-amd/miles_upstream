import pandas as pd,re,collections
d=pd.read_csv('/it-share-prj2-1/data/goramesh-instella-rl/out/v7_improve/data_audit/master.csv',keep_default_na=False)
par={c:c for c in d.cid}
def f(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
def u(a,b):
    if b in par: par[f(a)]=f(b)
def base(r):
    x=r.v2_reason or r.v1_reason
    m=re.match(r'(.*?(duplicate[^:]*|replaced[^:]*)):(\S+)',x)
    return m
for r in d.itertuples():
    for x in (r.v1_reason,r.v2_reason):
        m=re.search(r'(duplicate_statement_of|minhash_duplicate_of|duplicate_of_recovered|duplicate_of_final_v1|replaced_by_recovered_duplicate_with_more_tests)[^:]*:([\w\-\.]+)',x)
        if m: u(r.cid,m.group(2))
d['cluster']=[f(c) for c in d.cid]
def cat(r):
    if r.in_v2: return 'KEPT'
    if r.v2_reason:
        x=r.v2_reason
        x=re.sub(r':.*','',x)
        return 'v2:'+x
    return re.split(r'[:(]',r.v1_reason)[0]
d['cat']=d.apply(cat,axis=1)
cl_kept=d.groupby('cluster').in_v2.max()
d['cluster_kept']=d.cluster.map(cl_kept)
d.to_csv('/it-share-prj2-1/data/goramesh-instella-rl/out/v7_improve/data_audit/master2.csv',index=False)
s=d[(d.pid!='')&(d.route0=='stdio')].copy()
for c in ['v6_ever','v6_solve','v6_pref']: s[c]=pd.to_numeric(s[c],errors='coerce')
print('v6 stdio rows',len(s),'clusters',s.cluster.nunique(),'clusters with a member in v2',s[s.cluster_kept==1].cluster.nunique())
g=s.groupby('cat').agg(n=('cid','size'),cluster_kept=('cluster_kept','sum'),ever=('v6_ever','mean'),solve=('v6_solve','mean'),pref=('v6_pref','mean')).sort_values('n',ascending=False)
print(g.round(3).to_string())
# lost unique problems: clusters with no member in v2, cat of best member
lost=s[s.cluster_kept==0]
print('lost rows',len(lost),'lost clusters',lost.cluster.nunique())
# for lost clusters, assign cluster category = first non-dup cat among members
rank=lambda c: 1 if ('dup' in c or 'S7' in c or 'P7' in c or 'replaced' in c) else 0
lc=lost.sort_values('cat',key=lambda x:x.map(rank)).groupby('cluster').agg(cat=('cat','first'),ever=('v6_ever','max'),solve=('v6_solve','mean'))
print(lc.groupby('cat').agg(n=('ever','size'),ever=('ever','mean'),solve=('solve','mean')).sort_values('n',ascending=False).round(3).to_string())
