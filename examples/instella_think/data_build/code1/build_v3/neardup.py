import json,re,sys,collections,numpy as np,zlib,gzip
def toks(p):
    p=re.sub(r'\\[a-zA-Z]+|\$','',p.lower()); return re.findall(r'[a-z0-9]+',p)
def shingles(p,n=5):
    t=toks(p); return {zlib.crc32(' '.join(t[i:i+n]).encode()) for i in range(max(1,len(t)-n+1))}
def load(paths):
    out=[]
    for pth,tag in paths:
        op=gzip.open if pth.endswith('.gz') else open
        for l in op(pth,'rt'):
            r=json.loads(l)
            if 'prompt' in r and isinstance(r['prompt'],list): p=r['prompt'][0]['content']; m=r['metadata']; cid=m['custom_id']; src=m['src']; route=m['route']
            else: p=r['prompt']; cid=r['cid']; src=r['src']; route='stdio'
            out.append((tag,cid,src,route,p))
    return out
def run(items,thr=0.7):
    K=64; rng=np.random.RandomState(0); A=rng.randint(1,2**31-1,K,dtype=np.int64); B=rng.randint(0,2**31-1,K,dtype=np.int64); P=2**31-1
    S=[shingles(x[4]) for x in items]; sig=np.zeros((len(items),K),dtype=np.int64)
    for i,s in enumerate(S):
        a=np.fromiter(s,dtype=np.int64); sig[i]=((a[:,None]*A+B)%P).min(0)
    cand=set()
    for b in range(16):
        bk=collections.defaultdict(list)
        for i in range(len(items)): bk[sig[i,b*4:(b+1)*4].tobytes()].append(i)
        for v in bk.values():
            if 1<len(v)<200:
                for x in range(len(v)):
                    for y in range(x+1,len(v)): cand.add((v[x],v[y]))
    pairs=[]
    for i,j in cand:
        jac=len(S[i]&S[j])/len(S[i]|S[j])
        if jac>=thr:
            di=re.findall(r'\d+',items[i][4]); dj=re.findall(r'\d+',items[j][4])
            pairs.append((i,j,round(jac,3),di==dj))
    return pairs
if __name__=='__main__':
    import os
    T=os.environ.get('INSTELLA_RL_ROOT','/it-share-prj2-1/data/goramesh-instella-rl')+'/out/v7_data/final_v2/train.jsonl'
    items=load([(T,'train')]+[(p,'cand') for p in sys.argv[1:]])
    pairs=run(items)
    with open('neardup_pairs.jsonl' if len(sys.argv)==1 else 'neardup_pairs_with_cands.jsonl','w') as f:
        for i,j,jac,same in pairs:
            f.write(json.dumps(dict(a=items[i][1],b=items[j][1],ta=items[i][0],tb=items[j][0],sa=items[i][2]+'|'+items[i][3],sb=items[j][2]+'|'+items[j][3],jac=jac,same_digits=same))+'\n')
    c=collections.Counter((items[i][0],items[j][0],items[i][2]+'|'+items[i][3],items[j][2]+'|'+items[j][3],same) for i,j,_,same in pairs)
    for k,v in sorted(c.items(),key=lambda x:-x[1]): print(v,k)
