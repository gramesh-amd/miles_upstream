import json,collections,hashlib,re,sys
def norm(s): return ' '.join(str(s).split())
def keys(row):
    lab=json.loads(row['label']) if isinstance(row['label'],str) else row['label']
    if row['metadata']['route']=='stdio':
        return {hashlib.md5((norm(t['input'])+'\x00'+norm(t['output'])).encode()).hexdigest() for t in lab if len(norm(t['input']))>=8}
    return {hashlib.md5(norm(re.sub(r'^assert\s+','',t)).encode()).hexdigest() for t in lab if isinstance(t,str)}
def pairs(rows,minshare=5,frac=0.5):
    K=[keys(r) for r in rows]; idx=collections.defaultdict(list)
    for i,k in enumerate(K):
        for h in k: idx[h].append(i)
    cnt=collections.Counter()
    for h,v in idx.items():
        if len(v)>30: continue
        for a in range(len(v)):
            for b in range(a+1,len(v)): cnt[v[a],v[b]]+=1
    out=[]
    for (i,j),n in cnt.items():
        m=min(len(K[i]),len(K[j]))
        if n>=min(minshare,m) and n>=frac*m and m>=3: out.append((i,j,n,m))
    return out
if __name__=='__main__':
    import os
    rows=[json.loads(l) for l in open(os.environ.get('INSTELLA_RL_ROOT','/it-share-prj2-1/data/goramesh-instella-rl')+'/out/v7_data/final_v2/train.jsonl')]
    P=pairs(rows)
    c=collections.Counter((rows[i]['metadata']['src']+'|'+rows[i]['metadata']['route'],rows[j]['metadata']['src']+'|'+rows[j]['metadata']['route']) for i,j,_,_ in P)
    print(len(P),c.most_common())
    import random; random.seed(0)
    for i,j,n,m in random.sample(P,min(6,len(P))):
        print(n,m,rows[i]['metadata']['custom_id'],rows[j]['metadata']['custom_id']); print('   A:',rows[i]['prompt'][0]['content'][:160].replace('\n',' ')); print('   B:',rows[j]['prompt'][0]['content'][:160].replace('\n',' '))
    json.dump([(rows[i]['metadata']['custom_id'],rows[j]['metadata']['custom_id'],n,m) for i,j,n,m in P],open('testdup_pairs.json','w'))
