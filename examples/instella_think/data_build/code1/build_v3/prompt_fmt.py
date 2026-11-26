import json,re,collections
T='/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/final_v2/train.jsonl'
C=collections.defaultdict(collections.Counter); N=collections.Counter(); tails=collections.defaultdict(collections.Counter)
heads=collections.defaultdict(collections.Counter)
for l in open(T):
    r=json.loads(l); m=r['metadata']; k=(m['src'],m['route']); p=r['prompt'][0]['content']; N[k]+=1
    pl=p.lower()
    C[k]['mentions_stdin']+=bool(re.search(r'stdin|standard input',pl))
    C[k]['mentions_stdout/print']+=bool(re.search(r'stdout|standard output|\bprint',pl))
    C[k]['says_read_stdin_and_print']+=bool(re.search(r'(read|given|input)[^.]{0,40}(stdin|standard input)',pl) and re.search(r'stdout|standard output|print',pl))
    C[k]['has_python_fence_instr']+=('```python' in p)
    C[k]['backtick_def_sig']+=bool(re.search(r'`def \w+\(',p))
    C[k]['def_anywhere']+=bool(re.search(r'\bdef \w+\(',p))
    C[k]['implement_line']+=bool(re.search(r'Implement `def',p))
    lab=json.loads(r['label'])
    if m['route']=='function':
        fn=re.match(r'assert\s+(?:not\s+)?\(?\s*([A-Za-z_]\w*)',lab[0]) if isinstance(lab[0],str) else None
        if fn: C[k]['test_fn_named_in_prompt']+=fn.group(1) in p
        C[k]['class_based']+=bool(re.search(r'\bclass \w+',p))
    C[k]['has_examples_block']+=bool(re.search(r'example',pl))
    C[k]['len>6000']+=len(p)>6000
    tails[k][re.sub(r'\s+',' ',p[-110:]).strip()[-90:]]+=1
    heads[k][re.sub(r'\s+',' ',p[:70])]+=1
res={}
for k in N:
    res['|'.join(k)]={'rows':N[k],**{a:f'{b} ({b/N[k]:.0%})' for a,b in C[k].items()},
       'top_tails':tails[k].most_common(4),'top_heads':heads[k].most_common(3)}
json.dump(res,open('prompt_format.json','w'),indent=1)
for k,v in res.items():
    print('==',k); [print('  ',a,b) for a,b in v.items() if a not in('top_tails','top_heads')]
    for t,n in v['top_tails']: print('   TAIL',n,repr(t))
    for t,n in v['top_heads']: print('   HEAD',n,repr(t))
