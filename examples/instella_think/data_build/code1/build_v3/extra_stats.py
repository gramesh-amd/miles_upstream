import pandas as pd
d=pd.read_csv('master2.csv',keep_default_na=False)
for c in ['v6_ever','v6_solve']: d[c]=pd.to_numeric(d[c],errors='coerce')
s=d[(d.pid!='')&(d.route0=='stdio')]
dupcats=['S7_duplicate_statement_of','P7_minhash_duplicate_of','v2:duplicate_of_recovered','v2:duplicate_of_final_v1','v2:replaced_by_recovered_duplicate_with_more_tests']
dk=s[s.cat.isin(dupcats)&(s.cluster_kept==1)]
print('dup rows whose problem still trained',len(dk), dk.groupby('cat').size().to_dict())
img=s[s.v1_reason.str.startswith('S4_image')]
print('v6 S4 image rows',len(img),'ever',round(img.v6_ever.mean(),3),'solve',round(img.v6_solve.mean(),3),'in v2',img.in_v2.sum(),'cluster kept',img.cluster_kept.sum())
mu=s[s.v1_reason.str.startswith('S3_multi')]
print('v6 S3 multi rows',len(mu),'ever',round(mu.v6_ever.mean(),3),'in_v2',mu.in_v2.sum(),'cluster kept',mu.cluster_kept.sum())
ft=s[s.v1_reason.str.match(r'(S_lt3|P6_lt5)')]
print('v6 few-tests rows',len(ft),'ever',round(ft.v6_ever.mean(),3))
