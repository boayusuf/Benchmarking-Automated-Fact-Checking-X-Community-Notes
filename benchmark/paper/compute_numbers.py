"""Recomputes every statistic reported in the paper from master_results.csv and writes numbers.json.
Run from anywhere: python benchmark/paper/compute_numbers.py"""
import sys, math, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gmap import map_rating
import numpy as np, pandas as pd
from scipy.stats import binomtest, norm
from sklearn.metrics import cohen_kappa_score
R=str(HERE.parent / 'data') + '/'
m=pd.read_csv(R+'results/master_results.csv',dtype=str)
ex=pd.read_csv(R+'filtered/extracted_claims.csv',dtype=str)
err=pd.read_csv(R+'results/error_analysis_final.csv',dtype=str)
has=m.google_verdict!='NO_RESULT'
m.loc[has,'google_verdict']=m.loc[has,'google_raw_rating'].fillna('').map(map_rating)
S={'A':'gpt4o_verdict','B':'tavily_gpt4o_verdict','C':'google_verdict','D':'filtered_rag_verdict','E':'llama_verdict'}
m['y']=(m.cn_verdict=='NOT_MISLEADING').astype(int)
for k,c in S.items(): m['p'+k]=m[c].map({'TRUE':1,'FALSE':0})
orig=m[m.tavily_gpt4o_verdict.notna()|m.filtered_rag_verdict.notna()]
def met(df,k):
    s=df[df['p'+k].notna()]; yt=s.y.values; yp=s['p'+k].astype(int).values
    tp=((yt==1)&(yp==1)).sum(); fp=((yt==0)&(yp==1)).sum(); fn=((yt==1)&(yp==0)).sum(); tn=((yt==0)&(yp==0)).sum()
    P=tp/(tp+fp) if tp+fp else 0; Rr=tp/(tp+fn) if tp+fn else 0; F=2*P*Rr/(P+Rr) if P+Rr else 0
    PM=tn/(tn+fn) if tn+fn else 0; RM=tn/(tn+fp) if tn+fp else 0
    return dict(n=len(s),acc=(yt==yp).mean(),kappa=cohen_kappa_score(yt,yp),P=P,R=Rr,F1=F,PM=PM,RM=RM,tn=tn,fp=fp,fn=fn,tp=tp,
                base_mis=(yt==0).mean(),pred_false=(yp==0).mean())
out={}
for k in S:
    out['orig_'+k]=met(orig,k)
    out['exp_'+k]=met(m,k)
    for cat in ['POLITICAL','HEALTH','SCIENTIFIC']:
        out[f'exp_{k}_{cat}']=met(m[m.category==cat],k); out[f'orig_{k}_{cat}']=met(orig[orig.category==cat],k)
    v=m[S[k]].dropna(); out['mixed_'+k]=dict(rate=(v=='MIXED').mean(), n=len(v))
def mc(df,a,b):
    s=df[df['p'+a].notna()&df['p'+b].notna()]; ca=s['p'+a]==s.y; cb=s['p'+b]==s.y
    n10=int((ca&~cb).sum()); n01=int((~ca&cb).sum())
    return dict(n=len(s),both=int((ca&cb).sum()),a_only=n10,b_only=n01,neither=int((~ca&~cb).sum()),p=binomtest(min(n10,n01),n10+n01,.5).pvalue if n10+n01 else 1.0,
                acc_a=ca.mean(),acc_b=cb.mean(),agree=(s['p'+a]==s['p'+b]).mean())
for a,b,df,tag in [('A','B',orig,'o'),('A','D',orig,'o'),('A','E',orig,'o'),('B','D',orig,'o'),('A','E',m,'x')]:
    out[f'mc_{a}{b}_{tag}']=mc(df,a,b)
s=m[m.pA.notna()&m.pE.notna()]; ca=(s.pA==s.y).values.astype(float); ce=(s.pE==s.y).values.astype(float)
rng=np.random.default_rng(42); idx=rng.integers(0,len(s),(5000,len(s))); dif=(ca[idx].mean(1)-ce[idx].mean(1))*100
out['boot']=dict(lo=np.percentile(dif,2.5),hi=np.percentile(dif,97.5),diff=(ca.mean()-ce.mean())*100)
nd=out['mc_AE_x']['a_only']+out['mc_AE_x']['b_only']; N=out['mc_AE_x']['n']
out['power']={str(d):float(1-norm.cdf(1.96-d/100*N/math.sqrt(nd))+norm.cdf(-1.96-d/100*N/math.sqrt(nd))) for d in [1,1.55,2]}
cov=m.google_verdict.isin(['TRUE','FALSE']); mis=m.y==0
out['google']=dict(any_match=has.mean(),strict=cov.mean(),mis_any=(has&mis).sum()/mis.sum(),mis_strict=(cov&mis).sum()/mis.sum(),
  mis_flag_all=((m.pC==0)&mis).sum()/mis.sum(), mis_graded_false=((m.pC==0)&mis).sum()/(cov&mis).sum(), not_graded=int((cov&~mis).sum()), not_graded_false=int(((m.pC==0)&~mis).sum()),
  pol=cov[m.category=='POLITICAL'].mean(),hea=cov[m.category=='HEALTH'].mean(),sci=cov[m.category=='SCIENTIFIC'].mean())
m=m.merge(ex[['noteId','image_dependent']],on='noteId',how='left'); m['img']=m.image_dependent.str.lower().eq('true')
out['img']=dict(exp=m.img.mean(),orig=m[m.noteId.isin(orig.noteId)].img.mean())
for k in S: out['noimg_'+k]=met(m[~m.img] if k in 'ACE' else m[(~m.img)&m.noteId.isin(orig.noteId)],k)
sf=lambda col: pd.to_datetime(np.array([(int(x)>>22)+1288834974657 for x in col]),unit='ms',utc=True)
m['post']=sf(m.tweetId)>=pd.Timestamp('2023-10-01',tz='UTC')
nm=m[m.y==1]; out['nnn']=nm.summary.str.contains(r'\bNNN\b').mean(); out['nm_post']=nm.post.mean()
for k in 'AE':
    for era in [False,True]:
        s=m[(m['p'+k].notna())&(m.post==era)]; out[f'ratio_{k}_{"post" if era else "pre"}']=dict(n=len(s),ratio=(s['p'+k]==0).mean()/(s.y==0).mean())
out['err']=err.error_category.value_counts().to_dict(); out['err_topic']=pd.crosstab(err.error_category,err.category).to_dict()
out['neg']=ex.extracted_claim.str.lower().str.contains(r"\b(?:not|never|cannot)\b|n't").mean()
def conv(o):
    if isinstance(o,dict): return {str(k):conv(v) for k,v in o.items()}
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,float)): return round(float(o),4)
    return o
json.dump(conv(out),open(HERE / 'numbers.json','w'),indent=1)
o=conv(out)
for k in S: print(k,'orig',{x:o['orig_'+k][x] for x in ['n','acc','kappa','P','R','F1','PM','RM']},'\n  exp',{x:o['exp_'+k][x] for x in ['n','acc','kappa','P','R','F1','PM','RM','base_mis','pred_false']},'mixed',o['mixed_'+k])
for kk in ['mc_AB_o','mc_AD_o','mc_AE_o','mc_BD_o','mc_AE_x','boot','power','google','img','nnn','nm_post','ratio_A_pre','ratio_A_post','ratio_E_pre','ratio_E_post','err','neg']: print(kk,o[kk])
for k in S:
    print(k,'topics exp kappa',[o[f'exp_{k}_{c}']['kappa'] for c in ['POLITICAL','HEALTH','SCIENTIFIC']],'orig kappa',[o[f'orig_{k}_{c}']['kappa'] for c in ['POLITICAL','HEALTH','SCIENTIFIC']],'noimg F1',o['noimg_'+k]['F1'],'kappa',o['noimg_'+k]['kappa'],'n',o['noimg_'+k]['n'])
print(o['err_topic'])
