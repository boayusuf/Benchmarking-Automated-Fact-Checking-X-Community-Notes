"""Figures 2 and 3 of the paper (kappa with bootstrap CIs; raw verdict distribution).
Run from anywhere: python benchmark/paper/make_figures.py. Outputs go to benchmark/paper/."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gmap import map_rating
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

OUT = str(HERE) + '/'
U = str(HERE.parent / 'data' / 'results') + '/'
m = pd.read_csv(U + 'master_results.csv', dtype=str)
has = m.google_verdict != 'NO_RESULT'
m.loc[has, 'google_verdict'] = m.loc[has, 'google_raw_rating'].fillna('').map(map_rating)
m.loc[~has, 'google_verdict'] = np.nan
S = {'A': 'gpt4o_verdict', 'B': 'tavily_gpt4o_verdict', 'C': 'google_verdict', 'D': 'filtered_rag_verdict', 'E': 'llama_verdict'}
NAMES = {'A': 'A: GPT-4o', 'B': 'B: GPT-4o + Tavily', 'C': 'C: Google FC', 'D': 'D: GPT-4o + filtered', 'E': 'E: Llama 3.3 70B'}
m['y'] = (m.cn_verdict == 'NOT_MISLEADING').astype(int)
for k, c in S.items():
    m['p' + k] = m[c].map({'TRUE': 1, 'FALSE': 0})
orig = m[m.tavily_gpt4o_verdict.notna() | m.filtered_rag_verdict.notna()]

plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42})

# ---------- Figure: kappa with bootstrap CIs ----------
rng = np.random.default_rng(42)
def boot_kappa(df, k, B=2000):
    s = df[df['p' + k].notna()]
    yt = s.y.values; yp = s['p' + k].astype(int).values
    pt = cohen_kappa_score(yt, yp)
    n = len(s); idx = rng.integers(0, n, (B, n))
    ks = np.array([cohen_kappa_score(yt[i], yp[i]) for i in idx])
    return pt, np.percentile(ks, 2.5), np.percentile(ks, 97.5), n
res = {}
for k in 'ABCDE':
    res[('orig', k)] = boot_kappa(orig, k)
for k in 'ACE':
    res[('exp', k)] = boot_kappa(m, k)
json.dump({f'{a}_{b}': [round(float(x), 3) for x in v[:3]] + [int(v[3])] for (a, b), v in res.items()},
          open(OUT + 'kappa_ci.json', 'w'), indent=1)
print({f'{a}_{b}': [round(float(x), 3) for x in v[:3]] + [int(v[3])] for (a, b), v in res.items()})

fig, axes = plt.subplots(2, 1, figsize=(3.05, 2.9), gridspec_kw={'height_ratios': [5, 3]})
for ax, (tag, keys, title) in zip(axes, [('orig', 'ABCDE', 'Original set (87% MIS)'), ('exp', 'ACE', 'Expanded set (20% MIS)')]):
    ys = np.arange(len(keys))[::-1]
    for yy, k in zip(ys, keys):
        pt, lo, hi, n = res[(tag, k)]
        ax.plot([lo, hi], [yy, yy], color='#444444', lw=1.0, solid_capstyle='butt')
        ax.plot(pt, yy, 'o', color='#1f4e79' if k in 'AE' else ('#b35806' if k in 'BD' else '#555555'), ms=4, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([NAMES[k] for k in keys])
    ax.axvline(0, color='#999999', lw=0.6, ls=':')
    ax.set_xlim(-0.12, 0.42)
    if tag=='exp': ax.set_xlabel("Cohen's $\\kappa$ (strict)")
    ax.set_title(title, fontsize=8, loc='left')
    ax.tick_params(axis='y', length=0)
plt.tight_layout(h_pad=1.0)
plt.savefig(OUT + 'fig_kappa.pdf'); plt.close()

# ---------- Figure: verdict distribution by system and CN label ----------
def dist(df, k):
    v = df[S[k]]
    out = {}
    out['TRUE'] = (v == 'TRUE').mean(); out['FALSE'] = (v == 'FALSE').mean(); out['MIXED'] = (v == 'MIXED').mean()
    out['UNCLEAR'] = (v == 'UNCLEAR').mean(); out['NONE'] = 1 - sum(out.values())
    return out
rows = []
for setname, df, keys in [('Original', orig, 'ABCDE'), ('Expanded', m, 'ACE')]:
    for lab in ['MISLEADING', 'NOT_MISLEADING']:
        for k in keys:
            d = dist(df[df.cn_verdict == lab], k); d.update(set=setname, label=lab, sys=k, n=int((df.cn_verdict == lab).sum())); rows.append(d)
D = pd.DataFrame(rows)
D.round(4).to_csv(OUT + 'verdict_dist.csv', index=False)
print(D.round(3).to_string())

cols = ['TRUE', 'FALSE', 'MIXED', 'UNCLEAR', 'NONE']
colors = {'TRUE': '#2b6ca3', 'FALSE': '#c0504d', 'MIXED': '#e8b84a', 'UNCLEAR': '#bbbbbb', 'NONE': '#f0f0f0'}
keys='ABCDE'
groups=[]  # (xpos, system, setname, label, sublabel)
x=0.0
for k in keys:
    bars=[('Original','MISLEADING','MIS')]
    bars.append(('Original','NOT_MISLEADING','NM$_o$'))
    if k in 'ACE': bars.append(('Expanded','NOT_MISLEADING','NM$_x$'))
    for b in bars:
        groups.append((x,k)+b); x+=1
    x+=0.7
fig, ax = plt.subplots(figsize=(5.5, 2.3))
for (xp,k,setname,lab,sub) in groups:
    d=D[(D.set==setname)&(D.label==lab)&(D.sys==k)].iloc[0]
    bottom=0
    for c in cols:
        ax.bar(xp, d[c], 0.8, bottom=bottom, color=colors[c], edgecolor='white', linewidth=0.4, label=c if xp==0 else None)
        bottom+=d[c]
    ax.text(xp, -0.04, sub, ha='center', va='top', fontsize=6)
centers={}
for (xp,k,*_ ) in groups: centers.setdefault(k,[]).append(xp)
for k,v in centers.items():
    ax.text(np.mean(v), -0.17, NAMES[k].replace(': ','\n'), ha='center', va='top', fontsize=7)
ax.set_xticks([]); ax.set_xlim(-0.6, x-0.7+0.6)
ax.set_ylim(0,1); ax.set_yticks([0,.25,.5,.75,1]); ax.set_yticklabels(['0','25','50','75','100'])
ax.set_ylabel('Share of claims (%)')
ax.spines['bottom'].set_visible(False)
ax.legend(ncol=5, fontsize=6.5, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0)
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig(OUT + 'fig_verdicts.pdf'); plt.close()
