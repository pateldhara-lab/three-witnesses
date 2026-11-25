"""Sensitivity checks for Report 2: lag ranking shares, Jan+Feb merge, matched scope, seasonally adjusted levels.
Run from the repo root:  python scripts/run_sensitivity.py [--table data/side_by_side.csv] [--brazil-exporter data/brazil_exporter.csv]
"""
import sys, argparse, pathlib, numpy as np, pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
ap=argparse.ArgumentParser(); ap.add_argument("--table",default=str(ROOT/"data/side_by_side.csv")); ap.add_argument("--brazil-exporter",default=str(ROOT/"data/brazil_exporter.csv")); A=ap.parse_args()
from table_io import load_table, TC_NAMES
from triple_collocation import triple_collocation, yoy_changes
import study
from study import bootstrap_ranges
df=load_table(A.table)
cal=pd.date_range("2019-01-01","2024-12-01",freq="MS")
def aligned(L,lc,le):
    out=0
    ch=L.china_raw_t.copy(); ch.index=ch.index-pd.DateOffset(months=lc)
    eu=L.eu_raw_t.copy(); eu.index=eu.index-pd.DateOffset(months=le)
    return ch.reindex(L.index)+eu.reindex(L.index)
def build(c,lc=None,le=None,jan_feb=False,exp_override=None):
    L=df[df.country==c].set_index("month").sort_index()
    imp=L.importer_aligned_total_t if lc is None else aligned(L,lc,le)
    exp=L.exporter_t if exp_override is None else exp_override
    X=pd.DataFrame({"e":exp,"i":imp,"s":L.ship_t}).reindex(cal)
    flagged=L.flag.reindex(cal).fillna("").str.contains("missing|known_source_error")
    X.loc[flagged.values,:]=np.nan
    if jan_feb:
        for y in range(2019,2025):
            j=pd.Timestamp(y,1,1); f=pd.Timestamp(y,2,1)
            X.loc[f]=X.loc[j]+X.loc[f]; X.loc[j]=np.nan
    Y=yoy_changes(X.to_numpy(float),12); k=np.isfinite(Y).all(1); return Y[k]
def summ(Y,seed=20261124):
    r=triple_collocation(Y,TC_NAMES); lo,hi,sn,_=bootstrap_ranges(Y,500,6,np.random.default_rng(seed))
    return r,lo,hi,sn
def rank_share(Y,seed=20261124,n=500,bl=6):
    rng=np.random.default_rng(seed); idx=study.block_bootstrap_indices(Y.shape[0],bl,n,rng)
    noise,signal=study.tc_batch(study.cov_batch(Y[idx]))
    ok=np.isfinite(signal)&(signal>0)&np.isfinite(noise)
    rel=np.sqrt(np.clip(noise,0,None)/np.where(signal>0,signal,np.nan))
    g=ok[:,1]&ok[:,2]; a=(rel[:,2]<rel[:,1])[g]; return a.mean(), g.mean()
print("== Brazil lag rank share (ships cleaner than importers)")
for lc,le in [(1,2),(2,3),(3,4)]:
    Y=build("BRA",lc,le); r,lo,hi,sn=summ(Y); print((lc,le),len(Y),np.round(r.rel_noise,2),"rank share",np.round(rank_share(Y),3))
print("== all countries table lags rank share ships<importers")
for c in ["AUS","BRA","LBR","PER","ZAF"]:
    Y=build(c); print(c,np.round(rank_share(Y),3))
print("== Jan+Feb merged (all three records)")
for c in ["AUS","BRA","LBR","PER","ZAF"]:
    Y=build(c,jan_feb=True); r,lo,hi,sn=summ(Y); print(c,len(Y),np.round(r.rel_noise,2),"lo",np.round(lo,2),"hi",np.round(hi,2),r.reasons)
print("== Brazil matched scope (exporter = China + EU27 destinations only)")
b=pd.read_csv(A.brazil_exporter,parse_dates=["month"])
eu27="AUT BEL BGR HRV CYP CZE DNK EST FIN FRA DEU GRC HUN IRL ITA LVA LTU LUX MLT NLD POL PRT ROU SVK SVN ESP SWE".split()
sub=b[b.counterparty.isin(eu27+["CHN"])].groupby("month").tonnes.sum()
L=df[df.country=="BRA"].set_index("month").sort_index()
print("share of brazil exporter that goes to CHN+EU27:", round(sub.reindex(cal).sum()/L.exporter_t.reindex(cal).sum(),3))
Y=build("BRA",exp_override=sub.reindex(L.index)); r,lo,hi,sn=summ(Y); print(len(Y),np.round(r.rel_noise,2),"lo",np.round(lo,2),"hi",np.round(hi,2),"undef",np.round(sn,2))
print("rank share matched",np.round(rank_share(Y),3))
print("== Brazil EU lag only (china fixed 2)")
for le in range(0,5):
    Y=build("BRA",2,le); r=triple_collocation(Y,TC_NAMES); print(le,np.round(r.rel_noise,2))
print("== power A4 etc from scenario file: 12.8%")
print("== strict rank share: ships defined (noise>=0) AND ships<importers, among all resamples; and among usable-defined")
def strict(Y,seed=20261124,n=500,bl=6):
    rng=np.random.default_rng(seed); idx=study.block_bootstrap_indices(Y.shape[0],bl,n,rng)
    noise,signal=study.tc_batch(study.cov_batch(Y[idx]))
    ok=np.isfinite(signal)&(signal>0)&np.isfinite(noise)
    rel=np.sqrt(np.where(noise>=0,noise,np.nan)/np.where(signal>0,signal,np.nan))
    relc=np.sqrt(np.clip(noise,0,None)/np.where(signal>0,signal,np.nan))
    g=ok[:,1]&ok[:,2]
    clipped=(relc[:,2]<relc[:,1])[g].mean()
    # alternative: compare noise variance relative (importer vs ship) among resamples where both defined
    d=g&(noise[:,2]>=0)&(noise[:,1]>=0)
    both=(rel[:,2]<rel[:,1])[d].mean() if d.any() else np.nan
    allr=((rel[:,2]<rel[:,1])&g&d).sum()/n
    return round(clipped,3),round(both,3),round(d.mean(),3),round(allr,3)
for lc,le in [(1,2),(2,3),(3,4)]:
    print((lc,le),strict(build("BRA",lc,le)))

print("== Seasonally adjusted levels (month-of-year means removed)")
from table_io import TC_COLUMNS
for c in ["AUS","BRA","LBR","PER","ZAF"]:
    L=df[df.country==c].set_index("month").reindex(cal); X=L[list(TC_COLUMNS)].copy()
    fl=L.flag.fillna("").str.contains("missing|known_source_error"); X.loc[fl.values,:]=np.nan; X=X.dropna()
    Z=X-X.groupby(X.index.month).transform("mean"); r=triple_collocation(Z.to_numpy(float),TC_NAMES)
    print(c,len(Z),np.round(r.rel_noise,2),r.reasons)
