"""Notebook 16 frozen cascade. Run: uvicorn notebook16_service:app --host 127.0.0.1 --port 8016 --workers 1.
Place beside notebook15_service.py in project root. Uses cached candidates/features; actual LGBM and DeepFM inference.
"""
from __future__ import annotations
import json,time
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI,HTTPException,Query
from notebook15_service import Engine as BaseEngine, mmr_one_user, CATS, ROOT, REPORT, SPLIT

CONFIG=REPORT/'notebook16_cascade_config.json'
class CascadeEngine(BaseEngine):
    def __init__(self):
        if not CONFIG.exists(): raise FileNotFoundError(f'Run Notebook16 validation selection first: {CONFIG}')
        cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
        if cfg.get('selected_on')!='validation' or cfg.get('mmr_lambda')!=0.9:
            raise ValueError('Cascade config must be frozen on validation with MMR lambda 0.9')
        self.keep=int(cfg['selected_keep'])
        if self.keep not in (100,150,200,300,500): raise ValueError('Invalid selected_keep')
        super().__init__()
        # Reference from Notebook13 remains the full-500 DeepFM+MMR, NOT expected exact parity after pruning.

    def recommend(self,uid:int,k:int=20):
        if uid not in self.feature_cache: raise KeyError(uid)
        with self.lock:
            self.sync();start=time.perf_counter();d=self.feature_cache[uid];tm={}
            t=time.perf_counter();xb=self.prep_lgbm(d);tm['lgbm_preprocess']=(time.perf_counter()-t)*1000
            t=time.perf_counter();ls=self.booster.predict(xb,num_threads=1);tm['lgbm_inference']=(time.perf_counter()-t)*1000
            # Stable, deterministic tie break: score desc, original retrieval rank, item id.
            t=time.perf_counter();order=np.lexsort((d.item_id.to_numpy(),d.retrieval_rank.to_numpy(),-np.asarray(ls)))[:self.keep]
            pruned=d.iloc[order].copy();tm['candidate_pruning']=(time.perf_counter()-t)*1000
            t=time.perf_counter();xn,xc=self.prep_deepfm(pruned);tm['deepfm_preprocess']=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            with torch.inference_mode():
                scores=self.model(torch.from_numpy(xn).to(self.device),torch.from_numpy(xc).to(self.device)).detach().cpu().numpy().astype(np.float64)
            self.sync();tm['deepfm_inference']=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            dr=pruned.retrieval_rank.to_numpy();items=pruned.item_id.to_numpy(dtype=np.int64)
            rank=np.lexsort((items,dr,-scores));sc=scores[rank]
            cat=pruned.iloc[rank][CATS].apply(pd.to_numeric,errors='coerce').fillna(-1).to_numpy(dtype=np.int64)
            # Notebook15 MMR assumes 500 items; use the identical algorithm on a padded candidate pool.
            # Padding is never selected: the top min(100,keep) is computed using a generalized version below.
            local=mmr_general(sc,cat,self.lam,pool=min(150,self.keep),depth=min(100,self.keep))
            ranked=items[rank[local]]
            tm['mmr_greedy']=(time.perf_counter()-t)*1000
            return {'user_id':uid,'k':k,'candidate_count':500,'prerank_keep':self.keep,
                    'fine_rank_count':len(pruned),'mmr_lambda':self.lam,
                    'items':ranked[:k].tolist(),
                    'stage_ms':{a:round(b,4) for a,b in tm.items()},
                    'total_ms':round((time.perf_counter()-start)*1000,4)}

def mmr_general(scores,cat,lam,pool=150,depth=100):
    n=len(scores)
    if n==500: return mmr_one_user(scores,cat,lam,pool,depth)
    p=min(pool,n);d=min(depth,p)
    sc=np.asarray(scores[:p],dtype=np.float64);span=sc.max()-sc.min()
    rel=((sc-sc.min())/span if span>1e-12 else np.ones(p)).astype(np.float32)
    remaining=np.ones(p,dtype=bool);maxsim=np.zeros(p,dtype=np.float32);chosen=[]
    for _ in range(d):
        objective=np.float32(lam)*rel-np.float32(1-lam)*maxsim
        objective[~remaining]=-np.inf
        j=int(np.argmax(objective));chosen.append(j);remaining[j]=False
        for col,s in ((2,.4),(1,.7),(0,1.)):
            if cat[j,col]>=0:
                maxsim=np.maximum(maxsim,np.where(cat[:p,col]==cat[j,col],np.float32(s),0.))
    return np.r_[np.asarray(chosen,dtype=np.int32),np.flatnonzero(remaining),np.arange(p,n)].astype(np.int32)

ENGINE=None
@asynccontextmanager
async def lifespan(app):
    global ENGINE
    ENGINE=CascadeEngine()
    yield
    ENGINE=None
app=FastAPI(title='Notebook 16 True Cascade Demo',version='0.1.0',lifespan=lifespan)
@app.get('/health')
def health():
    if ENGINE is None:raise HTTPException(503,'Not ready')
    return {'status':'ok','device':str(ENGINE.device),'users_cached':len(ENGINE.users),
            'prerank_keep':ENGINE.keep,'mmr_lambda':ENGINE.lam,'real_model_inference':True,
            'retrieval':'precomputed','features':'precomputed'}
@app.get('/recommend')
def recommend(user_id:int=Query(...,ge=0),k:int=Query(20,ge=1,le=100)):
    if ENGINE is None:raise HTTPException(503,'Not ready')
    try:return ENGINE.recommend(user_id,k)
    except KeyError:raise HTTPException(404,'Unknown cached user')
@app.get('/reference/{user_id}')
def reference(user_id:int,k:int=Query(20,ge=1,le=100)):
    if ENGINE is None:raise HTTPException(503,'Not ready')
    if user_id not in ENGINE.reference:raise HTTPException(404,'Unknown user')
    return {'user_id':user_id,'items':ENGINE.reference[user_id][:k].tolist()}
