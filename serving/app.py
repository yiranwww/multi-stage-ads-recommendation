"""Offline RetailRocket ranking-serving demo; not a live ad or CTR prediction system."""
from pathlib import Path
import os, pickle, time
from functools import lru_cache
from threading import Lock
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from fastapi import FastAPI, HTTPException, Query

ROOT=Path(os.environ.get('RECSYS_ROOT', Path(__file__).resolve().parent.parent)).resolve()
SERVING=ROOT/'serving'; REPORTS=ROOT/'reports'; MODELS=ROOT/'models'
TOP_N=20; FINAL_K=10; MMR_LAMBDA=.9

class NumericDeepFM(nn.Module):
    def __init__(self,n_fields,embed_dim=8,hidden=(64,32),dropout=.2):
        super().__init__();self.linear=nn.Linear(n_fields,1)
        self.fm_vectors=nn.Parameter(torch.randn(n_fields,embed_dim)*.02)
        layers=[];d=n_fields*embed_dim
        for width in hidden:
            layers.extend([nn.Linear(d,width),nn.ReLU(),nn.Dropout(dropout)]);d=width
        layers.append(nn.Linear(d,1));self.mlp=nn.Sequential(*layers)
    def forward(self,x):
        e=x.unsqueeze(-1)*self.fm_vectors.unsqueeze(0);s=e.sum(dim=1)
        fm=.5*(s.square()-(e.square()).sum(dim=1)).sum(dim=1,keepdim=True)
        return (self.linear(x)+fm+self.mlp(e.flatten(start_dim=1))).squeeze(-1)

def mmr_order(items,scores,wv,lam=.9,k=10):
    items=np.asarray(items,dtype=np.int64);scores=np.asarray(scores,dtype=np.float64)
    lo,hi=float(scores.min()),float(scores.max())
    rel=(scores-lo)/(hi-lo) if hi>lo else np.zeros(len(scores),dtype=np.float64)
    vectors=np.zeros((len(items),wv.vector_size),dtype=np.float32)
    for j,item in enumerate(items):
        key=str(int(item))
        if key in wv: vectors[j]=wv[key]
    norms=np.linalg.norm(vectors,axis=1,keepdims=True)
    np.divide(vectors,np.maximum(norms,1e-12),out=vectors)
    sim=np.clip(vectors@vectors.T,0,1);np.fill_diagonal(sim,0)
    remaining=np.ones(len(items),dtype=bool);redundancy=np.zeros(len(items));chosen=[]
    for _ in range(k):
        scores_mmr=lam*rel-(1-lam)*redundancy
        scores_mmr[~remaining]=-np.inf
        j=int(np.argmax(scores_mmr));chosen.append(j);remaining[j]=False
        redundancy=np.maximum(redundancy,sim[:,j])
    return np.asarray(chosen,dtype=np.int64)

class RankingService:
    def __init__(self):
        with open(REPORTS/'notebook09_lightgbm_preranker.pkl','rb') as f:
            pkg=pickle.load(f)
        self.lgb=pkg['model'];self.features=list(pkg['features'])
        try: p=torch.load(REPORTS/'notebook10_deepfm_model.pt',map_location='cpu',weights_only=True)
        except TypeError: p=torch.load(REPORTS/'notebook10_deepfm_model.pt',map_location='cpu')
        assert self.features==list(p['features']), 'Feature ordering mismatch'
        self.deepfm=NumericDeepFM(p['n_fields'],p['embed_dim'],p['hidden'],p['dropout'])
        self.deepfm.load_state_dict(p['state_dict']);self.deepfm.eval()
        self.scaler=StandardScaler()
        self.scaler.mean_=np.asarray(p['scaler_mean'],dtype=np.float64)
        self.scaler.scale_=np.asarray(p['scaler_scale'],dtype=np.float64)
        self.scaler.var_=self.scaler.scale_**2
        self.scaler.n_features_in_=len(self.features)
        self.wv=Word2Vec.load(str(MODELS/'notebook06_test_item2vec.model')).wv
        frame=pd.read_csv(SERVING/'demo_candidates.csv')
        assert frame.groupby('visitorid').size().eq(100).all()
        assert not frame.duplicated(['visitorid','itemid']).any()
        self.candidates={int(u):g.sort_values(['rank','itemid'],kind='mergesort').reset_index(drop=True)
                         for u,g in frame.groupby('visitorid',sort=False)}
        self.lock=Lock() # avoid cross-request model CPU thread contention in this demo
    def recommend(self,visitorid:int,k:int=10,mmr:bool=True):
        if visitorid not in self.candidates: raise KeyError(visitorid)
        g=self.candidates[visitorid]
        x=g[self.features].astype('float32') # preserve feature names for LightGBM
        a=time.perf_counter_ns()
        with self.lock:
            ls=self.lgb.predict_proba(x)[:,1]
            b=time.perf_counter_ns()
            # Match Notebook 09: descending score, ascending retrieval rank, then item ID
            gate=np.lexsort((g.itemid.to_numpy(),g['rank'].to_numpy(),-ls))[:TOP_N]
            c=time.perf_counter_ns()
            z=np.clip(self.scaler.transform(x.iloc[gate].to_numpy(dtype=np.float32)),-8,8).astype('float32')
            with torch.inference_mode(): ds=self.deepfm(torch.from_numpy(z)).numpy()
            d=time.perf_counter_ns()
            gated_items=g.itemid.to_numpy()[gate]
            # Match Notebook 13: deepfm_score descending, lgb_rank ascending, item ID
            ordered=np.lexsort((gated_items,np.arange(1,TOP_N+1),-ds))
            ordered_items=gated_items[ordered];ordered_scores=ds[ordered]
            if mmr: idx=mmr_order(ordered_items,ordered_scores,self.wv,MMR_LAMBDA,k)
            else: idx=np.arange(k)
            e=time.perf_counter_ns()
        recs=[{'rank':i+1,'itemid':int(ordered_items[j]),'ranking_score':float(ordered_scores[j])}
              for i,j in enumerate(idx)]
        return {'visitorid':visitorid,'k':k,'mmr':mmr,'snapshot_cutoff':'2015-09-01',
                'candidate_pool':100,'deepfm_candidates':TOP_N,'recommendations':recs,
                'latency_ms':{'lightgbm':(b-a)/1e6,'gate':(c-b)/1e6,
                              'deepfm_and_scaler':(d-c)/1e6,'sort_and_mmr':(e-d)/1e6,
                              'ranking_total':(e-a)/1e6},
                'limitations':'Precomputed candidates/features; ranking scores are not calibrated CTR.'}

@lru_cache(maxsize=1)
def get_service(): return RankingService()

app=FastAPI(title='RetailRocket Multi-Stage Ranking Demo',version='1.0.0',
            description='Frozen-snapshot recommendation proxy; not an online ads auction system.')

@app.get('/health')
def health():
    s=get_service()
    return {'status':'ok','loaded_users':len(s.candidates),'candidate_pool':100,'deepfm_gate':TOP_N,
            'snapshot_cutoff':'2015-09-01'}

@app.get('/demo-users')
def demo_users(limit:int=Query(default=10,ge=1,le=100)):
    return {'visitorids':sorted(get_service().candidates)[:limit]}

@app.get('/recommend/{visitorid}')
def recommend(visitorid:int,k:int=Query(default=10,ge=1,le=10),mmr:bool=True):
    try: return get_service().recommend(visitorid,k=k,mmr=mmr)
    except KeyError: raise HTTPException(status_code=404,detail='User not in frozen demo cohort')
