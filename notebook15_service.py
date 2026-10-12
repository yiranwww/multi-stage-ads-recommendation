"""Notebook 15 local demo service: precomputed test candidates/features; real LightGBM and DeepFM inference.
Run from project root: uvicorn notebook15_service:app --host 127.0.0.1 --port 8015 --workers 1
Requires NOTEBOOK15_PROJECT_ROOT environment variable or cwd project root.
"""
from __future__ import annotations
import os, json, time, threading
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get('NOTEBOOK15_PROJECT_ROOT', Path.cwd())).resolve()
OUT = ROOT/'data'/'processed'
REPORT = ROOT/'data'/'reports'
MODELS = ROOT/'models'
SPLIT = os.environ.get('NOTEBOOK15_SPLIT','test')
TOP_K_DEFAULT=20
MMR_POOL=150
MMR_DEPTH=100
CATS=['category_id','parent_category_id','root_category_id']

class DeepFM(torch.nn.Module):
    def __init__(self,n_num,cards,dim=8,hidden=64,dropout=.35):
        super().__init__();self.dim=dim;self.n_num=n_num
        self.num_linear=torch.nn.Parameter(torch.zeros(n_num));self.num_fm=torch.nn.Parameter(torch.randn(n_num,dim)*.01)
        self.cat_linear=torch.nn.ModuleList([torch.nn.Embedding(c,1,padding_idx=0) for c in cards])
        self.cat_fm=torch.nn.ModuleList([torch.nn.Embedding(c,dim,padding_idx=0) for c in cards])
        self.bias=torch.nn.Parameter(torch.zeros(()))
        self.deep=torch.nn.Sequential(torch.nn.Linear(n_num+len(cards)*dim,hidden),torch.nn.ReLU(),torch.nn.Dropout(dropout),torch.nn.Linear(hidden,hidden//2),torch.nn.ReLU(),torch.nn.Dropout(dropout),torch.nn.Linear(hidden//2,1))
    def forward(self,x,cat):
        linear=(x*self.num_linear).sum(1)+self.bias;embeddings=[x.unsqueeze(-1)*self.num_fm.unsqueeze(0)];deep_c=[]
        for j,(e1,e2) in enumerate(zip(self.cat_linear,self.cat_fm)):
            linear=linear+e1(cat[:,j]).squeeze(-1);z=e2(cat[:,j]);embeddings.append(z.unsqueeze(1));deep_c.append(z)
        z=torch.cat(embeddings,dim=1);fm=.5*((z.sum(dim=1)**2-z.square().sum(dim=1)).sum(dim=1))
        return linear+fm+self.deep(torch.cat([x]+deep_c,dim=1)).squeeze(-1)

def similarity_to_one(cat,selected):
    result=np.zeros(len(cat),dtype=np.float32)
    for col,sim in ((2,.4),(1,.7),(0,1.)):
        if selected[col]>=0:result=np.where(cat[:,col]==selected[col],np.float32(sim),result)
    return result

def mmr_one_user(scores,cat,lam,pool=150,depth=100):
    n=len(scores);assert n==500
    if lam>=1:return np.arange(n,dtype=np.int32)
    p=min(pool,n);d=min(depth,p);sc=np.asarray(scores[:p],dtype=np.float64);span=sc.max()-sc.min()
    rel=((sc-sc.min())/span if span>1e-12 else np.ones(p)).astype(np.float32)
    maxsim=np.zeros(p,dtype=np.float32);remaining=np.ones(p,dtype=bool);chosen=np.empty(d,dtype=np.int32)
    for t in range(d):
        obj=np.float32(lam)*rel-np.float32(1-lam)*maxsim;obj[~remaining]=-np.inf
        j=int(np.argmax(obj));chosen[t]=j;remaining[j]=False
        maxsim=np.maximum(maxsim,similarity_to_one(cat[:p],cat[j]))
    return np.concatenate([chosen,np.flatnonzero(remaining).astype(np.int32),np.arange(p,n,dtype=np.int32)])

class RecommendResponse(BaseModel):
    user_id:int
    k:int
    candidate_count:int
    mmr_lambda:float
    items:list[int]
    stage_ms:dict[str,float]
    total_ms:float

class Engine:
    def __init__(self):
        required=[OUT/f'{SPLIT}_fusion_rrf_top500.parquet',OUT/f'{SPLIT}_ranking_features_top500.parquet',
                  OUT/f'{SPLIT}_mmr_top500.parquet',MODELS/'notebook10_lightgbm_binary.txt',
                  MODELS/'notebook11_deepfm_checkpoint.pt',REPORT/'notebook13_mmr_report.json']
        missing=[str(p) for p in required if not p.exists()]
        if missing:raise FileNotFoundError('Missing upstream artifacts: '+', '.join(missing))
        self.lam=float(json.loads((REPORT/'notebook13_mmr_report.json').read_text(encoding='utf-8'))['mmr']['selected_lambda'])
        if abs(self.lam-.9)>1e-9:raise ValueError('Expected frozen MMR lambda=0.9, got '+str(self.lam))
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.booster=lgb.Booster(model_file=str(MODELS/'notebook10_lightgbm_binary.txt'))
        try: ckpt=torch.load(MODELS/'notebook11_deepfm_checkpoint.pt',map_location='cpu',weights_only=True)
        except TypeError:raise RuntimeError('PyTorch weights_only=True unavailable; upgrade PyTorch')
        self.nums=list(ckpt['num_features']);self.cats=list(ckpt['categorical_features'])
        self.model=DeepFM(len(self.nums),list(ckpt['category_cardinalities']),int(ckpt['embedding_dim']),int(ckpt['hidden_dim']),float(ckpt['dropout']))
        self.model.load_state_dict(ckpt['state_dict'],strict=True);self.model.to(self.device).eval()
        self.med=np.asarray(ckpt['numeric_median'],dtype=np.float64);self.logmask=np.asarray(ckpt['numeric_log_mask'],dtype=bool)
        self.lo=np.asarray(ckpt['numeric_clip_low'],dtype=np.float64);self.hi=np.asarray(ckpt['numeric_clip_high'],dtype=np.float64)
        self.mu=np.asarray(ckpt['numeric_mean'],dtype=np.float64);self.sigma=np.asarray(ckpt['numeric_std'],dtype=np.float64)
        self.vocabs=ckpt['category_vocabularies'];self.cards=list(ckpt['category_cardinalities'])
        self.features=list(self.booster.feature_name());self.lgbcats=[c for c in self.features if c in self.cats]
        assert len(self.booster.pandas_categorical or [])==len(self.lgbcats)
        self.levels=dict(zip(self.lgbcats,self.booster.pandas_categorical))
        # Optimized LGBM: numeric columns as one contiguous float32 block; categorical columns converted only when needed.
        self.numeric_lgb=[c for c in self.features if c not in self.lgbcats]
        if len(self.features)!=len(self.nums)+len(self.cats):raise ValueError('Feature schema mismatch')
        r=pd.read_parquet(OUT/f'{SPLIT}_fusion_rrf_top500.parquet',columns=['user_id','item_id','retrieval_rank'])
        f=pd.read_parquet(OUT/f'{SPLIT}_ranking_features_top500.parquet')
        self.users={}
        for uid,g in r.groupby('user_id',sort=False):
            self.users[int(uid)]=g.sort_values('retrieval_rank',kind='stable').item_id.to_numpy(dtype=np.int64)
        # Align each feature group once at startup; request path does not merge DataFrames.
        self.feature_cache={}
        for uid,g in f.groupby('user_id',sort=False):
            uid=int(uid)
            if uid in self.users:
                aligned=g.set_index('item_id').loc[self.users[uid]].reset_index()
                if len(aligned)!=500 or aligned.item_id.nunique()!=500:raise ValueError('Bad candidates '+str(uid))
                self.feature_cache[uid]=aligned
        if set(self.users)!=set(self.feature_cache):raise ValueError('Candidate/feature user mismatch')
        self.reference={}
        mmr=pd.read_parquet(OUT/f'{SPLIT}_mmr_top500.parquet',columns=['user_id','item_id','mmr_rank'])
        for uid,g in mmr.groupby('user_id',sort=False):
            self.reference[int(uid)]=g.sort_values('mmr_rank').item_id.to_numpy(dtype=np.int64)
        self.lock=threading.Lock() # serialize GPU/LightGBM inference; no concurrency scaling claims
        self.loaded_at=time.time()

    def sync(self):
        if self.device.type=='cuda':torch.cuda.synchronize()

    def prep_lgbm(self,df):
        # Single vectorized conversion instead of Notebook14's per-numeric-column astype loop.
        x=pd.DataFrame(df[self.numeric_lgb].to_numpy(dtype=np.float32,copy=True),columns=self.numeric_lgb,index=df.index)
        for c in self.lgbcats:
            v=pd.to_numeric(df[c],errors='coerce').fillna(-1).astype('int64')
            x[c]=pd.Categorical(v,categories=self.levels[c])
        return x[self.features]

    def prep_deepfm(self,df):
        x=df[self.nums].to_numpy(dtype=np.float64,copy=True)
        x=np.where(np.isfinite(x),x,self.med)
        x[:,self.logmask]=np.log1p(np.maximum(x[:,self.logmask],0))
        x=np.clip(x,self.lo,self.hi);x=((x-self.mu)/self.sigma).astype('float32');x=np.clip(x,-8,8)
        cat=np.stack([pd.to_numeric(df[c],errors='coerce').fillna(-1).astype('int64').map(self.vocabs[c]).fillna(0).to_numpy(dtype=np.int64) for c in self.cats],axis=1)
        for j in range(len(self.cats)):
            if cat[:,j].min()<0 or cat[:,j].max()>=self.cards[j]:raise ValueError('Category OOB '+self.cats[j])
        return x,cat

    def recommend(self,uid:int,k:int=20):
        if uid not in self.feature_cache:raise KeyError(uid)
        with self.lock:
            self.sync();start=time.perf_counter();d=self.feature_cache[uid];timings={}
            t=time.perf_counter();xb=self.prep_lgbm(d);timings['lgbm_preprocess']=(time.perf_counter()-t)*1000
            t=time.perf_counter();_lgb_score=self.booster.predict(xb,num_threads=1);timings['lgbm_inference']=(time.perf_counter()-t)*1000
            t=time.perf_counter();xn,xc=self.prep_deepfm(d);timings['deepfm_preprocess']=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            with torch.inference_mode():
                score=self.model(torch.from_numpy(xn).to(self.device),torch.from_numpy(xc).to(self.device)).detach().cpu().numpy().astype(np.float64)
            self.sync();timings['deepfm_inference']=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            order=np.lexsort((d.item_id.to_numpy(),d.retrieval_rank.to_numpy(),-score))
            sorted_scores=score[order]
            cat=d.iloc[order][CATS].apply(pd.to_numeric,errors='coerce').fillna(-1).to_numpy(dtype=np.int64)
            local=mmr_one_user(sorted_scores,cat,self.lam,MMR_POOL,MMR_DEPTH)
            ranked=d.item_id.to_numpy(dtype=np.int64)[order[local]]
            timings['mmr_greedy']=(time.perf_counter()-t)*1000
            total=(time.perf_counter()-start)*1000
            return {'user_id':uid,'k':k,'candidate_count':len(d),'mmr_lambda':self.lam,
                    'items':ranked[:k].tolist(),'stage_ms':{key:round(val,4) for key,val in timings.items()},'total_ms':round(total,4)}

ENGINE=None
@asynccontextmanager
async def lifespan(app:FastAPI):
    global ENGINE
    ENGINE=Engine()
    yield
    ENGINE=None

app=FastAPI(title='Notebook 15 Recommendation Demo',version='0.1.0',lifespan=lifespan)

@app.get('/health')
def health():
    if ENGINE is None:raise HTTPException(503,'Models not loaded')
    return {'status':'ok','device':str(ENGINE.device),'users_cached':len(ENGINE.users),
            'mmr_lambda':ENGINE.lam,'retrieval':'precomputed','features':'precomputed','real_model_inference':True}

@app.get('/recommend',response_model=RecommendResponse)
def recommend(user_id:int=Query(...,ge=0),k:int=Query(20,ge=1,le=100)):
    if ENGINE is None:raise HTTPException(503,'Models not loaded')
    try:return ENGINE.recommend(user_id,k)
    except KeyError:raise HTTPException(404,'User not present in offline replay cache')

@app.get('/reference/{user_id}')
def reference(user_id:int,k:int=Query(20,ge=1,le=100)):
    # Local demo audit only; disable this endpoint in production.
    if ENGINE is None:raise HTTPException(503,'Models not loaded')
    if user_id not in ENGINE.reference:raise HTTPException(404,'Unknown user')
    return {'user_id':user_id,'items':ENGINE.reference[user_id][:k].tolist()}
