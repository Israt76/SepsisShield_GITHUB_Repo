import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, lightgbm as lgb, time
from sklearn.isotonic import IsotonicRegression
import train
from features import feature_columns
from metrics import best_threshold, hourly
df = pd.read_parquet(train.FEAT); cols = feature_columns(df)
tr,va,te = train.split_patients(df)
mtr=df.patient_id.isin(tr).to_numpy(); mva=df.patient_id.isin(va).to_numpy()
y=df.SepsisLabel.to_numpy()
grid=np.concatenate([np.linspace(0.005,0.1,39),np.linspace(0.11,0.5,40)])
variants={
 'base':{},
 'slow_small':dict(learning_rate=0.02,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.5,metric='auc'),
 'mid_auc':dict(learning_rate=0.03,num_leaves=63,min_data_in_leaf=300,feature_fraction=0.5,lambda_l2=5.0,metric='auc'),
}
for k,v in variants.items():
    t=time.time(); p=dict(train.PARAMS,seed=11,**v)
    m=lgb.train(p,lgb.Dataset(df.loc[mtr,cols],y[mtr]),3000,valid_sets=[lgb.Dataset(df.loc[mva,cols],y[mva])],callbacks=[lgb.early_stopping(150,verbose=False)])
    pv=m.predict(df.loc[mva,cols],num_iteration=m.best_iteration)
    iso=IsotonicRegression(out_of_bounds='clip').fit(pv,y[mva]); thr,u=best_threshold(df.loc[mva],iso.predict(pv),grid)
    h=hourly(y[mva],pv); print(k,'iters',m.best_iteration,'valAUROC %.4f AUPRC %.4f util %.4f thr %.3f  %ds'%(h['auroc'],h['auprc'],u,thr,time.time()-t),flush=True)
