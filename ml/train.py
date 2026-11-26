"""Reproducible next-day repair-rate prediction on SYNTHETIC solar module data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
raw = pd.read_csv(ROOT/'data/fact_module_quality.csv', parse_dates=['production_date'])
assert raw['is_synthetic'].astype(str).str.lower().eq('true').all()
assert not raw['module_id'].duplicated().any(), 'Duplicate module_id detected'
assert raw['has_repair'].notna().all()
daily = (raw.groupby(['line_id','production_date'], as_index=False)
         .agg(modules=('module_id','size'), repairs=('has_repair','sum')))
daily['repair_rate_pct'] = 100*daily['repairs']/daily['modules']
daily = daily.sort_values(['line_id','production_date']).reset_index(drop=True)
g = daily.groupby('line_id',group_keys=False)
# At end of day t, all predictors below are observable. Target is t+1.
daily['rate_lag_1'] = g['repair_rate_pct'].shift(1)
daily['rate_rolling_3'] = g['repair_rate_pct'].transform(lambda s:s.rolling(3,min_periods=3).mean())
daily['rate_rolling_7'] = g['repair_rate_pct'].transform(lambda s:s.rolling(7,min_periods=7).mean())
daily['modules_rolling_7'] = g['modules'].transform(lambda s:s.rolling(7,min_periods=7).mean())
daily['target_next_day'] = g['repair_rate_pct'].shift(-1)
daily['prediction_date'] = daily['production_date']+pd.Timedelta(days=1)
daily['day_of_week_next'] = daily['prediction_date'].dt.dayofweek
# Confirm that shift(-1) really means tomorrow, not next available record.
daily['observed_next_date'] = g['production_date'].shift(-1)
daily.loc[daily['observed_next_date'] != daily['prediction_date'],'target_next_day'] = np.nan
cols=['rate_lag_1','rate_rolling_3','rate_rolling_7','modules_rolling_7','target_next_day']
frame=daily.dropna(subset=cols).copy()
dates=sorted(frame['prediction_date'].unique())
cutoff=dates[int(len(dates)*0.8)]
train=frame.loc[frame.prediction_date<cutoff].copy()
test=frame.loc[frame.prediction_date>=cutoff].copy()
assert train.prediction_date.max()<test.prediction_date.min()
features=['line_id','repair_rate_pct','rate_lag_1','rate_rolling_3','rate_rolling_7','modules_rolling_7','day_of_week_next']
pre=ColumnTransformer([('line',OneHotEncoder(handle_unknown='ignore'),['line_id']),('numeric',SimpleImputer(strategy='median'),[c for c in features if c!='line_id'])])
model=Pipeline([('preprocess',pre),('rf',RandomForestRegressor(n_estimators=250,min_samples_leaf=4,random_state=42,n_jobs=1))])
model.fit(train[features],train.target_next_day)
y=test.target_next_day.to_numpy()
pred=np.clip(model.predict(test[features]),0,100)
base=test.repair_rate_pct.to_numpy()
# Stronger honest benchmark: per-line historical training mean (no test labels).
line_mean=train.groupby('line_id')['target_next_day'].mean()
mean_base=test['line_id'].map(line_mean).to_numpy()
def score(a,b):
 return {'mae_pp':round(float(mean_absolute_error(a,b)),4),'rmse_pp':round(float(np.sqrt(mean_squared_error(a,b))),4),'r2':round(float(r2_score(a,b)),4)}
metrics={'dataset':'Synthetic solar manufacturing; not actual factory production data','task':'next-day repair rate by line (%)','prediction_time':'end of production day t','split':'chronological, common cutoff across lines','train_rows':len(train),'test_rows':len(test),'train_last_target_date':str(train.prediction_date.max().date()),'test_first_target_date':str(test.prediction_date.min().date()),'test_last_target_date':str(test.prediction_date.max().date()),'model':score(y,pred),'persistence_baseline':score(y,base),'historical_line_mean_baseline':score(y,mean_base),'mae_improvement_pct':round(100*(mean_absolute_error(y,base)-mean_absolute_error(y,pred))/mean_absolute_error(y,base),2),'per_line':{}}
for line in sorted(test.line_id.unique()):
 mask=test.line_id.to_numpy()==line
 metrics['per_line'][line]={'rows':int(mask.sum()),'model':score(y[mask],pred[mask]),'baseline':score(y[mask],base[mask]),'historical_mean':score(y[mask],mean_base[mask])}
res=test[['line_id','prediction_date']].copy()
res['actual_repair_rate_pct']=y
res['predicted_repair_rate_pct']=pred
res['baseline_repair_rate_pct']=base
res['historical_mean_baseline_pct']=mean_base
res.to_csv(OUT/'holdout_predictions.csv',index=False,float_format='%.5f')
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(11,5))
agg=res.groupby('prediction_date')[['actual_repair_rate_pct','predicted_repair_rate_pct','baseline_repair_rate_pct']].mean()
agg.plot(ax=ax,linewidth=2)
ax.set(title='Next-day repair rate: chronological holdout (synthetic data)',xlabel='Prediction date',ylabel='Repair rate (%)')
ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(OUT/'actual_vs_predicted.png',dpi=160);plt.close(fig)
print(json.dumps(metrics,indent=2))

# Expanding-window, date-aligned backtests. No target-date overlap between training and test.
backtests=[]
for frac in (0.55,0.65,0.75,0.85):
    start=dates[int(len(dates)*frac)]
    end=dates[min(len(dates)-1,int(len(dates)*(frac+0.10)))]
    tr=frame[frame.prediction_date<start]
    te=frame[(frame.prediction_date>=start)&(frame.prediction_date<end)]
    if len(tr)<40 or len(te)<8: continue
    fold_model=Pipeline([('preprocess',pre),('rf',RandomForestRegressor(n_estimators=250,min_samples_leaf=4,random_state=42,n_jobs=1))])
    fold_model.fit(tr[features],tr.target_next_day)
    actual=te.target_next_day.to_numpy()
    fpred=np.clip(fold_model.predict(te[features]),0,100)
    persistence=te.repair_rate_pct.to_numpy()
    means=te.line_id.map(tr.groupby('line_id').target_next_day.mean()).to_numpy()
    backtests.append({'train_end':str(tr.prediction_date.max().date()),'test_start':str(te.prediction_date.min().date()),'test_end':str(te.prediction_date.max().date()),'n':len(te),'random_forest':score(actual,fpred),'persistence':score(actual,persistence),'historical_line_mean':score(actual,means)})
(OUT/'backtests.json').write_text(json.dumps(backtests,indent=2),encoding='utf-8')
print('Backtest folds:',len(backtests))
