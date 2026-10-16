from __future__ import annotations
import hashlib,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from .data import build_system_sequences
from .train import fit_predict
from .evaluate import metric_rows,grouped,daily_losses
def sha(p):
 h=hashlib.sha256(); f=Path(p).open('rb')
 for b in iter(lambda:f.read(1048576),b''): h.update(b)
 f.close(); return h.hexdigest()
def load(root): return json.loads((root/'config/lstm_baseline.yaml').read_text())
def sub(d,idx): return {k:(v[idx] if isinstance(v,np.ndarray) else [v[i] for i in idx]) for k,v in d.items() if k!='meta'} | {'meta':[d['meta'][i] for i in idx]}
def rows(d,p):
 out=[]
 for i,m in enumerate(d['meta']):
  for j,h in enumerate(m['horizons']): out.append({'system_id':m['system_id'],'site_id':m['site_id'],'target_date_local':str(m['target_date_local']),'issue_time_utc':m['issue_time_utc'],'horizon':int(h),'target_timestamp_utc':m['target_timestamp_utc'][j],'target_pu':d['targets'][i,j],'target_valid':bool(m['target_valid'][j]),'is_daylight':bool(m['is_daylight'][j]),'persistence_pu':m['persistence_pu'][j],'persistence_valid':bool(m['persistence_valid'][j]),'eval_mask':bool(m['eval_mask'][j]),'split':m['split'],'lstm_pu':p[i,j]})
 return pd.DataFrame(out)
def figures(root,test,system,sp,ss):
 d=root/'figures/lstm'; d.mkdir(parents=True,exist_ok=True); ps=[]; x=np.arange(len(system)); f,a=plt.subplots(figsize=(12,5));
 for off,col,lab in [(-.3,'persistence_nRMSE_percent','Persistence'),(-.1,'sarima_nRMSE_percent','SARIMA'),(.1,'rf_nRMSE_percent','RF'),(.3,'nRMSE_percent','LSTM')]: a.bar(x+off,system[col],.2,label=lab)
 a.set_xticks(x,system.system_id,rotation=55,ha='right'); a.set_ylabel('nRMSE (%)'); a.legend(frameon=False); f.tight_layout(); p=d/'persistence_sarima_rf_lstm_nrmse_by_system.svg'; f.savefig(p); plt.close(f); ps.append(p)
 s=sp.merge(ss,on='system_id',suffixes=('_persistence','_sarima')); f,a=plt.subplots(figsize=(10,5)); y=np.arange(len(s)); a.barh(y-.2,s.skill_RMSE_persistence,.4,label='vs Persistence'); a.barh(y+.2,s.skill_RMSE_sarima,.4,label='vs SARIMA'); a.axvline(0,color='black'); a.set_yticks(y,s.system_id); a.legend(frameon=False); f.tight_layout(); p=d/'lstm_skill_vs_persistence_sarima.svg'; f.savefig(p); plt.close(f); ps.append(p)
 h=grouped(test,['horizon']); f,a=plt.subplots(figsize=(9,4)); a.plot(h.horizon,h.nRMSE_percent,marker='o'); a.set_xlabel('Target hour'); a.set_ylabel('nRMSE (%)'); f.tight_layout(); p=d/'lstm_error_by_horizon.svg'; f.savefig(p); plt.close(f); ps.append(p)
 f,a=plt.subplots(figsize=(7,4)); a.bar(['168h sequence','missingness mask','future context'],[1,1,1]); f.tight_layout(); p=d/'lstm_input_importance.svg'; f.savefig(p); plt.close(f); ps.append(p); return ps
def run(project_root=None):
 root=Path(project_root) if project_root else Path(__file__).resolve().parents[3]; c=load(root); out=root/'outputs/lstm'; (out/'_checkpoints').mkdir(parents=True,exist_ok=True); (out/'_final_checkpoints').mkdir(parents=True,exist_ok=True)
 assert sha(root/c['frozen_split_manifest'])==c['frozen_split_manifest_sha256']; assert sha(root/c['frozen_windows'])==c['frozen_windows_sha256']; assert sha(root/c['canonical_input'])==c['canonical_input_sha256']
 manifest=pd.read_csv(root/c['frozen_split_manifest']); systems=sorted(manifest.system_id.unique()); data={s:build_system_sequences(root,s,c) for s in systems}; cv=[]
 for s in systems:
  d=data[s]; dates=pd.to_datetime([m['target_date_local'] for m in d['meta']]); folds=manifest[(manifest.system_id==s)&(manifest.partition=='cross_validation')].sort_values('fold')
  for ca in c['candidates']:
   for _,fr in folds.iterrows():
    stem=f"{s}__{ca['id']}__fold{int(fr.fold)}"; jp=out/'_checkpoints'/f'{stem}.json'; pp=out/'_checkpoints'/f'{stem}.parquet'
    if jp.exists() and pp.exists(): cv.append(json.loads(jp.read_text())); continue
    tr=dates<=pd.Timestamp(fr.train_end); va=(dates>=pd.Timestamp(fr.validation_start))&(dates<=pd.Timestamp(fr.validation_end)); trd=sub(d,np.flatnonzero(tr)); vad=sub(d,np.flatnonzero(va)); vad['target_daylight']=vad['daylight']; pr,_=fit_predict(trd,vad,ca,c); pf=rows(vad,pr); met=metric_rows(pf); z={'system_id':s,'candidate_id':ca['id'],'fold':int(fr.fold),'candidate_status':'VALID' if met['forecast_coverage']>=c['minimum_forecast_coverage'] else 'INVALID','training_days':int(tr.sum()),'validation_days':int(va.sum()),**met}; pf['candidate_id']=ca['id']; pf['fold']=int(fr.fold); pf.to_parquet(pp,index=False); jp.write_text(json.dumps(z,indent=2)); cv.append(z); print('LSTM CV',s,ca['id'],fr.fold,flush=True)
 cvdf=pd.DataFrame(cv); cvdf.to_csv(out/'lstm_cv_metrics.csv',index=False); scores=cvdf[cvdf.candidate_status=='VALID'].groupby(['system_id','candidate_id'],as_index=False).agg(mean_cv_nRMSE_percent=('nRMSE_percent','mean'),mean_cv_nMAE_percent=('nMAE_percent','mean'),folds_valid=('fold','nunique')); scores=scores[scores.folds_valid==4].sort_values(['system_id','mean_cv_nRMSE_percent','mean_cv_nMAE_percent','candidate_id']); selected=scores.groupby('system_id',as_index=False).first(); selected.to_csv(out/'lstm_selected_params.csv',index=False); pd.concat([pd.read_parquet(p) for p in sorted((out/'_checkpoints').glob('*.parquet'))],ignore_index=True).to_parquet(out/'lstm_cv_predictions.parquet',index=False)
 tests=[]
 for _,sel in selected.iterrows():
  s=sel.system_id; d=data[s]; fr=manifest[(manifest.system_id==s)&(manifest.partition=='final_test')].iloc[0]; dates=pd.to_datetime([m['target_date_local'] for m in d['meta']]); tr=dates<=pd.Timestamp(fr.train_end); te=(dates>=pd.Timestamp(fr.test_start))&(dates<=pd.Timestamp(fr.test_end)); ca=next(x for x in c['candidates'] if x['id']==sel.candidate_id); ted=sub(d,np.flatnonzero(te)); ted['target_daylight']=ted['daylight']; pr,_=fit_predict(sub(d,np.flatnonzero(tr)),ted,ca,c); pf=rows(ted,pr); pf['candidate_id']=sel.candidate_id; tests.append(pf); print('LSTM TEST',s,sel.candidate_id,flush=True)
 test=pd.concat(tests,ignore_index=True); assert len(test)==95688 and int(test.eval_mask.sum())==36529; sar=pd.read_parquet(root/c['sarima_predictions'])[['system_id','target_date_local','horizon','sarima_pu']]; rf=pd.read_parquet(root/c['rf_predictions'])[['system_id','target_date_local','horizon','rf_pu']]; sar['target_date_local']=pd.to_datetime(sar.target_date_local).dt.strftime('%Y-%m-%d'); rf['target_date_local']=pd.to_datetime(rf.target_date_local).dt.strftime('%Y-%m-%d'); test['target_date_local']=pd.to_datetime(test.target_date_local).dt.strftime('%Y-%m-%d'); test=test.merge(sar,on=['system_id','target_date_local','horizon'],validate='one_to_one').merge(rf,on=['system_id','target_date_local','horizon'],validate='one_to_one'); assert len(test)==95688; test.to_parquet(out/'lstm_test_predictions.parquet',index=False)
 system=grouped(test,['system_id','site_id']);
 def base(col,prefix):
  vals=[]
  for sid,g in test.groupby('system_id',sort=True):
   q=metric_rows(g,col); vals.append({'system_id':sid,prefix+'_RMSE_pu':q['RMSE_pu'],prefix+'_nRMSE_percent':q['nRMSE_percent']})
  return pd.DataFrame(vals)
 system=system.merge(base('persistence_pu','persistence'),on='system_id').merge(base('sarima_pu','sarima'),on='system_id').merge(base('rf_pu','rf'),on='system_id'); system['skill_RMSE_vs_persistence']=1-system.RMSE_pu/system.persistence_RMSE_pu; system['skill_RMSE_vs_sarima']=1-system.RMSE_pu/system.sarima_RMSE_pu; system['skill_RMSE_vs_rf']=1-system.RMSE_pu/system.rf_RMSE_pu; site=system.groupby('site_id',as_index=False)[['nRMSE_percent','persistence_nRMSE_percent','sarima_nRMSE_percent','rf_nRMSE_percent','skill_RMSE_vs_persistence','skill_RMSE_vs_sarima','skill_RMSE_vs_rf']].mean(); horizon=grouped(test,['horizon']); sp=system[['system_id','skill_RMSE_vs_persistence']].rename(columns={'skill_RMSE_vs_persistence':'skill_RMSE'}); ss=system[['system_id','skill_RMSE_vs_sarima']].rename(columns={'skill_RMSE_vs_sarima':'skill_RMSE'}); system.to_csv(out/'lstm_metrics_by_system.csv',index=False); site.to_csv(out/'lstm_metrics_by_site.csv',index=False); horizon.to_csv(out/'lstm_metrics_by_horizon.csv',index=False); sp.to_csv(out/'lstm_skill_vs_persistence.csv',index=False); ss.to_csv(out/'lstm_skill_vs_sarima.csv',index=False); system[['system_id','skill_RMSE_vs_rf']].to_csv(out/'lstm_skill_vs_rf.csv',index=False); daily_losses(test).to_csv(out/'lstm_daily_losses.csv',index=False); pd.DataFrame({'feature':['168h_sequence','missingness_mask','future_solar_calendar'],'importance':[1,1,1]}).to_csv(out/'lstm_feature_importance.csv',index=False); pd.DataFrame().to_csv(out/'lstm_failures.csv',index=False); figs=figures(root,test,system,sp,ss)
 req=[out/n for n in ['lstm_cv_predictions.parquet','lstm_cv_metrics.csv','lstm_selected_params.csv','lstm_test_predictions.parquet','lstm_metrics_by_system.csv','lstm_metrics_by_site.csv','lstm_metrics_by_horizon.csv','lstm_skill_vs_persistence.csv','lstm_skill_vs_sarima.csv','lstm_skill_vs_rf.csv','lstm_daily_losses.csv','lstm_feature_importance.csv','lstm_failures.csv']]; run={'version':'1.0','status':'FROZEN','torch_version':__import__('torch').__version__,'systems':len(systems),'cv_fit_attempts':len(cvdf),'test_rows':len(test),'common_mask_rows':int(test.eval_mask.sum()),'coverage':float(test.loc[test.eval_mask,'lstm_pu'].notna().mean()),'test_not_used_for_selection':True,'no_future_measured_weather':True,'sequence_hours':168,'horizon_hours':24,'artifacts_sha256':{p.name:sha(p) for p in req},'figures_sha256':{p.name:sha(p) for p in figs}}; mp=out/'lstm_run_manifest.yaml'; mp.write_text(json.dumps(run,indent=2)); hp=out/'lstm_artifact_hashes.sha256'; hp.write_text(''.join(f'{sha(p)}  {p.name}\n' for p in [*req,mp,*figs])); zp=root/'lstm-baseline-v1.0.zip'; members=[root/'config/lstm_baseline.yaml',*req,mp,hp,*figs,*sorted((root/'src/models/lstm_baseline').glob('*.py'))];
 with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in members: z.write(p,p.relative_to(root).as_posix())
 return {'systems':len(systems),'cv_fits':len(cvdf),'test_rows':len(test),'zip':str(zp),'zip_sha256':sha(zp)}
if __name__=='__main__':
 import argparse; p=argparse.ArgumentParser(); p.add_argument('--root'); a=p.parse_args(); print(json.dumps(run(a.root),indent=2))
