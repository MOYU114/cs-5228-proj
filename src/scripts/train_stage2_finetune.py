"""Training stage 2: fine-tune the pretrained backbone with a regression head on the standardized rent.

Round A (select) starts from round A's backbone, trains on the training months and keeps the epoch with the lowest
validation RMSE; it writes <results-dir>/validation_summary.json and validation_predictions.csv. Round B (refit)
starts from round B's backbone and trains that many epochs on every labelled row; tests/test.py predicts with it.
Writes <model-dir>/{select,refit}/finetune/model.pt. Run src/scripts/train_stage1_pretrain.py first.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from bert_rent.data import TARGET,month_label
from bert_rent.engine import atomic,load_regressor,metrics,predict,save_json,train_stage
from bert_rent.run import arguments,best_epoch,setup

ORIGINAL='MONTHLY_RENT_ORIGINAL'


def target_stats(frame):
    y=frame[TARGET].to_numpy(dtype=np.float64);return {'mean':float(y.mean()),'scale':float(y.std()) or 1.}


def scaled_target(frame,stats):return (frame[TARGET].to_numpy(dtype=np.float64)-stats['mean'])/stats['scale']


def original_rents(data_dir,frame):
    """Rents before stage-1 cleaning replaced identical-feature rentals by their mean (removed outlier rows stay out)."""
    path=data_dir.parent/'stage1/train.csv'
    if not path.exists():raise ValueError(f'Original rents not found: {path}')
    rents=pd.read_csv(path,usecols=['ROW_ID',ORIGINAL]).set_index('ROW_ID')[ORIGINAL]
    return rents.loc[frame.ROW_ID].to_numpy(dtype=np.float64)


def select(run):
    """Round A: gradients from the training months only; the validation months pick the epoch."""
    cfg,schema,train,val,root=run.cfg,run.schema,run.train,run.val,run.root/'select'
    stats=target_stats(train)
    train_stage('finetune',schema,schema.dataset(train,scaled_target(train,stats)),cfg,root,target_stats=stats,
                validation={'dataset':schema.dataset(val),'truth':val[TARGET].to_numpy()})
    model,loaded,saved=load_regressor(root/'finetune/model.pt')
    p=predict(model,loaded.dataset(val),cfg['micro_batch'],'cuda',saved['target'],'select/validation predict')
    truth=val[TARGET].to_numpy(dtype=np.float64);score=metrics(truth,p)
    if not np.isclose(score['rmse'],saved['criterion'],atol=1e-6,rtol=0):raise ValueError('Reload metric mismatch')
    # Early stopping scored the cleaned rents. Identical features get identical predictions, so the squared error
    # against the original rents differs from it only by a constant: it would pick the same epoch, and is reported
    # because the test set is scored on uncleaned rents.
    original=original_rents(run.data_dir,val);month=val.MONTH.map(month_label).to_numpy(dtype=str)
    by_month={m:dict(metrics(truth[month==m],p[month==m]),original_rent_rmse=metrics(original[month==m],p[month==m])['rmse'],
                     rows=int((month==m).sum())) for m in sorted(set(month))}
    pretrain=json.loads((root/'pretrain/summary.json').read_text())
    summary=dict(score,original_rent=metrics(original,p),rows=len(val),pretrain_best_epoch=pretrain['best_epoch'],
                 best_reconstruction_loss=pretrain['best_reconstruction_loss'],finetune_best_epoch=saved['epoch'],by_month=by_month)
    save_json(run.results/'validation_summary.json',summary)
    out=pd.DataFrame({'ROW_ID':val.ROW_ID.to_numpy(),'RENT_APPROVAL_DATE':month,TARGET:truth,ORIGINAL:original,'Predicted':p})
    atomic(run.results/'validation_predictions.csv',lambda path:out.to_csv(path,index=False))
    print(json.dumps({'stage':'finetune','mode':'select','best_epoch':saved['epoch'],'validation_rmse':score['rmse'],
                      'validation_rmse_original_rent':summary['original_rent']['rmse']}),flush=True)
    del model;torch.cuda.empty_cache()


def refit(run):
    """Round B: every labelled row, round A's epoch count, no validation and no further selection."""
    cfg,schema,frame=run.cfg,run.schema,run.full
    epochs={stage:best_epoch(run.root/'select',stage) for stage in ['pretrain','finetune']}
    stats=target_stats(frame)
    summary=train_stage('finetune',schema,schema.dataset(frame,scaled_target(frame,stats)),cfg,run.root/'refit',target_stats=stats,epochs=epochs['finetune'])
    save_json(run.results/'refit.json',{'rows':len(frame),'epochs':epochs,'schedule_epochs':{s:cfg[s]['epochs'] for s in epochs}})
    print(json.dumps({'stage':'finetune',**summary}),flush=True)


def main():
    args=arguments(__doc__);run=setup(args,'finetune')
    if args.round in ['select','all']:select(run)
    if args.round in ['refit','all']:refit(run)


if __name__=='__main__':main()
