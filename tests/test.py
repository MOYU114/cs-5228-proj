"""Test: predict the test set and write the submission file.

By default the round-B model (refit on every labelled row) predicts; --round select uses the round-A model instead,
which never saw the validation months. Run after src/scripts/train_stage1_pretrain.py and src/scripts/train_stage2_finetune.py.
The test set has no rents, so this produces predictions only; the local score is round A's results/validation_summary.json.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from bert_rent.data import read_frame
from bert_rent.engine import atomic,configure,load_regressor,predict


def run_inference(model_dir,data_dir,output,round_='refit'):
    if output.exists():raise ValueError('Output already exists; choose another path')
    for protected in [ROOT.parent/'data',ROOT.parent/'data_preprocess',data_dir]:
        if output.resolve()==protected.resolve() or protected.resolve() in output.resolve().parents:raise ValueError('Cannot overwrite data')
    path=model_dir/round_/'finetune/summary.json'
    if not path.exists() or not json.loads(path.read_text())['complete']:
        raise ValueError(f'Stage 2 of round {round_} has not completed; run src/scripts/train_stage2_finetune.py first')
    model,schema,saved=load_regressor(model_dir/round_/'finetune/model.pt')
    if saved['mode']!=round_:raise ValueError(f'{model_dir/round_} does not hold a round-{round_} model')
    frame=read_frame(data_dir,'test')
    pred=predict(model,schema.dataset(frame),saved['config']['micro_batch'],'cuda',saved['target'],'test predict')
    example=pd.read_csv(ROOT.parent/'data/example-submission.csv')
    if example.columns.tolist()!=['Id','Predicted'] or not np.array_equal(example.Id,frame.ROW_ID):raise ValueError('Example ID mismatch')
    if not np.isfinite(pred).all() or (np.round(pred,1)<=0).any():raise ValueError('Invalid rounded predictions')
    atomic(output,lambda p:pd.DataFrame({'Id':frame.ROW_ID,'Predicted':pred}).to_csv(p,index=False,float_format='%.1f',lineterminator='\n'))
    return {'path':str(output),'rows':len(pred),'minimum':float(np.round(pred,1).min()),'maximum':float(np.round(pred,1).max())}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'models')
    parser.add_argument('--round',choices=['select','refit'],default='refit',
                        help='refit = round B model, trained on every labelled row (default); select = round A model, trained without the validation months')
    parser.add_argument('--data-dir',type=Path,help='stage-3 folder holding test.csv; default: the data_dir of the model\'s config.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/submission_bert_two_stage.csv')
    args=parser.parse_args()
    config=json.loads((args.model_dir/'config.json').read_text())
    configure(config['seed'])
    print(json.dumps(run_inference(args.model_dir,args.data_dir or (ROOT/config['data_dir']).resolve(),args.output,args.round)),flush=True)


if __name__=='__main__':main()
