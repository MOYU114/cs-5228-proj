"""Shared start of the two training scripts: arguments, data, time split and the frozen run config."""
import argparse
import json
from pathlib import Path
import platform
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import transformers
from .data import Schema,month_label,read_frame,split_frame
from .engine import configure,probe,save_json

ROOT=Path(__file__).resolve().parents[1]


def arguments(description):
    parser=argparse.ArgumentParser(description=description,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--round',choices=['select','refit','all'],default='all',
                        help='select = round A (choose the epoch on the validation months), refit = round B (all rows), all = A then B')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/bert.json')
    parser.add_argument('--model-dir',type=Path,default=ROOT/'models')
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results')
    return parser.parse_args()


def setup(args,stage):
    """Round A of stage 1 starts a model directory: it probes the GPU for the micro-batch and freezes config.json.
    Every later step must find that file with unchanged settings."""
    cfg=json.loads(args.config.read_text());data_dir=(ROOT/cfg['data_dir']).resolve()
    configure(cfg['seed'])
    for protected in [ROOT.parent/'data',ROOT.parent/'data_preprocess']:
        for target in [args.model_dir,args.results_dir]:
            if target.resolve()==protected.resolve() or protected.resolve() in target.resolve().parents:raise ValueError('Outputs cannot overwrite data')
    root=args.model_dir;resolved=root/'config.json'
    if not resolved.exists() and (stage!='pretrain' or args.round=='refit'):
        raise ValueError(f'{root} has no run yet; start with scripts/train_stage1_pretrain.py')
    schema=Schema.load(data_dir);frame=read_frame(data_dir,'train')
    train,val=split_frame(frame,cfg['validation_months'])
    full=frame.sort_values(['MONTH','ROW_ID']).copy()
    root.mkdir(parents=True,exist_ok=True)
    if resolved.exists():
        previous=json.loads(resolved.read_text());expected=dict(previous);expected.pop('micro_batch')
        if expected!=cfg:raise ValueError('Run config changed; choose a new model directory')
        cfg=previous
    else:
        size,attempts=probe(schema,train,cfg);cfg['micro_batch']=size
        save_json(root/'memory_probe.json',attempts);save_json(resolved,cfg)
    results=args.results_dir;results.mkdir(parents=True,exist_ok=True)
    span=lambda part:[month_label(part.MONTH.iloc[0]),month_label(part.MONTH.iloc[-1])]
    save_json(results/'split.json',{'validation_months':cfg['validation_months'],
                                   'train':{'rows':len(train),'months':span(train)},
                                   'validation':{'rows':len(val),'months':span(val)},
                                   'refit':{'rows':len(full)}})
    save_json(results/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
                                        'transformers':transformers.__version__,'numpy':np.__version__,'pandas':pd.__version__,
                                        'gpu':torch.cuda.get_device_name(0),'precision':'float32','seed':cfg['seed']})
    return SimpleNamespace(cfg=cfg,schema=schema,train=train,val=val,full=full,root=root,results=results,data_dir=data_dir)


def best_epoch(select_root,stage):
    """The epoch round A kept for this stage; round B trains exactly that many."""
    path=select_root/stage/'summary.json'
    summary=json.loads(path.read_text()) if path.exists() else {}
    if summary.get('mode')!='select' or not summary.get('complete'):raise ValueError(f'Round A of {stage} has not completed; run it first')
    return summary['best_epoch']
