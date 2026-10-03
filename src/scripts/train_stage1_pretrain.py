"""Training stage 1: pretrain the BERT backbone by reconstructing masked fields. Rent labels are not used.

Round A (select) trains on the training months and keeps the epoch with the lowest reconstruction loss on the
validation months' features. Round B (refit) trains that many epochs on every labelled row.
Writes <model-dir>/{select,refit}/pretrain/backbone.pt. Next: src/scripts/train_stage2_finetune.py.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from torch.utils.data import TensorDataset
from bert_rent.engine import train_stage
from bert_rent.model import fixed_masks
from bert_rent.run import arguments,best_epoch,setup


def main():
    args=arguments(__doc__);run=setup(args,'pretrain');cfg,schema=run.cfg,run.schema
    if args.round in ['select','all']:
        masks=fixed_masks(len(run.val),cfg['pretrain']['mask_probability'],cfg['seed'])
        summary=train_stage('pretrain',schema,schema.dataset(run.train),cfg,run.root/'select',
                            validation={'dataset':TensorDataset(*schema.tensors(run.val),masks)})
        print(json.dumps({'stage':'pretrain',**summary}),flush=True)
    if args.round in ['refit','all']:
        summary=train_stage('pretrain',schema,schema.dataset(run.full),cfg,run.root/'refit',epochs=best_epoch(run.root/'select','pretrain'))
        print(json.dumps({'stage':'pretrain',**summary}),flush=True)


if __name__=='__main__':main()
