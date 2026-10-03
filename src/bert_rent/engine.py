from contextlib import contextmanager,nullcontext
import gc
import json
import math
import os
from pathlib import Path
import random
import time
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import Schema
from .model import Pretrainer, Regressor


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def configure(seed,cuda=True):
    """Call before any CUDA work: cuBLAS reads its workspace setting when the first handle is created."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if cuda and not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed_all(seed)


def atomic(path,writer):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        writer(temp);temp.replace(path)
    finally:temp.unlink(missing_ok=True)


def save_json(path,obj):atomic(path,lambda p:p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8'))
def save_torch(path,obj):atomic(path,lambda p:torch.save(obj,p))


def rng_state(generator):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'loader':generator.get_state()}


def restore_rng(state,generator):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader'])


def build_optimizer(model,config,stage):
    cfg=config[stage]
    if stage=='pretrain':params=[{'params':[p for p in model.parameters() if p.requires_grad],'lr':cfg['lr']}]
    else:params=[{'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':cfg['lr']},
                 {'params':model.head.parameters(),'lr':cfg['head_lr']}]
    return torch.optim.AdamW(params,weight_decay=config['weight_decay'],foreach=False,fused=False)


def schedule(epoch,cfg):
    if epoch<cfg['warmup']:return (epoch+1)/cfg['warmup']
    progress=min((epoch-cfg['warmup'])/max(cfg['epochs']-1-cfg['warmup'],1),1.)
    return .1+.9*.5*(1+math.cos(math.pi*progress))


def progress(iterable,desc):
    """Per-batch bar on stderr; disabled when stderr is not a terminal, so redirected logs stay clean."""
    return tqdm(iterable,desc=desc,unit='batch',leave=False,dynamic_ncols=True,disable=None)


@torch.inference_mode()
def predict(model,dataset,batch,device,target_stats,desc='predict'):
    model.eval();values=[]
    for raw in progress(DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0),desc):
        values.append(model(*(x.to(device) for x in raw[:4])).cpu().numpy())
    pred=np.concatenate(values).astype(np.float64)*target_stats['scale']+target_stats['mean']
    if not np.isfinite(pred).all():raise FloatingPointError('Nonfinite predictions')
    return pred


def metrics(y,p):
    r=np.asarray(p,dtype=np.float64)-np.asarray(y,dtype=np.float64)
    if not np.isfinite(r).all():raise FloatingPointError('Nonfinite residuals')
    return {'rmse':float(np.sqrt(np.mean(r*r))),'mae':float(np.mean(np.abs(r))),'mean_error':float(np.mean(r))}


OUTPUT={'pretrain':'backbone.pt','finetune':'model.pt'}
CRITERION={'pretrain':'reconstruction_loss','finetune':'rmse'}


@torch.inference_mode()
def reconstruction_loss(model,dataset,batch,device,desc='validate'):
    """Masked-field loss on held-out rows; the dataset's last tensor holds fixed masks."""
    model.eval();total=0.;count=0
    for raw in progress(DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0),desc):
        raw=[x.to(device) for x in raw]
        total+=float(model.loss(tuple(raw[:4]),raw[4]))*len(raw[4]);count+=len(raw[4])
    return total/count


def evaluate(stage,model,validation,config,device,target_stats,desc='validate'):
    if stage=='pretrain':
        return {'reconstruction_loss':reconstruction_loss(model,validation['dataset'],config['micro_batch'],device,desc)}
    return metrics(validation['truth'],predict(model,validation['dataset'],config['micro_batch'],device,target_stats,desc))


class EMA:
    """Exponential moving average of all weights: shadow <- d*shadow + (1-d)*weight after every optimizer step.
    d = min(decay, (1+t)/(10+t)) ramps up so the randomly initialised head is forgotten quickly."""
    def __init__(self,model,decay):
        self.decay=decay;self.step=0
        self.shadow={k:v.detach().clone() for k,v in model.state_dict().items()}

    @torch.no_grad()
    def update(self,model):
        self.step+=1;d=min(self.decay,(1+self.step)/(10+self.step))
        for k,v in model.state_dict().items():
            if v.dtype.is_floating_point:self.shadow[k].lerp_(v,1-d)
            else:self.shadow[k].copy_(v)

    def state_dict(self):return {'decay':self.decay,'step':self.step,'shadow':self.shadow}

    def load_state_dict(self,state):
        if state['decay']!=self.decay:raise ValueError('EMA decay mismatch')
        self.step=state['step']
        for k,v in state['shadow'].items():self.shadow[k].copy_(v)

    @contextmanager
    def applied(self,model):
        """Temporarily load the shadow weights into `model`; the training weights wait on the CPU."""
        backup={k:v.detach().to('cpu',copy=True) for k,v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:yield
        finally:model.load_state_dict(backup)


def train_stage(stage,schema,dataset,config,root,validation=None,target_stats=None,epochs=None,device='cuda',stop_after=None):
    """Round A (validation given): early stopping, keep the best validation epoch.
    Round B (epochs given): no validation, train exactly `epochs` epochs and keep the last.
    The learning-rate schedule always spans config[stage]['epochs'], so round B follows round A's trajectory."""
    if (validation is None)==(epochs is None):raise ValueError('Pass exactly one of validation or epochs')
    if epochs is not None and not 1<=epochs<=config[stage]['epochs']:raise ValueError(f'Refit epochs out of range: {epochs}')
    mode='select' if epochs is None else 'refit'
    root=Path(root);directory=root/stage;directory.mkdir(parents=True,exist_ok=True)
    seed_all(config['seed']);cls=Pretrainer if stage=='pretrain' else Regressor
    model=cls(schema.cardinalities,config).to(device)
    if stage=='finetune':
        path=root/'pretrain'/'summary.json';summary=json.loads(path.read_text()) if path.exists() else {}
        if not summary.get('complete') or summary.get('mode')!=mode:
            raise ValueError(f'Stage 1 of round {mode} has not completed; run src/scripts/train_stage1_pretrain.py first')
        pretrained=torch.load(root/'pretrain'/OUTPUT['pretrain'],map_location='cpu',weights_only=False)
        if pretrained['schema']!=schema.state or pretrained['config']!=config or pretrained['mode']!=mode:
            raise ValueError('Pretraining config/schema/round mismatch')
        model.backbone.load_state_dict(pretrained['backbone']);del pretrained
    decay=config[stage].get('ema_decay');ema=EMA(model,decay) if decay else None
    optimizer=build_optimizer(model,config,stage)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:schedule(e,config[stage]))
    generator=torch.Generator().manual_seed(config['seed'])
    loader=DataLoader(dataset,batch_size=config['effective_batch'],shuffle=True,generator=generator,num_workers=0)
    history=[];start=0;best=float('inf');best_epoch=0;bad=0
    signature={'stage':stage,'mode':mode,'config':config,'schema':schema.state,'rows':len(dataset),'target':target_stats,
               'validation_rows':len(validation['dataset']) if validation is not None else 0,'epochs':epochs}
    latest=directory/'latest.pt';output=directory/OUTPUT[stage]
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck['signature']!=signature:raise ValueError('Resume signature mismatch; use a new model directory')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        history=ck['history'];start=ck['epoch'];best=ck['best'];best_epoch=ck['best_epoch'];bad=ck['bad']
        if ema:ema.load_state_dict(ck['ema'])
        restore_rng(ck['rng'],generator);del ck
        if best_epoch:
            saved=torch.load(output,map_location='cpu',weights_only=False)
            if saved['epoch']!=best_epoch or saved['criterion']!=(best if mode=='select' else None):
                raise ValueError('Best/recovery checkpoint mismatch')
            del saved
    print(json.dumps({'stage':stage,'mode':mode,'parameters':sum(p.numel() for p in model.parameters()),'resume_epoch':start,'micro_batch':config['micro_batch']}),flush=True)
    maximum=config[stage]['epochs'] if mode=='select' else epochs
    patience=config[stage]['patience'] if mode=='select' else None
    for epoch in range(start+1,maximum+1):
        if patience is not None and bad>=patience:break
        model.train();tick=time.perf_counter();loss_total=0.;count=0
        lrs=[g['lr'] for g in optimizer.param_groups]
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        bar=progress(loader,f'{mode}/{stage} epoch {epoch}/{maximum}')
        for group in bar:
            optimizer.zero_grad(set_to_none=True);n=len(group[0])
            for offset in range(0,n,config['micro_batch']):
                batch=tuple(x[offset:offset+config['micro_batch']].to(device) for x in group)
                loss=model.loss(batch)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                fraction=len(batch[0])/n;(loss*fraction).backward()
                loss_total+=float(loss.detach())*len(batch[0]);count+=len(batch[0])
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();bar.set_postfix(loss=f'{loss_total/count:.4f}',refresh=False)
            if ema:ema.update(model)
        scheduler.step();score=None;keep=False
        # With EMA, validation and the kept checkpoint both use the averaged weights; training resumes from the raw ones.
        with ema.applied(model) if ema else nullcontext():
            if mode=='select':
                score=evaluate(stage,model,validation,config,device,target_stats,f'{mode}/{stage} epoch {epoch} validate')
                current=score[CRITERION[stage]]
                if current<best:best=current;best_epoch=epoch;bad=0;keep=True
                else:bad+=1
            elif epoch==maximum:best_epoch=epoch;keep=True
            if keep:
                state={'schema':schema.state,'config':config,'epoch':epoch,'mode':mode,'ema_decay':decay,
                       'criterion':best if mode=='select' else None}
                if stage=='pretrain':state['backbone']=model.backbone.state_dict()
                else:state.update(model=model.state_dict(),target=target_stats)
                save_torch(output,state)
        entry={'epoch':epoch,'loss':loss_total/count,'learning_rates':lrs,'validation':score,'best_epoch':best_epoch,
               'bad_epochs':bad,
               'seconds':time.perf_counter()-tick,'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
        history.append(entry)
        save_torch(latest,{'signature':signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                          'scheduler':scheduler.state_dict(),'epoch':epoch,'history':history,'best':best,
                          'best_epoch':best_epoch,'bad':bad,'ema':ema.state_dict() if ema else None,'rng':rng_state(generator)})
        save_json(directory/'history.json',history);print(json.dumps({'stage':stage,'mode':mode,**entry}),flush=True)
        if stop_after is not None and epoch>=stop_after:break
    last=history[-1]['epoch'] if history else start
    complete=last>=maximum or (patience is not None and bad>=patience)
    summary={'mode':mode,'complete':complete,'epoch':last,'best_epoch':best_epoch}
    if mode=='select':summary['best_'+CRITERION[stage]]=best
    if complete:
        save_json(directory/'summary.json',summary)
    del model,optimizer,scheduler,loader,ema;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return summary


def load_regressor(path,device='cuda'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    schema=Schema.restore(saved['schema']);model=Regressor(schema.cardinalities,saved['config']).to(device)
    model.load_state_dict(saved['model']);return model,schema,saved


def probe(schema,features,config):
    attempts=[]
    for size in config['micro_batch_candidates']:
        success=True
        for stage,cls in [('pretrain',Pretrainer),('finetune',Regressor)]:
            seed_all(config['seed']);model=cls(schema.cardinalities,config).cuda();opt=build_optimizer(model,config,stage)
            ema=EMA(model,config[stage]['ema_decay']) if config[stage].get('ema_decay') else None
            tensors=schema.tensors(features.iloc[:size]);batch=tuple(x.cuda() for x in tensors)
            if stage=='finetune':batch+=(torch.zeros(size,device='cuda'),)
            loss=None
            try:
                torch.cuda.reset_peak_memory_stats()
                for _ in range(2):
                    opt.zero_grad(set_to_none=True);loss=model.loss(batch);loss.backward();opt.step()
                    if ema:ema.update(model)
                torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
                fits=peak<.9*torch.cuda.get_device_properties(0).total_memory
                attempts.append({'stage':stage,'size':size,'fits':fits,'peak_gpu_bytes':peak})
                if not fits:success=False
            except torch.cuda.OutOfMemoryError:
                attempts.append({'stage':stage,'size':size,'fits':False});success=False
            finally:
                del model,opt,ema,batch,tensors,loss;gc.collect();torch.cuda.empty_cache()
            if not success:break
        if success:return size,attempts
    raise RuntimeError('No configured micro-batch fits; adjust config explicitly')
