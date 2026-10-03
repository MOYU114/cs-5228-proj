import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel
from .data import DIMS,IDS,NUMS


# One token per field: the category IDs, the numbers, the calendar month (sin and cos together) and the imputation flag.
TOKENS=IDS+NUMS+['CALENDAR','FLAG']
N_IDS,N_TOKENS=len(IDS),len(TOKENS)
CALENDAR_TOKEN,FLAG_TOKEN=TOKENS.index('CALENDAR'),TOKENS.index('FLAG')
# Fields that determine one another are masked together, otherwise reconstruction is a lookup: the month index gives
# the calendar month, and with the month visible lease year <-> age and completion year <-> completion age.
LINKED=[[TOKENS.index(t) for t in group] for group in [['MONTH_INDEX','CALENDAR'],['LEASE_COMMENCE_DATE','AGE_YEARS','YEAR_COMPLETED','COMPLETION_AGE']]]
GROUPS=LINKED+[[i] for i in range(N_TOKENS) if i not in sum(LINKED,[])]


def sample_mask(batch,device,probability,generator=None):
    choices=torch.rand(batch,len(GROUPS),device=device,generator=generator)<probability
    empty=~choices.any(1)
    if empty.any():choices[empty,torch.randint(len(GROUPS),(int(empty.sum()),),device=device,generator=generator)]=True
    mask=torch.zeros(batch,N_TOKENS,dtype=torch.bool,device=device)
    for i,group in enumerate(GROUPS):mask[:,group]=choices[:,i:i+1]
    return mask


def fixed_masks(rows,probability,seed):
    """Validation masks drawn once from a private generator, so every epoch is scored on the same task."""
    return sample_mask(rows,'cpu',probability,torch.Generator().manual_seed(seed))


class Backbone(nn.Module):
    def __init__(self,cardinalities,config):
        super().__init__();w=config['width']
        self.categories=nn.ModuleList([nn.Embedding(n,d,padding_idx=0) for n,d in zip(cardinalities,DIMS)])
        self.projections=nn.ModuleList([nn.Linear(d,w) for d in DIMS])
        self.numbers=nn.ModuleList([nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,w)) for _ in NUMS])
        self.calendar=nn.Linear(2,w);self.flag=nn.Linear(1,w)
        self.cls=nn.Parameter(torch.randn(1,1,w)*.02);self.mask_token=nn.Parameter(torch.randn(1,1,w)*.02)
        cfg=BertConfig(vocab_size=1,type_vocab_size=1,hidden_size=w,num_hidden_layers=config['layers'],
                       num_attention_heads=config['heads'],intermediate_size=config['ffn'],
                       hidden_dropout_prob=config['dropout'],attention_probs_dropout_prob=config['dropout'],
                       max_position_embeddings=N_TOKENS+1,pad_token_id=0)
        cfg._attn_implementation='sdpa'
        self.bert=BertModel(cfg,add_pooling_layer=False)
        self.bert.embeddings.word_embeddings.weight.requires_grad_(False)

    def forward(self,categories,numbers,calendar,flag,mask=None):
        tokens=[p(e(categories[:,i])) for i,(e,p) in enumerate(zip(self.categories,self.projections))]
        tokens += [m(numbers[:,i:i+1]) for i,m in enumerate(self.numbers)]
        tokens += [self.calendar(calendar),self.flag(flag)]
        x=torch.stack(tokens,1)
        if mask is not None:x=torch.where(mask[:,:,None],self.mask_token,x)
        x=torch.cat([self.cls.expand(len(x),-1,-1),x],1)
        return self.bert(inputs_embeds=x).last_hidden_state


class Pretrainer(nn.Module):
    def __init__(self,cardinalities,config):
        super().__init__();self.backbone=Backbone(cardinalities,config);w=config['width']
        self.category_heads=nn.ModuleList([nn.Linear(w,n) for n in cardinalities])
        self.numeric_heads=nn.ModuleList([nn.Linear(w,1) for _ in NUMS])
        self.calendar_head=nn.Linear(w,2);self.flag_head=nn.Linear(w,1)
        self.probability=config['pretrain']['mask_probability']

    def loss(self,batch,mask=None):
        cat,num,cal,flag=batch
        if mask is None:mask=sample_mask(len(cat),cat.device,self.probability)
        h=self.backbone(cat,num,cal,flag,mask)[:,1:]
        category=[];numeric=[];binary=[]
        for i,head in enumerate(self.category_heads):
            valid=mask[:,i] & cat[:,i].ne(0)
            if valid.any():category.append(F.cross_entropy(head(h[valid,i]),cat[valid,i]))
        for i,head in enumerate(self.numeric_heads):
            valid=mask[:,N_IDS+i]
            if valid.any():numeric.append(F.mse_loss(head(h[valid,N_IDS+i]).squeeze(-1),num[valid,i]))
        valid=mask[:,CALENDAR_TOKEN]
        if valid.any():numeric.append(F.mse_loss(self.calendar_head(h[valid,CALENDAR_TOKEN]),cal[valid]))
        valid=mask[:,FLAG_TOKEN]
        if valid.any():binary.append(F.binary_cross_entropy_with_logits(self.flag_head(h[valid,FLAG_TOKEN]),flag[valid]))
        loss=h.sum()*0
        for group in [category,numeric,binary]:
            if group:loss=loss+torch.stack(group).mean()
        return loss


class Regressor(nn.Module):
    def __init__(self,cardinalities,config):
        super().__init__();self.backbone=Backbone(cardinalities,config)
        self.head=nn.Sequential(nn.LayerNorm(config['width']),nn.Linear(config['width'],256),nn.GELU(),
                                nn.Dropout(config['dropout']),nn.Linear(256,1))
    def forward(self,*batch):return self.head(self.backbone(*batch)[:,0]).squeeze(-1)
    def loss(self,batch):return F.mse_loss(self(*batch[:4]),batch[4])
