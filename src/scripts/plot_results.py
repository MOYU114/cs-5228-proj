"""Draw the figures of a finished run into <results-dir>/figures/, from what the run recorded.

fig1_training_curves.png      per-epoch loss of both training stages (models/<round>/<stage>/history.json)
fig2_validation_by_month.png  round A validation RMSE and mean error per month (results/validation_summary.json)
fig3_predicted_vs_recorded.png  round A validation predictions against the recorded rents (results/validation_predictions.csv)
fig4_monthly_mean_rent.png    recorded mean rent per month and the submission's mean prediction per test month

In fig1 fine-tuning trains on the standardized target, so its training loss is converted with
RMSE = sqrt(loss) * target std of that round's training rows. Training values are epoch averages in train mode
(dropout on, random masks), so they are not exactly the eval-mode error on the same rows.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap,LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bert_rent.data import read_frame,split_frame

# Reference palette, light mode: chart chrome and ink, categorical slots 1-3 (validated all-pairs) and the blue ramp.
SURFACE,INK,INK2,MUTED,GRID,AXIS='#fcfcfb','#0b0b0b','#52514e','#898781','#e1e0d9','#c3c2b7'
BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'
BLUE_RAMP=['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b']


def chrome(ax,title,xlabel,ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title,loc='left',fontsize=11,color=INK,pad=10)
    ax.set_xlabel(xlabel,color=MUTED,fontsize=9);ax.set_ylabel(ylabel,color=MUTED,fontsize=9)
    ax.grid(axis='y',color=GRID,linewidth=1,linestyle='-');ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED,labelsize=9,length=0)
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)


def lines(ax,series,colors):
    """2px lines with an end dot ringed in the surface colour."""
    for name,(x,y) in series.items():
        ax.plot(x,y,color=colors[name],linewidth=2,solid_capstyle='round',solid_joinstyle='round',label=name,zorder=3)
        ax.plot(x[-1],y[-1],'o',color=colors[name],markersize=8,markeredgecolor=SURFACE,markeredgewidth=2,zorder=4)


def end_labels(ax,series,fmt):
    """Labels in ink at the line ends. Series reaching the right edge are labelled to the right, pushed apart with a
    leader line if they would collide; a series that ends earlier is labelled under its end dot."""
    left=min(x[0] for x,_ in series.values());right=max(x[-1] for x,_ in series.values())
    ends=sorted(((y[-1],x[-1],name) for name,(x,y) in series.items() if x[-1]==right),key=lambda t:t[0])
    lo,hi=ax.get_ylim();gap=(hi-lo)*.07;dx=(right-left)*.03;last=-np.inf
    for y,x,name in ends:
        ty=max(y,last+gap);last=ty
        ax.annotate(f'{name}  {fmt(y)}',xy=(x,y),xytext=(x+dx,ty),textcoords='data',va='center',fontsize=9,color=INK2,
                    annotation_clip=False,arrowprops=dict(arrowstyle='-',color=AXIS,linewidth=1) if ty!=y else None)
    for name,(x,y) in series.items():
        if x[-1]<right:
            ax.annotate(f'{name}  {fmt(y[-1])}',xy=(x[-1],y[-1]),xytext=(0,-12),textcoords='offset points',
                        ha='center',va='top',fontsize=9,color=INK2)


def save(fig,path,legend_from=None,columns=3,note=None):
    if legend_from is not None:
        handles,labels=legend_from.get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=columns,frameon=False,fontsize=9,labelcolor=INK2,bbox_to_anchor=(.5,1.0))
    if note:fig.text(.5,-.02,note,ha='center',fontsize=8,color=MUTED)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160,bbox_inches='tight',facecolor=SURFACE);plt.close(fig)
    print(f'saved {path}')


def history(model_dir,round_,stage):
    path=model_dir/round_/stage/'history.json'
    return json.loads(path.read_text()) if path.exists() else []


def target_scales(config):
    """Same formula as train_stage2_finetune.target_stats: population std of MONTHLY_RENT on each round's training rows."""
    frame=read_frame((ROOT/config['data_dir']).resolve(),'train')
    train,_=split_frame(frame,config['validation_months'])
    return {'select':float(train.MONTHLY_RENT.std(ddof=0)),'refit':float(frame.MONTHLY_RENT.std(ddof=0))}


def training_curves(model_dir,figures):
    config=json.loads((model_dir/'config.json').read_text());scales=target_scales(config)
    colors={'A train':BLUE,'A validation':ORANGE,'B train':AQUA}
    panels=[('pretrain','Training stage 1 (pretraining): masked-field reconstruction loss','loss',lambda v:f'{v:.3f}',
             lambda h,r:[e['loss'] for e in h],lambda a:[e['validation']['reconstruction_loss'] for e in a]),
            ('finetune','Training stage 2 (fine-tuning): RMSE against the cleaned rents','RMSE (SGD)',lambda v:f'{v:.0f}',
             lambda h,r:[np.sqrt(e['loss'])*scales[r] for e in h],lambda a:[e['validation']['rmse'] for e in a])]
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),facecolor=SURFACE)
    for ax,(stage,title,ylabel,fmt,train,validation) in zip(axes,panels):
        a,b=history(model_dir,'select',stage),history(model_dir,'refit',stage);series={}
        if a:series['A train']=([e['epoch'] for e in a],train(a,'select'));series['A validation']=([e['epoch'] for e in a],validation(a))
        if b:series['B train']=([e['epoch'] for e in b],train(b,'refit'))
        if not series:raise SystemExit(f'No history.json under {model_dir}')
        chrome(ax,title,'epoch',ylabel);lines(ax,series,colors);end_labels(ax,series,fmt)
        summary=model_dir/'select'/stage/'summary.json'
        if summary.exists() and 'A validation' in series:
            best=json.loads(summary.read_text())['best_epoch'];x,y=series['A validation'];v=y[x.index(best)]
            ax.axvline(best,color=AXIS,linewidth=1,zorder=1)
            ax.plot(best,v,'o',color=ORANGE,markersize=8,markeredgecolor=SURFACE,markeredgewidth=2,zorder=5)
            ax.annotate(f'best epoch {best} · validation {fmt(v)}',xy=(best,1),xycoords=('data','axes fraction'),
                        xytext=(-6,0),textcoords='offset points',ha='right',va='top',fontsize=9,color=INK)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True));ax.margins(x=.02)
    fig.tight_layout(rect=(0,0,.93,.93),w_pad=6)
    save(fig,figures/'fig1_training_curves.png',axes[0],note='A = selection round (train 2021-01..2024-03, validation 2024-04..2025-03); '
         'B = full-data refit, fixed epochs. Training values are epoch averages in train mode; validation uses EMA weights where enabled.')


def validation_by_month(results,figures):
    summary=json.loads((results/'validation_summary.json').read_text());months=sorted(summary['by_month'])
    x=list(range(len(months)));rows=[summary['by_month'][m] for m in months]
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),facecolor=SURFACE)
    series={'recorded rents':(x,[r['original_rent_rmse'] for r in rows]),'cleaned rents':(x,[r['rmse'] for r in rows])}
    chrome(axes[0],f"Validation RMSE per month (all months: {summary['original_rent']['rmse']:.1f} recorded, {summary['rmse']:.1f} cleaned)",
           'approval month','RMSE (SGD)')
    lines(axes[0],series,{'recorded rents':BLUE,'cleaned rents':ORANGE});end_labels(axes[0],series,lambda v:f'{v:.0f}')
    bias=[r['mean_error'] for r in rows]
    chrome(axes[1],f"Mean error per month, prediction - rent (all months: {summary['mean_error']:+.1f})",'approval month','mean error (SGD)')
    axes[1].axhline(0,color=AXIS,linewidth=1,zorder=2)
    lines(axes[1],{'mean error':(x,bias)},{'mean error':BLUE})
    top=int(np.argmax(bias))
    axes[1].annotate(f'{bias[top]:+.0f}',xy=(x[top],bias[top]),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9,color=INK2)
    axes[1].annotate(f'{bias[-1]:+.0f}',xy=(x[-1],bias[-1]),xytext=(10,0),textcoords='offset points',va='center',fontsize=9,color=INK2,annotation_clip=False)
    for ax in axes:
        ax.set_xticks(x[::2]);ax.set_xticklabels(months[::2]);ax.margins(x=.03)
    axes[1].margins(y=.12)
    fig.tight_layout(rect=(0,0,.95,.93),w_pad=6)
    save(fig,figures/'fig2_validation_by_month.png',axes[0],columns=2,
         note=f"Round A model (trained on 2021-01..2024-03) on the {summary['rows']:,} held-out rentals. Cleaned rents: identical-feature rentals "
              'share their mean rent; recorded rents: as in the data, the way the test set is scored.')


def predicted_vs_recorded(results,figures):
    frame=pd.read_csv(results/'validation_predictions.csv');actual,predicted=frame.MONTHLY_RENT_ORIGINAL,frame.Predicted
    limit=float(np.ceil(max(actual.max(),predicted.max())/500)*500)
    fig,ax=plt.subplots(figsize=(6.6,5.6),facecolor=SURFACE)
    chrome(ax,f'Validation: predicted vs recorded rent ({len(frame):,} rentals)','recorded monthly rent (SGD)','predicted monthly rent (SGD)')
    ax.grid(axis='x',color=GRID,linewidth=1,linestyle='-')
    cells=ax.hexbin(actual,predicted,gridsize=55,extent=(0,limit,0,limit),mincnt=1,norm=LogNorm(),linewidths=.2,edgecolors=SURFACE,
                    cmap=LinearSegmentedColormap.from_list('blue',BLUE_RAMP),zorder=2)
    ax.plot([0,limit],[0,limit],color=INK2,linewidth=1,zorder=3)
    ax.annotate('prediction = recorded rent',xy=(limit*.93,limit*.93),xytext=(-8,6),textcoords='offset points',ha='right',fontsize=9,color=INK2)
    ax.set_xlim(0,limit);ax.set_ylim(0,limit);ax.set_aspect('equal')
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'));ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    bar=fig.colorbar(cells,ax=ax,shrink=.75,pad=.03);bar.outline.set_visible(False)
    bar.set_label('rentals per cell',color=MUTED,fontsize=9);bar.ax.tick_params(colors=MUTED,labelsize=9,length=0)
    bar.ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    fig.tight_layout()
    save(fig,figures/'fig3_predicted_vs_recorded.png')


def monthly_mean_rent(submission,figures):
    data=ROOT.parent/'data'
    train=pd.read_csv(data/'train.csv',usecols=['RENT_APPROVAL_DATE','MONTHLY_RENT'])
    test=pd.read_csv(data/'test.csv',usecols=['RENT_APPROVAL_DATE']);predicted=pd.read_csv(submission)
    if len(predicted)!=len(test):raise SystemExit('Submission and test set differ in length')
    index=lambda month:(int(month[:4])-2021)*12+int(month[5:7])-1
    recorded=train.groupby('RENT_APPROVAL_DATE').MONTHLY_RENT.mean();future=predicted.Predicted.groupby(test.RENT_APPROVAL_DATE.to_numpy()).mean()
    series={'recorded, training data':([index(m) for m in recorded.index],recorded.tolist()),
            'predicted, test set':([index(m) for m in future.index],future.tolist())}
    fig,ax=plt.subplots(figsize=(11,4.8),facecolor=SURFACE)
    chrome(ax,'Mean monthly rent: recorded in the training data and predicted for the test set','approval month','mean monthly rent (SGD)')
    lines(ax,series,{'recorded, training data':BLUE,'predicted, test set':ORANGE})
    x,y=series['recorded, training data']
    ax.annotate(f'recorded  {y[-1]:,.0f}',xy=(x[-1],y[-1]),xytext=(-10,12),textcoords='offset points',ha='right',fontsize=9,color=INK2)
    x,y=series['predicted, test set']
    ax.annotate(f'predicted  {y[-1]:,.0f}',xy=(x[-1],y[-1]),xytext=(10,0),textcoords='offset points',va='center',fontsize=9,color=INK2,annotation_clip=False)
    years=range(2021,2027);ax.set_xticks([(year-2021)*12 for year in years]);ax.set_xticklabels([f'{year}-01' for year in years])
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'));ax.margins(x=.02)
    fig.tight_layout(rect=(0,0,.93,.93))
    save(fig,figures/'fig4_monthly_mean_rent.png',ax,columns=2,
         note=f'Recorded: all {len(train):,} training rentals, {recorded.index[0]}..{recorded.index[-1]}. '
              f'Predicted: the submission, {len(test):,} test rentals, {future.index[0]}..{future.index[-1]}.')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'models')
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--submission',type=Path,default=ROOT/'results/submission_bert_two_stage.csv')
    args=parser.parse_args()
    plt.rcParams['font.family']=['Segoe UI','DejaVu Sans']
    figures=args.results_dir/'figures'
    training_curves(args.model_dir,figures)
    validation_by_month(args.results_dir,figures)
    predicted_vs_recorded(args.results_dir,figures)
    monthly_mean_rent(args.submission,figures)


if __name__=='__main__':main()
