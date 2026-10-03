"""Static descriptive figures, with full sample scope and units; no fitted models."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter

BLUE, ORANGE, GRAY = '#2563a6', '#d97625', '#6b7280'

def run(clean, stats: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white',
                         'savefig.facecolor':'white','axes.grid':True,'grid.alpha':.16,'axes.axisbelow':True})
    train,test=clean['train'],clean['test']
    scope=f"Train {train.RENT_APPROVAL_DATE.min()} to {train.RENT_APPROVAL_DATE.max()} (n={len(train):,}); all rows retained"
    both=scope.replace('; all rows retained','')+f"; test {test.RENT_APPROVAL_DATE.min()} to {test.RENT_APPROVAL_DATE.max()} (n={len(test):,})"
    def save(fig,name,title,subtitle):
        height=fig.get_size_inches()[1]
        fig.text(.5,1-.10/height,title,ha='center',va='top',fontsize=16,fontweight='bold')
        fig.text(.5,1-.44/height,subtitle,ha='center',va='top',fontsize=9,color=GRAY)
        fig.tight_layout(rect=[0,.015,1,1-.78/height])
        fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight')
        plt.close(fig)
    rent=train.MONTHLY_RENT
    fig,ax=plt.subplots(1,2,figsize=(12,4.8))
    ax[0].hist(rent,bins=np.arange(250,7751,100),color=BLUE,edgecolor='white',linewidth=.4)
    ax[0].set(xlabel='Monthly rent (SGD/month)',ylabel='Records (count)',title='Observed rent distribution (100 SGD bins)')
    ax[0].axvline(rent.median(),color=ORANGE,ls='--',label=f'Median {rent.median():,.0f} SGD')
    ax[0].legend(frameon=False)
    x=np.sort(rent.to_numpy()); ax[1].plot(x,np.arange(1,len(x)+1)/len(x),color=BLUE)
    ax[1].set(xlabel='Monthly rent (SGD/month)',ylabel='Cumulative fraction of records',title='Full range, including both tails',ylim=(0,1))
    for q in [.01,.5,.99]:
        v=rent.quantile(q); ax[1].scatter(v,q,color=ORANGE,s=25); ax[1].annotate(f'{q:.0%}: {v:,.0f}',(v,q),xytext=(7,-14 if q>.9 else 7),textcoords='offset points',fontsize=9)
    save(fig,'eda_rent_distribution','MONTHLY_RENT: distribution and tails',scope)

    counts=pd.read_csv(stats/'monthly_counts.csv',index_col=0); dates=pd.to_datetime(counts.index)
    monthly=pd.read_csv(stats/'rent_by_month.csv'); td=pd.to_datetime(monthly.RENT_APPROVAL_DATE)
    fig,ax=plt.subplots(2,1,figsize=(12.8,7),sharex=True,gridspec_kw={'height_ratios':[1,1.35]})
    ax[0].bar(dates,counts.train,width=23,color=BLUE,label='Train records')
    ax[0].bar(dates,counts.test,width=23,bottom=counts.train,color=ORANGE,label='Test records')
    ax[0].set(ylabel='Records / month',title='Approval-month coverage; March 2025 appears in both files',ylim=(0,counts.sum(axis=1).max()*1.25)); ax[0].legend(frameon=False,ncol=2)
    ax[1].fill_between(td,monthly.q25,monthly.q75,alpha=.15,color=BLUE,label='Train IQR (25th–75th percentiles)')
    ax[1].plot(td,monthly['mean'],color=BLUE,label='Train mean');ax[1].plot(td,monthly['median'],color=ORANGE,label='Train median')
    ax[1].set(ylabel='Monthly rent (SGD/month)',xlabel='Rent approval month',title='Observed training rent; test target is unavailable')
    ax[1].legend(frameon=False,loc='upper left',fontsize=9)
    ax[1].text(pd.Timestamp('2025-11-01'),2400,'No test rent observations',ha='center',color=GRAY)
    for a in ax:
        a.axvline(pd.Timestamp('2025-03-01'),color=GRAY,ls='--',lw=1)
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=6)); a.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    save(fig,'eda_month_coverage_and_rent','Time coverage and monthly rent',both)

    towns=pd.read_csv(stats/'rent_by_town.csv').sort_values('median')
    fig,ax=plt.subplots(figsize=(10.8,9.5)); y=np.arange(len(towns))
    ax.hlines(y,towns.q25,towns.q75,color=BLUE,lw=5,alpha=.5,label='IQR')
    ax.scatter(towns['median'],y,color=BLUE,s=22,label='Median',zorder=3)
    ax.scatter(towns['mean'],y,color=ORANGE,s=20,marker='x',label='Mean',zorder=3)
    ax.set_yticks(y,[f'{r.TOWN.title()} (n={r.n:,})' for r in towns.itertuples()])
    ax.set(xlabel='Monthly rent (SGD/month)',ylabel='Town',title='Unadjusted summaries; location, size and time are mixed')
    ax.legend(frameon=False,loc='lower right')
    save(fig,'eda_rent_by_town','Rental records by town',scope)

    order=['1-room','2-room','3-room','4-room','5-room','executive']
    ft=pd.read_csv(stats/'rent_by_flat_type.csv').set_index('FLAT_TYPE').reindex(order)
    fig,ax=plt.subplots(1,2,figsize=(12.8,5.3),gridspec_kw={'width_ratios':[1,1.35]})
    y=np.arange(len(ft));ax[0].hlines(y,ft.q25,ft.q75,color=BLUE,lw=7,alpha=.5,label='IQR');ax[0].scatter(ft['median'],y,color=BLUE,label='Median')
    ax[0].set_yticks(y,[f'{t} (n={int(ft.loc[t,"n"]):,})' for t in order]);ax[0].set(xlabel='Monthly rent (SGD/month)',ylabel='Normalized flat type',title='All training months');ax[0].legend(frameon=False,loc='upper left')
    mf=pd.read_csv(stats/'rent_by_month_flat_type.csv')
    for t in ['2-room','3-room','4-room','5-room','executive']:
        d=mf[mf.FLAT_TYPE.eq(t)];ax[1].plot(pd.to_datetime(d.RENT_APPROVAL_DATE),d['median'],label=t,lw=1.6)
    ax[1].set(xlabel='Rent approval month',ylabel='Median rent (SGD/month)',title='Within-type monthly medians (still unadjusted)')
    ax[1].xaxis.set_major_locator(mdates.YearLocator());ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax[1].legend(frameon=False,ncol=2,fontsize=9)
    save(fig,'eda_rent_by_flat_type','Flat type and time',scope+'; right panel excludes sparse 1-room')

    fig,ax=plt.subplots(1,2,figsize=(12.8,5.2))
    for a,col,label in [(ax[0],'FLOOR_AREA_SQM','Floor area (m²)'),(ax[1],'AGE_YEARS','Approval year − lease commencement year (years)')]:
        h=a.hexbin(train[col],rent,gridsize=45,mincnt=1,bins='log',cmap='Blues')
        fig.colorbar(h,ax=a,label='Records per hexagon (log color scale)')
        a.set(xlabel=label,ylabel='Monthly rent (SGD/month)',title='Floor area' if col=='FLOOR_AREA_SQM' else 'Lease-based age proxy (includes flagged age < 0)')
    save(fig,'eda_rent_area_age','Rent, floor area and lease-based age',scope+'; descriptive association, no causal adjustment')

    comparison=pd.read_csv(stats/'category_distribution_comparison.csv')
    fig,ax=plt.subplots(1,2,figsize=(12.8,8.5),gridspec_kw={'width_ratios':[1,1.3]})
    d=comparison[(comparison.reference=='all_train')&(comparison.field=='FLAT_TYPE')].set_index('value').reindex(order)
    y=np.arange(len(d));ax[0].barh(y-.18,d.train_fraction*100,.36,color=BLUE,label='Train');ax[0].barh(y+.18,d.test_fraction*100,.36,color=ORANGE,label='Test')
    ax[0].set_yticks(y,order);ax[0].set(xlabel='Share of records (%)',ylabel='Normalized flat type',title='Flat-type composition');ax[0].legend(frameon=False)
    d=comparison[(comparison.reference=='all_train')&(comparison.field=='TOWN')].sort_values('delta_pp')
    y=np.arange(len(d));ax[1].barh(y,d.delta_pp,color=np.where(d.delta_pp>=0,ORANGE,BLUE))
    ax[1].set_yticks(y,d.value.str.title());ax[1].axvline(0,color=GRAY,lw=.7)
    ax[1].set(xlabel='Test share − train share (percentage points)',ylabel='Town',title='Town composition: percentage-point differences')
    save(fig,'eda_categorical_shift','Train–test feature distributions after text normalization',both)

    fig,ax=plt.subplots(1,2,figsize=(12.8,5))
    for a,col,label in [(ax[0],'FLOOR_AREA_SQM','Floor area (m²)'),(ax[1],'AGE_YEARS','Approval year − lease commencement year (years)')]:
        for d,color,label2 in [(train,BLUE,'All train'),(train[train.RENT_APPROVAL_DATE.ge('2024-10')],GRAY,'Recent train: 2024-10 to 2025-03'),(test,ORANGE,'Test')]:
            x=np.sort(d[col]);a.plot(x,np.arange(1,len(x)+1)/len(x),color=color,label=f'{label2} (n={len(x):,})')
        a.set(xlabel=label,ylabel='Cumulative fraction of records',ylim=(0,1));a.legend(frameon=False,fontsize=8)
    save(fig,'eda_numeric_shift','Floor area and age: full-train and recent-train comparisons',both)
