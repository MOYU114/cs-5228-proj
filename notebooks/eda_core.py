"""Core-data EDA. Only derived copies are normalized; input CSVs are never written."""
from pathlib import Path
import json
import re
import unicodedata
import numpy as np
import pandas as pd

TEXT = ['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'FLAT_MODEL', 'FURNISHED']
KEY = ['TOWN', 'BLOCK', 'STREET']

def normalize_text(value):
    if pd.isna(value):
        return value
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value)).strip().lower())

def normalized_copy(df):
    """Deterministic, target-free normalization; no learned vocabulary or imputation."""
    out = df.copy()
    for col in TEXT:
        out[col] = out[col].map(normalize_text)
    out['FLAT_TYPE'] = out.FLAT_TYPE.str.replace(r'^(\d+)\s*[- ]\s*room$', r'\1-room', regex=True)
    return out

def key_index(df):
    return pd.MultiIndex.from_frame(df[KEY])

def dump_json(value, path):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')

def summarize_groups(df, keys):
    return df.groupby(keys, observed=True, dropna=False).MONTHLY_RENT.agg(
        n='size', mean='mean', median='median', q25=lambda x: x.quantile(.25),
        q75=lambda x: x.quantile(.75), min='min', max='max').reset_index()

def tvd(a, b):
    p, q = a.value_counts(normalize=True, dropna=False).align(b.value_counts(normalize=True, dropna=False), fill_value=0)
    return float((p - q).abs().sum() / 2)

def run(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    raw = {s: pd.read_csv(root / 'data' / f'{s}.csv', dtype={'BLOCK': 'string'}) for s in ['train', 'test']}
    assert 'MONTHLY_RENT' in raw['train'] and 'MONTHLY_RENT' not in raw['test'], 'Test targets must not be available or consumed.'
    clean = {s: normalized_copy(d) for s, d in raw.items()}
    profiles, numeric, frequencies, mappings, anomalies, duplicate_summary = [], [], [], [], [], []
    for split, d in raw.items():
        c = clean[split]
        for col in d:
            s = d[col]
            p = dict(split=split, field=col, dtype=str(s.dtype), n=len(s), missing=int(s.isna().sum()),
                     blank=int(s.astype('string').str.strip().eq('').sum()), unique=int(s.nunique()), constant=s.nunique(dropna=False) == 1,
                     min=s.min(), max=s.max())
            profiles.append(p)
            if pd.api.types.is_numeric_dtype(s):
                numeric.append(dict(split=split, field=col, **s.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).to_dict()))
            else:
                for val, count in s.value_counts(dropna=False).items():
                    frequencies.append(dict(split=split, field=col, value=val, n=int(count), proportion=count/len(d)))
            if col in TEXT:
                m = pd.DataFrame({'raw': s, 'normalized': c[col]}).value_counts(dropna=False).rename('n').reset_index()
                m.insert(0, 'field', col); m.insert(0, 'split', split)
                mappings.append(m)
        date = pd.to_datetime(d.RENT_APPROVAL_DATE, format='%Y-%m', errors='coerce')
        c['AGE_YEARS'] = date.dt.year - d.LEASE_COMMENCE_DATE
        checks = {'invalid_month':date.isna(), 'nonpositive_area':d.FLOOR_AREA_SQM.le(0),
                  'nonfinite_area':~np.isfinite(d.FLOOR_AREA_SQM), 'negative_age':c.AGE_YEARS.lt(0),
                  'invalid_lease_year':d.LEASE_COMMENCE_DATE.isna() | d.LEASE_COMMENCE_DATE.mod(1).ne(0),
                  'nonfinite_fee':~np.isfinite(d.FEE)}
        if split == 'train':
            checks.update(nonpositive_rent=d.MONTHLY_RENT.le(0), nonfinite_rent=~np.isfinite(d.MONTHLY_RENT))
        for name, mask in checks.items():
            anomalies.append(dict(split=split, rule=name, lower=None, upper=None, n=int(mask.sum())))
            if mask.any():
                d.loc[mask].assign(source_row_0based=d.index[mask]).to_csv(output/f'{split}_{name}.csv',index=False)
        for col in ['FLOOR_AREA_SQM', 'AGE_YEARS'] + (['MONTHLY_RENT'] if split=='train' else []):
            # Flag thresholds are always learned from train, including the test audit.
            ref = clean['train'][col]
            q1,q3 = ref.quantile([.25,.75]); lo,hi = q1-1.5*(q3-q1),q3+1.5*(q3-q1)
            mask = c[col].lt(lo) | c[col].gt(hi)
            anomalies.append(dict(split=split, rule=f'{col}_train_IQR_1.5', lower=lo, upper=hi, n=int(mask.sum())))
        for name, frame in [('raw', d), ('normalized', c.drop(columns='AGE_YEARS'))]:
            variants = [('all_available_fields', list(frame)), ('X_only', [x for x in frame if x!='MONTHLY_RENT']),
                        ('block', KEY), ('block_month', KEY+['RENT_APPROVAL_DATE'])]
            for label, cols in variants:
                sizes = frame.groupby(cols, dropna=False, observed=True).size()
                repeated = sizes[sizes.gt(1)]
                duplicate_summary.append(dict(split=split, form=name, definition=label,
                    unique_groups=len(sizes), repeated_groups=len(repeated), rows_in_repeated_groups=int(repeated.sum()),
                    excess_rows=int((sizes-1).sum()), max_multiplicity=int(sizes.max())))
                if name=='normalized' and label in ['block','block_month']:
                    sizes.rename('n').sort_values(ascending=False).to_csv(output/f'{split}_{label}_counts.csv')
        c.groupby(KEY).agg(n=('RENT_APPROVAL_DATE','size'), months=('RENT_APPROVAL_DATE','nunique'),
                           first=('RENT_APPROVAL_DATE','min'), last=('RENT_APPROVAL_DATE','max')).reset_index().to_csv(output/f'{split}_block_history.csv',index=False)
    pd.DataFrame(profiles).to_csv(output/'field_overview.csv', index=False)
    pd.DataFrame(numeric).to_csv(output/'numeric_summary.csv', index=False)
    pd.DataFrame(frequencies).to_csv(output/'category_frequencies_raw.csv',index=False)
    pd.concat(mappings,ignore_index=True).to_csv(output/'category_normalization_map.csv',index=False)
    pd.DataFrame(anomalies).to_csv(output/'anomaly_flags.csv', index=False)
    pd.DataFrame(duplicate_summary).to_csv(output/'duplicates.csv', index=False)

    train,test = clean['train'],clean['test']
    monthly = pd.concat([d.RENT_APPROVAL_DATE.value_counts().sort_index().rename(s) for s,d in clean.items()],axis=1).fillna(0).astype(int)
    monthly.index.name = 'month'; monthly.to_csv(output/'monthly_counts.csv')
    grouped = {}
    for col, name in [('RENT_APPROVAL_DATE','month'),('TOWN','town'),('FLAT_TYPE','flat_type'),('AGE_YEARS','age'),('FLOOR_AREA_SQM','area')]:
        grouped[name] = summarize_groups(train, col)
        grouped[name].to_csv(output/f'rent_by_{name}.csv',index=False)
    train['YEAR'] = train.RENT_APPROVAL_DATE.str[:4]
    summarize_groups(train, 'YEAR').to_csv(output/'rent_by_year.csv',index=False)
    summarize_groups(train, ['RENT_APPROVAL_DATE','FLAT_TYPE']).to_csv(output/'rent_by_month_flat_type.csv',index=False)
    train['AGE_BAND'] = pd.cut(train.AGE_YEARS, bins=[-np.inf,-1,9,19,29,39,49,np.inf],
                              labels=['<0','0-9','10-19','20-29','30-39','40-49','50+'])
    summarize_groups(train,'AGE_BAND').to_csv(output/'rent_by_age_band.csv',index=False)
    train['AREA_BAND'] = pd.cut(train.FLOOR_AREA_SQM, bins=[0,50,70,90,110,130,150,np.inf],right=False,
                               labels=['<50','50-<70','70-<90','90-<110','110-<130','130-<150','150+'])
    summarize_groups(train,'AREA_BAND').to_csv(output/'rent_by_area_band.csv',index=False)
    summarize_groups(train,['YEAR','FLAT_TYPE']).to_csv(output/'rent_by_year_flat_type.csv',index=False)
    train.groupby('FLAT_TYPE').FLOOR_AREA_SQM.agg(['size','min','median','max']).to_csv(output/'area_by_flat_type.csv')
    extremes = pd.concat([train.nsmallest(20,'MONTHLY_RENT'),train.nlargest(20,'MONTHLY_RENT')]).drop_duplicates()
    extremes.assign(source_row_0based=extremes.index).to_csv(output/'rent_extreme_examples.csv',index=False)
    train.MONTHLY_RENT.value_counts().rename_axis('rent_SGD_month').rename('n').to_csv(output/'rent_value_frequencies.csv')
    rent_stats = train.MONTHLY_RENT.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).to_dict()
    rent_stats.update(skew=float(train.MONTHLY_RENT.skew()), multiple_50_fraction=float(train.MONTHLY_RENT.mod(50).eq(0).mean()),
                      pearson_area=float(train.MONTHLY_RENT.corr(train.FLOOR_AREA_SQM)), spearman_area=float(train.MONTHLY_RENT.rank().corr(train.FLOOR_AREA_SQM.rank())),
                      pearson_age=float(train.MONTHLY_RENT.corr(train.AGE_YEARS)))

    # Descriptive shift statistics, no significance or causal claims. Test X used only for audit.
    drift, distribution = [], []
    periods={'all_train':train,'recent_train_2024-10_to_2025-03':train[train.RENT_APPROVAL_DATE.ge('2024-10')]}
    for form in ['raw','normalized']:
        for reference in periods:
            a = (raw['train'] if form=='raw' else train)
            if reference!='all_train': a=a[a.RENT_APPROVAL_DATE.ge('2024-10')]
            b = raw['test'] if form=='raw' else test
            for col in TEXT:
                unseen=~b[col].isin(a[col])
                drift.append(dict(form=form,reference=reference,field=col,metric='TVD',value=tvd(a[col],b[col]),
                                  test_unseen_rows=int(unseen.sum()),test_unseen_categories=int(b.loc[unseen,col].nunique())))
                if form=='normalized':
                    p,q=a[col].value_counts(normalize=True).align(b[col].value_counts(normalize=True),fill_value=0)
                    for val in p.index:
                        distribution.append(dict(reference=reference,field=col,value=val,train_fraction=p[val],test_fraction=q[val],delta_pp=100*(q[val]-p[val])))
    for ref,a in periods.items():
        for col in ['FLOOR_AREA_SQM','LEASE_COMMENCE_DATE','AGE_YEARS']:
            av,bv=a[col],test[col]
            pooled=np.sqrt((av.var()+bv.var())/2)
            drift.append(dict(form='normalized',reference=ref,field=col,metric='standardized_mean_difference',value=(bv.mean()-av.mean())/pooled,
                              train_mean=av.mean(),test_mean=bv.mean(),train_median=av.median(),test_median=bv.median(),
                              train_min=av.min(),train_max=av.max(),test_min=bv.min(),test_max=bv.max(),
                              test_outside_train_range=int((bv.lt(av.min())|bv.gt(av.max())).sum())))
    pd.DataFrame(drift).to_csv(output/'feature_drift.csv',index=False)
    pd.DataFrame(distribution).to_csv(output/'category_distribution_comparison.csv',index=False)
    coverage=[]
    for ref,a in periods.items():
        ki,kj=key_index(a),key_index(test)
        seen=kj.isin(ki.unique())
        coverage.append(dict(reference=ref,train_unique_blocks=len(ki.unique()),test_unique_blocks=len(kj.unique()),
                             shared_unique_blocks=int(kj.unique().isin(ki.unique()).sum()),test_seen_rows=int(seen.sum()),
                             test_new_rows=int((~seen).sum()),test_new_blocks=int((~kj.unique().isin(ki.unique())).sum())))
        test.loc[~seen,KEY].value_counts().rename('test_rows').to_csv(output/f'test_new_blocks_{ref}.csv')
    overlap_months=sorted(set(train.RENT_APPROVAL_DATE)&set(test.RENT_APPROVAL_DATE))
    feature_columns=list(raw['test'].columns)
    train_x=pd.MultiIndex.from_frame(train[feature_columns]); test_x=pd.MultiIndex.from_frame(test[feature_columns])
    cross_x = test_x.isin(train_x.unique())
    test.assign(source_row_0based=test.index).loc[cross_x, ['source_row_0based']+feature_columns].to_csv(output/'cross_split_identical_X_test_rows.csv',index=False)
    conflict=train.groupby(feature_columns,observed=True,dropna=False).MONTHLY_RENT.agg(n='size',targets='nunique')
    conflict=conflict[conflict.targets.gt(1)]
    conflict.reset_index().to_csv(output/'identical_X_multiple_targets.csv',index=False)
    # Describe the suggested team's temporal boundary, without adopting a validation protocol.
    past=train[train.RENT_APPROVAL_DATE.le('2024-09')]; recent=periods['recent_train_2024-10_to_2025-03']
    suggested={'fit_rows':len(past),'holdout_rows':len(recent),'holdout_new_block_rows':int((~key_index(recent).isin(key_index(past).unique())).sum())}
    summary = dict(rows={s:len(d) for s,d in raw.items()},columns={s:len(d.columns) for s,d in raw.items()},
        dates={s:dict(min=d.RENT_APPROVAL_DATE.min(),max=d.RENT_APPROVAL_DATE.max(),months=d.RENT_APPROVAL_DATE.nunique()) for s,d in raw.items()},
        rent=rent_stats,block_coverage=coverage,overlap_months=overlap_months,
        test_rows_overlapping_months=int(test.RENT_APPROVAL_DATE.isin(overlap_months).sum()),
        test_rows_after_train_max=int(test.RENT_APPROVAL_DATE.gt(train.RENT_APPROVAL_DATE.max()).sum()),
        test_rows_identical_X_in_train=int(cross_x.sum()),identical_X_multi_target_groups=len(conflict),identical_X_multi_target_rows=int(conflict.n.sum()),
        horizon_months=int(pd.Period(test.RENT_APPROVAL_DATE.max(),'M').ordinal-pd.Period(train.RENT_APPROVAL_DATE.max(),'M').ordinal),
        suggested_window_counts=suggested,
        normalization={s:{col:dict(raw_unique=int(raw[s][col].nunique()),normalized_unique=int(clean[s][col].nunique()),changed_rows=int(raw[s][col].ne(clean[s][col]).sum())) for col in TEXT} for s in raw})
    dump_json(summary,output/'summary.json')
    return summary, clean
