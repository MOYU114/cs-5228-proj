import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

# Model inputs, in ../data_preprocess/stage3 column order. Stage 3 already holds category IDs (0 = unknown) and
# standardized numbers, so nothing is fitted here.
TARGET='MONTHLY_RENT'
IDS=['TOWN_ID','STREET_ID','BUILDING_ID','SUBZONE_ID','PLANNING_AREA_ID','REGION_ID','FLAT_MODEL_ID','FLAT_TYPE_ORD']
DIMS=[16,16,32,8,8,8,8,8]
POI=['MRT_NEAREST_M','MRT_NEAREST3_MEAN_M','MRT_COUNT_500M','MRT_COUNT_1000M','PRIMARY_NEAREST_M','SECONDARY_NEAREST_M','PRIMARY_COUNT_1000M','MALL_NEAREST_M','MALL_COUNT_1000M','MALL_COUNT_2000M']
NUMS=['FLOOR_AREA_SQM','LEASE_COMMENCE_DATE','AGE_YEARS','MONTH_INDEX','LATITUDE','LONGITUDE','MAX_FLOOR','YEAR_COMPLETED','COMPLETION_AGE']+POI
CALENDAR=['MONTH_SIN','MONTH_COS']
FLAG='HDB_INFO_MISSING'
FEATURES=IDS+NUMS+CALENDAR+[FLAG]


def month_label(index):return f'{2021+int(index)//12}-{int(index)%12+1:02d}'


class Schema:
    """The stage-3 layout a model is built for: field order and category sizes."""
    @classmethod
    def load(cls,folder):
        folder=Path(folder);schema=json.loads((folder/'schema.json').read_text(encoding='utf-8'))
        if schema.get('stage')!=3:raise ValueError(f'Expected stage-3 data in {folder}; run scripts/preprocess.py')
        columns=[c['name'] for c in schema['columns']]
        if columns!=['ROW_ID']+FEATURES+[TARGET]:raise ValueError(f'Stage-3 columns differ from the model inputs: {columns}')
        sizes={c['name']:c['cardinality'] for c in schema['columns'] if 'cardinality' in c}
        obj=cls();obj.state={'version':2,'ids':IDS,'nums':NUMS,'dims':DIMS,'cardinalities':[sizes[c] for c in IDS]}
        return obj

    @classmethod
    def restore(cls,state):
        if state.get('version')!=2 or state['ids']!=IDS or state['nums']!=NUMS or state['dims']!=DIMS:raise ValueError('Feature schema mismatch')
        obj=cls();obj.state=state;return obj

    @property
    def cardinalities(self):return self.state['cardinalities']

    def tensors(self,frame):
        cats=frame[IDS].to_numpy(dtype=np.int64)
        if (cats<0).any() or (cats>=np.array(self.cardinalities)).any():raise ValueError('Category ID outside its vocabulary')
        arrays=[cats]+[frame[c].to_numpy(dtype=np.float32) for c in [NUMS,CALENDAR,[FLAG]]]
        return tuple(torch.tensor(v) for v in arrays)

    def dataset(self,frame,target=None):
        tensors=self.tensors(frame)
        if target is not None:tensors+= (torch.from_numpy(np.asarray(target,dtype=np.float32)),)
        return TensorDataset(*tensors)


def decode_months(folder,frame):
    """Approval month as an integer (2021-01 = 0), decoded from the standardized MONTH_INDEX."""
    stats=json.loads((Path(folder)/'encoder.json').read_text(encoding='utf-8'))['standardize']['columns']['MONTH_INDEX']
    value=frame.MONTH_INDEX.to_numpy(dtype=np.float64)*stats['std']+stats['mean']
    index=np.rint(value).astype(np.int64)
    if np.abs(value-index).max()>1e-6:raise ValueError('MONTH_INDEX does not decode to whole months')
    return index


def read_frame(folder,split):
    """A stage-3 table; train also gets MONTH (integer approval month, not a model input) for the time split.
    Train ROW_IDs have gaps where stage 1 removed rent outliers; every test row must be present, in order."""
    folder=Path(folder);dtypes=json.loads((folder/'schema.json').read_text(encoding='utf-8'))['tables'][split]
    frame=pd.read_csv(folder/f'{split}.csv',dtype=dtypes)
    if frame.columns.tolist()!=['ROW_ID']+FEATURES+([TARGET] if split=='train' else []):raise ValueError(f'{split}: unexpected columns')
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():raise ValueError(f'{split}: stage-3 values must be finite numbers')
    if not (frame.ROW_ID.is_unique and frame.ROW_ID.is_monotonic_increasing):raise ValueError(f'{split}: ROW_ID must be unique and ordered')
    if split=='test' and not np.array_equal(frame.ROW_ID,np.arange(len(frame))):raise ValueError('test: every original row must be present')
    if split=='train':frame['MONTH']=decode_months(folder,frame)
    return frame


def split_frame(frame,months):
    """Whole-month time split: the last `months` approval months are validation."""
    ordered=frame.sort_values(['MONTH','ROW_ID'])
    periods=sorted(ordered.MONTH.unique())
    if not 0<months<len(periods):raise ValueError(f'validation_months must be in 1..{len(periods)-1}')
    held=ordered.MONTH.isin(periods[-months:]).to_numpy()
    return ordered[~held].copy(),ordered[held].copy()
