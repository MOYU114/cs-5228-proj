"""Explicit, bounded station availability; source STATUS is not authoritative."""
import json
from pathlib import Path

import pandas as pd

from .common import normalize_text


def load_calendar():
    return json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))


def active_stations(mrt, month):
    calendar = load_calendar()
    if not isinstance(month, str) or str(pd.Period(month, freq='M')) != month:
        raise ValueError(f'Invalid MRT month: {month}')
    if not calendar['min_month'] <= month <= calendar['max_month']:
        raise ValueError(f'MRT calendar does not cover {month}; update the explicit calendar first')
    codes = mrt.CODE.map(normalize_text)
    if codes.isna().any() or codes.duplicated().any():
        raise ValueError('MRT CODE must be nonmissing and unique')
    keep = []
    for code, name in zip(codes, mrt.NAME.map(normalize_text)):
        if code not in calendar['stations'] or name != calendar['stations'][code]['name']:
            raise ValueError(f'Unknown MRT code/name: {code}, {name}; update the explicit calendar')
        start = calendar['stations'][code]['available_from']
        keep.append(start is not None and start <= month)
    return mrt.loc[keep].drop_duplicates(['LATITUDE', 'LONGITUDE'])
