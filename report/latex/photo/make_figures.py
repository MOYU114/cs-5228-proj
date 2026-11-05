"""Render the report's data figures as vector PDFs sized for an IEEE two-column page.

Run from the project root:  python report/latex/photo/make_figures.py
Inputs (read only):
  experiments/eda/core/{monthly_counts,rent_by_month,rent_by_flat_type}.csv   -> fig1, fig2
  results/validation_summary.json                                             -> fig4
  results/validation_predictions.csv (local, gitignored; skipped if absent)   -> fig5
  results/baselines/round_a/by_month.csv + validation_summary.json            -> fig6
Every figure is drawn at its final printed width (one column = 3.5 in), so the
8 pt labels and 7 pt ticks print at that size and match across figures.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EDA, RESULTS, OUT = ROOT / 'experiments/eda/core', ROOT / 'results', Path(__file__).resolve().parent
COLUMN = 3.5  # IEEEtran \columnwidth in inches
BLUE, ORANGE, GRAY, INK = '#1f5a96', '#d1701f', '#6b7280', '#222222'
thousands = FuncFormatter(lambda v, _: f'{v:,.0f}')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'lines.linewidth': 1.1,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
    'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
    'axes.grid': True, 'grid.color': '#e5e5e5', 'grid.linewidth': .5, 'axes.axisbelow': True,
    'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.columnspacing': 1.2,
    'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02,
})


def month_index(month):
    return (int(month[:4]) - 2021) * 12 + int(month[5:7]) - 1


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f'saved {OUT / name}')


def rent_timeline():
    counts, rents = pd.read_csv(EDA / 'monthly_counts.csv'), pd.read_csv(EDA / 'rent_by_month.csv')
    assert counts.train.sum() == 150000 and counts.test.sum() == 50000
    x, xr = counts.month.map(month_index), rents.RENT_APPROVAL_DATE.map(month_index)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(COLUMN, 2.55), sharex=True,
                                      gridspec_kw={'height_ratios': [1, 1.7], 'hspace': .12})
    top.bar(x, counts.train, width=.8, color=BLUE, label='Train')
    top.bar(x, counts.test, width=.8, bottom=counts.train, color=ORANGE, label='Test')
    top.set_ylabel('Rows')
    top.set_ylim(0, 4600)
    top.yaxis.set_major_formatter(thousands)
    top.legend(ncol=2, loc='upper left', bbox_to_anchor=(0, 1.22))
    bottom.fill_between(xr, rents.q25, rents.q75, color=BLUE, alpha=.15, linewidth=0, label='IQR')
    bottom.plot(xr, rents['mean'], color=BLUE, label='Mean')
    bottom.plot(xr, rents['median'], color=ORANGE, linewidth=.9, label='Median')
    bottom.set_ylabel('Rent (SGD/month)')
    bottom.set_ylim(1500, 3800)
    bottom.yaxis.set_major_formatter(thousands)
    bottom.legend(ncol=3, loc='lower right')
    bottom.text(month_index('2025-11'), 2600, 'Test rent\nunavailable', ha='center', va='center', color=GRAY, fontsize=7)
    cut = month_index('2025-03') + .5
    for ax in (top, bottom):
        ax.axvline(cut, color=GRAY, linestyle='--', linewidth=.6)
        ax.grid(axis='x', visible=False)
    bottom.set_xticks([month_index(f'{y}-01') for y in range(2021, 2027)], [f'{y}-01' for y in range(2021, 2027)])
    bottom.set_xlim(-1, x.max() + 1)
    save(fig, 'fig1_rent_timeline.pdf')


def flat_type_iqr():
    rows = pd.read_csv(EDA / 'rent_by_flat_type.csv')
    assert rows.n.sum() == 150000
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(COLUMN, 1.75))
    ax.hlines(y, rows.q25, rows.q75, color=BLUE, alpha=.4, linewidth=5, label='P25–P75')
    ax.plot(rows['median'], y, 'o', color=BLUE, markersize=3.5, label='Median')
    ax.set_yticks(y, [f'{t} ($n$={n:,})' for t, n in zip(rows.FLAT_TYPE, rows.n)])
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Monthly rent (SGD/month)')
    ax.set_xlim(1000, 4000)
    ax.set_ylim(-.6, len(rows) - .4)
    ax.xaxis.set_major_formatter(thousands)
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right', ncol=2)
    save(fig, 'fig2_flat_type_iqr.pdf')


def validation_by_month():
    summary = json.loads((RESULTS / 'validation_summary.json').read_text())
    months = sorted(summary['by_month'])
    rows = [summary['by_month'][m] for m in months]
    x = np.arange(len(months))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(COLUMN, 2.6), sharex=True,
                                      gridspec_kw={'height_ratios': [1.6, 1], 'hspace': .12})
    top.plot(x, [r['original_rent_rmse'] for r in rows], 'o-', color=BLUE, markersize=2.5, label='Original rent')
    top.plot(x, [r['rmse'] for r in rows], 's-', color=ORANGE, markersize=2.5, label='Cleaned rent')
    top.set_ylabel('RMSE (SGD)')
    top.set_ylim(bottom=400)
    top.yaxis.set_major_locator(MaxNLocator(5))
    top.legend(ncol=2, loc='upper right')
    bias = [r['mean_error'] for r in rows]
    bottom.bar(x, bias, width=.6, color=GRAY)
    bottom.axhline(0, color=INK, linewidth=.6)
    bottom.set_ylabel('Mean error (SGD)')
    bottom.set_ylim(-65, 45)
    bottom.set_xticks(x[::2], months[::2])
    bottom.set_xlabel('Approval month')
    for ax in (top, bottom):
        ax.grid(axis='x', visible=False)
    save(fig, 'fig4_validation_by_month.pdf')


def predicted_vs_recorded():
    path = RESULTS / 'validation_predictions.csv'
    if not path.exists():
        print(f'skipped fig5: {path} not found')
        return
    frame = pd.read_csv(path)
    actual, predicted = frame.MONTHLY_RENT_ORIGINAL, frame.Predicted
    limit = float(np.ceil(max(actual.max(), predicted.max()) / 1000) * 1000)
    fig, ax = plt.subplots(figsize=(COLUMN * .92, 2.7))
    cells = ax.hexbin(actual, predicted, gridsize=45, extent=(0, limit, 0, limit), mincnt=1, norm=LogNorm(),
                      linewidths=0, cmap=LinearSegmentedColormap.from_list('b', ['#d6e4f3', BLUE, '#0b2a4a']))
    ax.plot([0, limit], [0, limit], color=INK, linewidth=.6)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect('equal')
    ax.set_xlabel('Recorded rent (SGD/month)')
    ax.set_ylabel('Predicted rent (SGD/month)')
    ax.xaxis.set_major_formatter(thousands)
    ax.yaxis.set_major_formatter(thousands)
    bar = fig.colorbar(cells, ax=ax, fraction=.046, pad=.03)
    bar.set_label('Rows per cell')
    bar.outline.set_linewidth(.5)
    save(fig, 'fig5_predicted_vs_recorded.pdf')


def model_bias_by_month():
    """Monthly mean error (prediction minus original rent), historical protocol, 35,058 rows."""
    frame = pd.read_csv(RESULTS / 'baselines/round_a/by_month.csv')
    frame = frame[(frame.protocol == 'historical') & (frame.features == 'full')]
    summary = json.loads((RESULTS / 'validation_summary.json').read_text())
    months = sorted(summary['by_month'])
    assert sum(summary['by_month'][m]['rows'] for m in months) == 35058
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    styles = [('B2_ridge', 'Ridge', '#7b3294', 'v-'), ('B1_recent', 'Recent 6-month mean', GRAY, 'D-'),
              ('B3_catboost', 'Shallow CatBoost', ORANGE, 's-')]
    for key, label, color, marker in styles:
        rows = frame[frame.model == key].set_index('RENT_APPROVAL_DATE').loc[months]
        assert rows.rows.sum() == 35058
        ax.plot(x, rows.bias, marker, color=color, markersize=2.5, label=label)
    ax.plot(x, [summary['by_month'][m]['mean_error'] for m in months], 'o-', color=BLUE, markersize=2.5,
            label='Two-stage BERT')
    ax.axhline(0, color=INK, linewidth=.6)
    ax.set_ylabel('Mean error (SGD/month)')
    ax.set_ylim(-120, 580)
    ax.set_xticks(x[::2], months[::2])
    ax.set_xlabel('Approval month')
    ax.grid(axis='x', visible=False)
    ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.0))
    save(fig, 'fig6_model_bias_by_month.pdf')


if __name__ == '__main__':
    rent_timeline()
    flat_type_iqr()
    validation_by_month()
    predicted_vs_recorded()
    model_bias_by_month()
