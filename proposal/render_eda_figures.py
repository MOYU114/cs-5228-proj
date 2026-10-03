"""Render two slide-sized evidence figures from existing EDA aggregates.

Run from the project root: python proposal/render_eda_figures.py
Requires matplotlib; reads aggregate CSVs only and never edits EDA outputs.
"""
from pathlib import Path
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / 'experiments/eda/core'
OUT = ROOT / 'proposal/figures'
BLUE, ORANGE, GRAY = '#2563a6', '#d97625', '#6b7280'


def read(name):
    with (STATS / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.axisbelow': True, 'savefig.facecolor': 'white'})
    counts, rents = read('monthly_counts.csv'), read('rent_by_month.csv')
    assert sum(int(r['train']) for r in counts) == 150000
    assert sum(int(r['test']) for r in counts) == 50000
    dates = [datetime.strptime(r['month'], '%Y-%m') for r in counts]
    observed = [datetime.strptime(r['RENT_APPROVAL_DATE'], '%Y-%m') for r in rents]
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 4.5), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1.6]})
    axes[0].bar(dates, [int(r['train']) for r in counts], width=23, color=BLUE, label='Train')
    axes[0].bar(dates, [int(r['test']) for r in counts], width=23,
                bottom=[int(r['train']) for r in counts], color=ORANGE, label='Test')
    axes[0].set_ylabel('Records')
    axes[0].set_ylim(0, 5500)
    axes[0].set_yticks([0, 4000], ['0', '4,000'])
    axes[0].legend(frameon=False, ncol=2, loc='upper right', fontsize=13)
    axes[1].fill_between(observed, [float(r['q25']) for r in rents],
                         [float(r['q75']) for r in rents], color=BLUE, alpha=.18, label='IQR')
    axes[1].plot(observed, [float(r['mean']) for r in rents], color=BLUE, lw=2, label='Mean')
    axes[1].plot(observed, [float(r['median']) for r in rents], color=ORANGE, lw=1.8, label='Median')
    axes[1].set_ylabel('Rent\n(SGD/month)')
    axes[1].set_yticks([1500, 2500, 3500])
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    axes[1].legend(frameon=False, ncol=3, loc='lower left', fontsize=12)
    axes[1].text(datetime(2025, 11, 1), 2450, 'Test rent\nunavailable', ha='center', color=GRAY, fontsize=14)
    ticks = [datetime(y, 1, 1) for y in range(2021, 2027)] + [datetime(2026, 7, 1)]
    axes[1].set_xticks(ticks)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].tick_params(axis='x', labelsize=12)
    for ax in axes:
        ax.axvline(datetime(2025, 3, 1), ls='--', color=GRAY, lw=1)
        ax.grid(axis='y', alpha=.15)
    fig.subplots_adjust(left=.12, right=.995, top=.99, bottom=.10, hspace=.10)
    fig.savefig(OUT / 'slides_eda_rent_timeline.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    rows = read('rent_by_flat_type.csv')
    assert sum(int(r['n']) for r in rows) == 150000
    fig, ax = plt.subplots(figsize=(11.8, 4.1))
    for i, row in enumerate(rows):
        ax.hlines(i, float(row['q25']), float(row['q75']), color=BLUE, alpha=.45, lw=8)
        ax.scatter(float(row['median']), i, color=BLUE, s=55, zorder=3)
    ax.set_yticks(range(len(rows)), [f"{r['FLAT_TYPE']} (n={int(r['n']):,})" for r in rows])
    ax.set_xlabel('Monthly rent (SGD/month)')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.set_xlim(900, 4100)
    ax.set_ylim(-.6, len(rows)-.4)
    ax.grid(axis='x', alpha=.16)
    ax.text(.99, .03, 'Dot: median   Bar: P25–P75', transform=ax.transAxes,
            ha='right', color=GRAY, fontsize=13)
    fig.subplots_adjust(left=.26, right=.995, top=.98, bottom=.18)
    fig.savefig(OUT / 'slides_eda_flat_type_iqr.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('Rendered two slide figures from existing EDA aggregate CSVs.')


if __name__ == '__main__':
    main()
