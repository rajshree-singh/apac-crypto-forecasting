from common import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

FIG = os.path.join(ROOT, 'figs'); os.makedirs(FIG, exist_ok=True)
T = os.path.join(OUT, 'tables')
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, MUTED, GRID, BASE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8.5, 'axes.edgecolor': BASE,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'savefig.facecolor': 'white'})


def save(fig, n):
    for ext, kw in [('eps', {}), ('tif', {'dpi': 600, 'pil_kwargs': {'compression': 'tiff_lzw'}}), ('png', {'dpi': 300})]:
        fig.savefig(f'{FIG}/Fig{n}.{ext}', bbox_inches='tight', **kw)
    plt.close(fig)


def fig1():
    fig, ax = plt.subplots(figsize=(6.8, 3.3)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    def box(x, y, w, h, txt, fc='#f0efec'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=BASE, lw=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=6.5, color=INK, linespacing=1.25)
    def arrow(x1, y1, x2, y2):
        ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.8, mutation_scale=8))
    box(0.1, 3.55, 2.2, 1.3, 'Global market\nBTC, ETH, LTC in USD\n(aggregate price)')
    box(0.1, 1.95, 2.2, 1.3, 'Asia-Pacific venues\nUpbit (KRW), bitbank (JPY),\nWazirX (INR)')
    box(0.1, 0.35, 2.2, 1.3, 'FX rates (FRED)\nKRW, JPY, INR per USD\nlocal premium')
    box(2.9, 1.95, 2.0, 2.9, 'Features known at the\nlocal close of day t\n\nlagged returns\nmoving-average gaps\nvolatility, range\nvolume\n\nlocal block:\npremium, Δpremium,\nlatest USD return')
    box(5.35, 3.75, 2.3, 1.1, 'Benchmarks\nrandom walk, rolling ARIMA,\nlinear spillover')
    box(5.35, 2.35, 2.3, 1.1, 'Learners\nXGBoost, LSTM, GRU,\nTransformer')
    box(5.35, 0.95, 2.3, 1.1, 'Combinations\nstacking (non-negative),\nequal-weight mean')
    box(8.0, 2.75, 1.98, 2.1, 'Evaluation\n2022-01-01 to\n2024-02-14\nh = 1, 3, 7 days\nRMSE, Theil U, DM,\ndirection, intervals')
    box(8.0, 0.35, 1.98, 2.1, 'Diagnostics\nSHAP, ablation,\nlook-back, tuning,\nseeds, transfer')
    for y in (4.2, 2.6, 1.0):
        arrow(2.3, y, 2.9, min(max(y, 2.2), 4.5))
    for y in (4.3, 2.9, 1.5):
        arrow(4.9, 3.4, 5.35, y)
    arrow(7.65, 4.3, 8.0, 4.0); arrow(7.65, 2.9, 8.0, 3.6); arrow(7.65, 1.5, 8.0, 3.2)
    arrow(7.65, 2.6, 8.0, 1.5)
    ax.text(5.0, 0.05, 'Rolling origins every six months; every model re-estimated using data available before each origin',
            ha='center', fontsize=7, color=INK2)
    save(fig, 1)


def fig2():
    fig, axs = plt.subplots(3, 1, figsize=(6.8, 4.6), sharex=True)
    for ax, (s, lab) in zip(axs, [('BTC-KRW', 'Korea, Upbit'), ('BTC-JPY', 'Japan, bitbank'), ('BTC-INR', 'India, WazirX')]):
        p = pd.read_csv(f'{OUT}/sync_premium_{s}.csv', index_col=0, parse_dates=True).iloc[:, 0]
        ax.plot(p.index, p, color='#b7d3f6', lw=0.5)
        ax.plot(p.index, p.rolling(30).mean(), color=C[0], lw=1.6)
        ax.axhline(0, color=BASE, lw=0.8); ax.axvline(TEST_START, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.set_ylim(-12, 16); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_ylabel('Premium (%)')
        ax.text(0.01, 0.92, f'{lab}: Bitcoin', transform=ax.transAxes, fontsize=8, color=INK, va='top')
    axs[2].text(TEST_START, -11, ' test period', fontsize=7, color=INK2, va='bottom')
    axs[0].plot([], [], color='#b7d3f6', lw=0.8, label='daily'); axs[0].plot([], [], color=C[0], lw=1.6, label='30-day mean')
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), fontsize=7, ncol=2)
    save(fig, 2)


def fig3():
    a = pd.read_csv(f'{T}/accuracy_all.csv'); a = a[(a.h == 1)]
    models = ['ARIMA', 'Linear', 'XGBoost', 'LSTM', 'GRU', 'Transformer', 'Stacking', 'Mean combination']
    U = a.pivot(index='series', columns='model', values='U').reindex(SERIES)[models]
    P = a.pivot(index='series', columns='model', values='DM_p').reindex(SERIES)[models]
    cmap = LinearSegmentedColormap.from_list('d', ['#1c5cab', '#86b6ef', '#f0efec', '#f2a27e', '#c2461c'])
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    im = ax.imshow(U.values, cmap=cmap, norm=TwoSlopeNorm(1.0, 0.94, 1.06), aspect='auto')
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            v, p = U.values[i, j], P.values[i, j]
            star = '**' if p < .05 else ('*' if p < .10 else '')
            ax.text(j, i, f'{v:.3f}{star}', ha='center', va='center', fontsize=6.8,
                    color='white' if abs(v - 1) > 0.045 else INK)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([m.replace(' combination', '\ncombination') for m in models], fontsize=7.2)
    ax.set_yticks(range(len(SERIES))); ax.set_yticklabels(SERIES, fontsize=7.4)
    for y in (2.5, 4.5, 7.5):
        ax.axhline(y, color='white', lw=2)
    ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Theil's U (RMSE relative to random walk)", fontsize=7.4)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    save(fig, 3)


def fig4():
    pool = pd.read_csv(f'{T}/U_by_market_h.csv')
    models = ['Linear', 'XGBoost', 'LSTM', 'Stacking']
    fig, axs = plt.subplots(1, 4, figsize=(6.8, 2.5), sharey=True)
    for ax, mk in zip(axs, ['Global (USD)', 'Korea (KRW)', 'Japan (JPY)', 'India (INR)']):
        g = pool[pool.market == mk].set_index('h')
        ax.axhline(1, color=INK2, lw=0.9, ls=(0, (3, 2)))
        for k, m in enumerate(models):
            ax.plot(g.index, g[m], color=C[k], lw=1.6, marker='os^D'[k], ms=4.5, mec='white', mew=0.8, label=m)
        ax.set_xticks(H); ax.set_xlabel('Horizon (days)'); ax.set_title(mk, fontsize=8, color=INK)
        ax.grid(axis='y', color=GRID, lw=0.5)
    axs[0].set_ylabel("Mean Theil's U"); axs[3].text(7.2, 1.003, 'RW = 1', fontsize=6.8, color=INK2, va='bottom', ha='right')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=7.4)
    save(fig, 4)


def fig5():
    iv = pd.read_csv(f'{T}/intervals_summary.csv')
    meths = ['MC dropout only', 'MC dropout + residual', 'RW + 60-day volatility']
    fig, ax = plt.subplots(figsize=(5.2, 2.6)); w = 0.26
    for k, m in enumerate(meths):
        g = iv[iv.method == m].set_index('h').reindex(H)
        x = np.arange(3) + (k - 1) * (w + 0.02)
        bars = ax.bar(x, 100 * g.coverage, w, color=C[k], label=m, edgecolor='white', linewidth=1)
        for xi, v in zip(x, 100 * g.coverage):
            ax.text(xi, v + 1.5, f'{v:.0f}', ha='center', fontsize=6.8, color=INK2)
    ax.axhline(90, color=INK2, lw=0.9, ls=(0, (3, 2))); ax.text(-0.5, 97, 'nominal 90%', fontsize=6.8, color=INK2, ha='left', va='bottom')
    ax.set_xticks(range(3)); ax.set_xticklabels([f'h = {h}' for h in H]); ax.set_ylabel('Empirical coverage (%)')
    ax.set_ylim(0, 110); ax.grid(axis='y', color=GRID, lw=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=7)
    save(fig, 5)


def fig6():
    sh = pd.read_csv(f'{T}/shap_groups.csv').set_index('series').reindex(SERIES[::-1]).fillna(0)
    groups = [('ret', 'Lagged returns'), ('ma', 'Moving-average gaps'), ('vol', 'Volatility and range'),
              ('volume', 'Volume'), ('local', 'Local-market block')]
    fig, ax = plt.subplots(figsize=(6.8, 3.4)); left = np.zeros(len(sh))
    for k, (g, lab) in enumerate(groups):
        v = sh[g].values
        ax.barh(range(len(sh)), v, left=left, color=C[k], edgecolor='white', linewidth=1.2, height=0.72, label=lab)
        for i, (l, x) in enumerate(zip(left, v)):
            if x >= 9:
                ax.text(l + x / 2, i, f'{x:.0f}', ha='center', va='center', fontsize=6.6, color=INK)
        left += v
    ax.set_yticks(range(len(sh))); ax.set_yticklabels(sh.index, fontsize=7.4)
    ax.set_xlim(0, 100); ax.set_xlabel('Share of mean |SHAP| (%), XGBoost, h = 1')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.13), ncol=5, fontsize=6.8, handlelength=1.2)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    save(fig, 6)


if __name__ == '__main__':
    # manuscript numbering: Fig1 design, Fig2 premium, Fig3 horizons, Fig4 intervals, Fig5 SHAP
    # (fig3 = U heatmap is an optional extra, saved as FigS1)
    fig1(); fig2()
    for fn, n in [(fig4, 3), (fig5, 4), (fig6, 5), (fig3, 'S1')]:
        _save = globals()['save']
        globals()['save'] = lambda fig, _n, n=n: _save(fig, n)
        fn(); globals()['save'] = _save
    print('ok')
