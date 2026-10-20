"""Economic check for the India result: long-or-flat in the local market on the sign of the h=1 forecast."""
from evaluate import load_fc
from common import *
fc = load_fc(); fc = fc[fc.h == 1]
rows = []
for s in SERIES:
    g = fc[fc.series == s].sort_values('date')
    for m in ['Linear', 'Mean combination', 'GRU', 'XGBoost']:
        pos = (g[m] > 0).astype(float).values; r = g.y.values / 100     # next-day log return
        trades = np.abs(np.diff(np.r_[0, pos])).sum()
        gross = (pos * r).sum(); bh = r.sum(); n = len(r)
        be = (gross - bh) / trades * 1e4 if trades else np.nan          # one-way cost (bp) erasing the gain over buy-and-hold
        sr = lambda x: np.sqrt(365) * x.mean() / x.std()
        rows.append(dict(series=s, model=m, days=n, time_in_market=pos.mean(), trades=int(trades),
                         ann_ret_strategy=100 * 365 * (pos * r).mean(), ann_ret_bh=100 * 365 * r.mean(),
                         sharpe_strategy=sr(pos * r), sharpe_bh=sr(r), breakeven_cost_bp=be))
out = pd.DataFrame(rows); out.to_csv(f'{OUT}/tables/economic.csv', index=False)
print(out[out.model == 'Linear'].round(2).to_string())
