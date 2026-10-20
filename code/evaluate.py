from common import *
from scipy import stats

T = os.path.join(OUT, 'tables'); os.makedirs(T, exist_ok=True)
MODELS = ['RW', 'ARIMA', 'Linear', 'XGBoost', 'LSTM', 'GRU', 'Transformer', 'Stacking', 'Mean combination']
MARKET = {'USD': 'Global (USD)', 'KRW': 'Korea (KRW)', 'JPY': 'Japan (JPY)', 'INR': 'India (INR)'}


def dm(e1, e2, h):
    """Diebold-Mariano with Harvey-Leybourne-Newbold correction, squared loss. >0 means model 2 better."""
    d = e1 ** 2 - e2 ** 2; n = len(d); dbar = d.mean()
    g = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    v = (g[0] + 2 * sum(g[1:])) / n
    if v <= 0:
        return np.nan, np.nan
    st = dbar / np.sqrt(v) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return st, 2 * stats.t.sf(abs(st), n - 1)


def load_fc():
    out = []
    for s in SERIES:
        fc = pd.read_csv(f'{OUT}/fc_{s}.csv', parse_dates=['date'])
        ar = pd.read_csv(f'{OUT}/arima_{s}.csv', parse_dates=['date'])
        li = pd.read_csv(f'{OUT}/linear_{s}.csv', parse_dates=['date'])
        out.append(fc.merge(ar[['date', 'h', 'ARIMA', 'order']], on=['date', 'h'], how='left')
                   .merge(li[['date', 'h', 'Linear']], on=['date', 'h'], how='left'))
    return pd.concat(out).dropna(subset=['y'])


def main():
    fc = load_fc(); fc['market'] = fc.series.str[-3:].map(MARKET)
    rows = []
    for (s, h), g in fc.groupby(['series', 'h']):
        e_rw = g.y.values
        for m in MODELS:
            e = (g.y - g[m]).values
            st, p = dm(e_rw, e, h) if m != 'RW' else (np.nan, np.nan)
            nz = (g.y != 0) & (g[m] != 0)      # a zero forecast makes no directional call
            da = np.nan if m == 'RW' else (np.sign(g[m][nz]) == np.sign(g.y[nz])).mean()
            rows.append(dict(series=s, market=g.market.iloc[0], h=h, model=m, n=len(g), n_calls=int(nz.sum()),
                             RMSE=np.sqrt((e ** 2).mean()), MAE=np.abs(e).mean(),
                             U=np.sqrt((e ** 2).mean()) / np.sqrt((e_rw ** 2).mean()),
                             DA=da, DA_p=np.nan if m == 'RW' else stats.binomtest(int(round(da * nz.sum())), int(nz.sum()), 0.5).pvalue,
                             DM=st, DM_p=p))
    acc = pd.DataFrame(rows); acc.to_csv(f'{T}/accuracy_all.csv', index=False)

    # Table: RMSE at h=1 by series and model, with DM significance flag
    a1 = acc[acc.h == 1].copy()
    def cell(r):
        star = '' if r.model == 'RW' or np.isnan(r.DM_p) else ('**' if r.DM_p < .05 else ('*' if r.DM_p < .10 else ''))
        sign = '' if star == '' else ('+' if r.DM > 0 else '−')
        return f'{r.RMSE:.3f}{sign}{star}'
    a1['cell'] = a1.apply(cell, axis=1)
    piv = a1.pivot(index='series', columns='model', values='cell').reindex(SERIES)[MODELS]
    piv.to_csv(f'{T}/rmse_h1.csv')
    # pooled Theil's U by market and horizon
    pool = acc.groupby(['market', 'h', 'model']).U.mean().unstack('model')[MODELS]
    pool.to_csv(f'{T}/U_by_market_h.csv')
    allU = acc.groupby(['h', 'model']).U.agg(['mean', 'median', 'min', 'max']).unstack('h')
    allU.to_csv(f'{T}/U_overall.csv')
    sig = acc[acc.model != 'RW'].groupby(['h', 'model']).apply(
        lambda g: pd.Series({'beats_RW_5pct': int(((g.DM_p < .05) & (g.DM > 0)).sum()),
                             'worse_RW_5pct': int(((g.DM_p < .05) & (g.DM < 0)).sum()),
                             'U<1': int((g.U < 1).sum())}))
    sig.to_csv(f'{T}/dm_counts.csv')
    da = acc[(acc.h == 1) & (acc.model != 'RW')].pivot(index='series', columns='model', values='DA').reindex(SERIES)
    da.to_csv(f'{T}/da_h1.csv')
    dap = acc[(acc.h == 1) & (acc.model != 'RW')].pivot(index='series', columns='model', values='DA_p').reindex(SERIES)
    dap.to_csv(f'{T}/da_p_h1.csv')

    # stacking weights
    w = pd.concat([pd.read_csv(f'{OUT}/weights_{s}.csv') for s in SERIES])
    w.groupby('h')[['XGBoost', 'LSTM', 'GRU', 'Transformer']].mean().assign(
        sum=lambda d: d.sum(1)).to_csv(f'{T}/stack_weights.csv')
    w.assign(tot=w[['XGBoost', 'LSTM', 'GRU', 'Transformer']].sum(1)).groupby('h').tot.describe().to_csv(f'{T}/stack_weight_total.csv')

    # interval forecasts (90%) from LSTM Monte Carlo dropout
    z = stats.norm.ppf(0.95); iv = []
    for (s, h), g in fc.groupby(['series', 'h']):
        for name, c, sd in [('MC dropout only', g.mc_mean, g.mc_sd),
                            ('MC dropout + residual', g.mc_mean, np.sqrt(g.mc_sd ** 2 + g.res_sd ** 2)),
                            ('RW + 60-day volatility', 0 * g.y, g.hist_sd)]:
            lo, hi = c - z * sd, c + z * sd
            cov = ((g.y >= lo) & (g.y <= hi)).mean()
            wid = (hi - lo).mean()
            wink = (hi - lo + 2 / 0.10 * ((lo - g.y).clip(lower=0) + (g.y - hi).clip(lower=0))).mean()
            n = len(g); x = int(round(cov * n))
            # Kupiec unconditional coverage LR test
            pi = x / n
            lr = -2 * ((n - x) * np.log(0.1) + x * np.log(0.9) - ((n - x) * np.log(max(1 - pi, 1e-12)) + x * np.log(max(pi, 1e-12))))
            iv.append(dict(series=s, market=g.market.iloc[0], h=h, method=name, coverage=cov, width=wid,
                           interval_score=wink, kupiec_p=stats.chi2.sf(lr, 1)))
    iv = pd.DataFrame(iv); iv.to_csv(f'{T}/intervals_all.csv', index=False)
    iv.groupby(['h', 'method'])[['coverage', 'width', 'interval_score']].mean().to_csv(f'{T}/intervals_summary.csv')
    iv.groupby(['h', 'method']).apply(lambda g: int((g.kupiec_p < .05).sum())).to_csv(f'{T}/intervals_kupiec_rejections.csv')

    # SHAP group shares
    sh = []
    for s in SERIES:
        d = pd.read_csv(f'{OUT}/shap_{s}.csv', index_col=0).mean()
        tot = d.sum()
        rowd = {'series': s}
        for g, c in GROUPS.items():
            cc = [x for x in c if x in d.index]
            rowd[g] = 100 * d[cc].sum() / tot if cc else np.nan
        rowd['top_feature'] = d.idxmax(); sh.append(rowd)
    pd.DataFrame(sh).to_csv(f'{T}/shap_groups.csv', index=False)
    print(piv.to_string()); print(pool.round(3).to_string()); print(sig.to_string())


if __name__ == '__main__':
    main()
