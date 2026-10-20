from common import *
import statsmodels.api as sm

rows = []
for s in [x for x in SERIES if not x.endswith('USD')]:
    f, y, r = load(s)
    # ex-post synchronous premium for description: local close of KRW/JPY venues is 15:00 UTC, so the USD
    # log price is interpolated 15/24 of the way between the USD closes of days t-1 and t.
    coin, cur = s.split('-')
    P = pd.read_csv(f'{PANEL}/{s}.csv', index_col=0, parse_dates=True)
    U = np.log(pd.read_csv(f'{PANEL}/{coin}-USD.csv', index_col=0, parse_dates=True).close).reindex(P.index)
    fx = pd.read_csv(f'{ROOT}/apdata/fred_fx.csv', parse_dates=['observation_date']).set_index('observation_date')
    fx = fx.reindex(pd.date_range('2017-11-01', END)).ffill()[{'KRW': 'DEXKOUS', 'JPY': 'DEXJPUS', 'INR': 'DEXINUS'}[cur]].reindex(P.index)
    u_sync = 9 / 24 * U.shift(1) + 15 / 24 * U if cur in ('KRW', 'JPY') else U
    sync = 100 * (np.log(P.close) - np.log(fx) - u_sync)
    sync.to_csv(f'{OUT}/sync_premium_{s}.csv')
    d = pd.DataFrame({'prem': f.prem, 'sync': sync, 'r_usd': f.r_usd, 'r_loc': r, 'y1': y[1]}).dropna()
    for per, dd in [('Full sample', d), ('Test period', d[d.index >= TEST_START])]:
        p = dd.sync
        ar = sm.OLS(p.values[1:], sm.add_constant(p.values[:-1])).fit().params[1]
        hl = np.log(0.5) / np.log(ar) if 0 < ar < 1 else np.nan
        X = sm.add_constant(dd[['prem', 'r_usd', 'r_loc']])
        reg = sm.OLS(dd.y1, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        rows.append(dict(series=s, period=per, start=dd.index.min().date(), n=len(dd),
                         mean=p.mean(), sd=p.std(), pct_pos=100 * (p > 0).mean(), p5=p.quantile(.05),
                         p95=p.quantile(.95), ar1=ar, half_life_days=hl,
                         b_prem=reg.params['prem'], t_prem=reg.tvalues['prem'],
                         b_rusd=reg.params['r_usd'], t_rusd=reg.tvalues['r_usd'],
                         b_rloc=reg.params['r_loc'], t_rloc=reg.tvalues['r_loc'], r2=reg.rsquared))
out = pd.DataFrame(rows); out.to_csv(f'{OUT}/premium.csv', index=False)
pd.set_option('display.width', 250); print(out.round(3).to_string())
