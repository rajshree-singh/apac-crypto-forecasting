from common import *
from sklearn.linear_model import LinearRegression

for s in SERIES:
    f, y, r = load(s)
    cols = ['r0'] + (['r_usd', 'prem'] if s.endswith(('KRW', 'JPY', 'INR')) else [])
    ok = f[cols].notna().all(1).values
    rows = []
    for i, t0 in enumerate(ORIGINS):
        nxt = ORIGINS[i + 1] if i + 1 < len(ORIGINS) else END + pd.Timedelta(days=1)
        cut = t0 - pd.Timedelta(days=max(H))
        tr = ok & (f.index <= cut) & (np.arange(len(f)) >= SEQ + 20)
        te = (f.index >= t0) & (f.index < nxt)
        for h in H:
            m = LinearRegression().fit(f.loc[tr, cols], y.loc[tr, h])
            p = m.predict(f.loc[te, cols].fillna(0))
            for d, v in zip(f.index[te], p):
                rows.append(dict(series=s, date=d.date(), h=h, Linear=v))
    pd.DataFrame(rows).to_csv(f'{OUT}/linear_{s}.csv', index=False)
print('done')
