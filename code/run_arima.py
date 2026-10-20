"""Rolling ARIMA(p,1,q) on log prices = ARMA(p,q) with constant on daily log returns.
Order chosen once by AIC (p,q <= 2) on pre-2022 data; parameters re-estimated every 30 days on a
730-day rolling window; between re-estimations the model is re-filtered on the latest window each day.
"""
import sys
from common import *
from statsmodels.tsa.arima.model import ARIMA

WIN, REEST = 730, 30


def run(s):
    f, y, r = load(s)
    r = r.dropna()
    pre = r[r.index < TEST_START]
    best = None
    for p in range(3):
        for q in range(3):
            try:
                a = ARIMA(pre.values[-WIN:], order=(p, 0, q), trend='c').fit().aic
            except Exception:
                continue
            if best is None or a < best[0]:
                best = (a, p, q)
    _, p, q = best
    rows, res, k = [], None, 0
    test_dates = r.index[(r.index >= TEST_START) & (r.index <= END)]
    for n, d in enumerate(test_dates):
        win = r[r.index <= d].values[-WIN:]
        if n % REEST == 0:
            res = ARIMA(win, order=(p, 0, q), trend='c').fit()
        else:
            res = res.apply(win)
        fc = res.forecast(max(H))
        for h in H:
            rows.append(dict(series=s, date=d.date(), h=h, ARIMA=fc[:h].sum()))
    out = pd.DataFrame(rows); out['order'] = f'({p},1,{q})'
    out.to_csv(f'{OUT}/arima_{s}.csv', index=False); print(s, p, q, flush=True)


if __name__ == '__main__':
    for s in sys.argv[1:]:
        run(s)
