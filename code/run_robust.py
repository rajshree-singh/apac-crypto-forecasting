"""Robustness experiments (one-day horizon unless stated), same rolling origins as the main run.
  ablation  : feature groups removed (XGBoost all series; LSTM on the four BTC series)
  lags      : look-back length (LSTM: 5/10/20/30 days; XGBoost: 1/5/10/20 return lags)
  hyper     : LSTM hidden units x learning rate x look-back (BTC-USD, BTC-KRW)
  seeds     : LSTM, five random seeds (BTC-USD, BTC-KRW)
  transfer  : models trained on BTC-USD applied without retraining to other coins / local markets
Writes results/robust_<part>.csv with one row per forecast.
"""
import sys
from common import *

BTC4 = ['BTC-USD', 'BTC-KRW', 'BTC-JPY', 'BTC-INR']


def extra_lags(f, r, n):
    f = f.copy()
    for k in range(5, n):
        f[f'r{k}'] = r.shift(k)
    return f


def forecasts(s_tr, s_te, model, drop=(), L=SEQ, hidden=32, lr=1e-3, seed=0, nlags=5):
    ftr, ytr, rtr = load(s_tr); fte, yte, rte = load(s_te)
    if nlags > 5:
        ftr, fte = extra_lags(ftr, rtr, nlags), extra_lags(fte, rte, nlags)
    same = s_tr == s_te
    base = [c for c in feat_cols(s_tr, drop) if not (c.startswith('r') and c[1:].isdigit())]
    lagc = [f'r{k}' for k in range(nlags)] if 'ret' not in drop else []
    cols = lagc + base
    if not same:                          # transfer: only features both series have
        cols = [c for c in cols if c in feat_cols(s_te) or c in lagc]
    scols = seq_cols(s_tr, drop) if same else [c for c in seq_cols(s_tr, drop) if c in seq_cols(s_te, drop)]
    out = []
    first_tr = ftr.index[(ftr[cols].notna().all(1)).values & (np.arange(len(ftr)) >= max(L, nlags) + 20)][0]
    for i, t0 in enumerate(ORIGINS):
        nxt = ORIGINS[i + 1] if i + 1 < len(ORIGINS) else END + pd.Timedelta(days=1)
        cut = t0 - pd.Timedelta(days=max(H))
        mtr = (ftr.index >= first_tr) & (ftr.index <= cut)
        ptr = np.where(mtr)[0]
        pte = np.where((fte.index >= t0) & (fte.index < nxt))[0]
        if model == 'XGBoost':
            m = fit_xgb(ftr.iloc[ptr][cols], ytr.iloc[ptr][1], seed)
            p = m.predict(fte.iloc[pte][cols])
        else:
            sc_tr = Scaler(ftr.iloc[ptr][scols])
            ym, ys = ytr.iloc[ptr].mean().values, ytr.iloc[ptr].std().values
            net = train_seq(make_seq(sc_tr(ftr[scols]), ptr, L), (ytr.iloc[ptr].values - ym) / ys,
                            'lstm', hidden, lr, seed)
            if same:
                sc_te, (ym2, ys2) = sc_tr, (ym, ys)
            else:                          # target's own pre-origin statistics, no test information
                pre = (fte.index <= cut) & fte[scols].notna().all(1).values
                sc_te = Scaler(fte.loc[pre, scols])
                ym2, ys2 = yte.loc[pre].mean().values, yte.loc[pre].std().values
            p = (predict_seq(net, make_seq(sc_te(fte[scols]), pte, L)) * ys2 + ym2)[:, 0]
        for n, ix in enumerate(pte):
            out.append(dict(date=fte.index[ix].date(), y=yte.iloc[ix][1], pred=p[n]))
    return pd.DataFrame(out)


def tag(df, **kw):
    for k, v in kw.items():
        df[k] = v
    return df


def part(name):
    res = []
    if name == 'ablation':
        variants = {'Full': (), 'No volatility': ('vol',), 'No moving average': ('ma',),
                    'No volume': ('volume',), 'No local-market block': ('local',),
                    'Returns only': ('ma', 'vol', 'volume', 'local')}
        for s in SERIES:
            for v, dr in variants.items():
                if v == 'No local-market block' and s.endswith('USD'):
                    continue
                res.append(tag(forecasts(s, s, 'XGBoost', dr), series=s, model='XGBoost', variant=v))
                if s in BTC4:
                    res.append(tag(forecasts(s, s, 'LSTM', dr), series=s, model='LSTM', variant=v))
                print(name, s, v, flush=True)
    if name == 'lags':
        for s in BTC4:
            for L in [5, 10, 20, 30]:
                res.append(tag(forecasts(s, s, 'LSTM', L=L), series=s, model='LSTM', setting=L))
            for nl in [1, 5, 10, 20]:
                res.append(tag(forecasts(s, s, 'XGBoost', nlags=nl), series=s, model='XGBoost', setting=nl))
            print(name, s, flush=True)
    if name == 'hyper':
        for s in ['BTC-USD', 'BTC-KRW']:
            for hd in [32, 64]:
                for lr in [1e-3, 5e-4]:
                    for L in [20, 30]:
                        res.append(tag(forecasts(s, s, 'LSTM', L=L, hidden=hd, lr=lr), series=s,
                                       hidden=hd, lr=lr, L=L))
            print(name, s, flush=True)
    if name == 'seeds':
        for s in ['BTC-USD', 'BTC-KRW']:
            for sd in range(5):
                res.append(tag(forecasts(s, s, 'LSTM', seed=sd), series=s, seed=sd))
            print(name, s, flush=True)
    if name == 'transfer':
        for tgt in ['BTC-USD', 'ETH-USD', 'LTC-USD', 'BTC-KRW', 'BTC-JPY', 'BTC-INR']:
            for m in ['XGBoost', 'LSTM']:
                res.append(tag(forecasts('BTC-USD', tgt, m, drop=('local',)), target=tgt, model=m, trained_on='BTC-USD'))
                if tgt != 'BTC-USD':
                    res.append(tag(forecasts(tgt, tgt, m, drop=('local',)), target=tgt, model=m, trained_on=tgt))
            print(name, tgt, flush=True)
    pd.concat(res).to_csv(f'{OUT}/robust_{name}.csv', index=False)


if __name__ == '__main__':
    for p in sys.argv[1:]:
        part(p)
