"""Main rolling-origin experiment: XGBoost, LSTM, GRU, Transformer, stacking, MC dropout.
Usage: python run_main.py SERIES [SERIES ...]   -> results/fc_<series>.csv, results/shap_<series>.csv
"""
import sys, time
from common import *
from sklearn.linear_model import LinearRegression

MC = 50


def fit_block(s, f, y, tr_idx, te_idx, cols, scols, L=SEQ, kinds=('lstm', 'gru', 'trf'), seed=0,
              hidden=32, lr=1e-3, xgb_on=True):
    """Fit all base learners on tr_idx (positions), predict te_idx. Returns dict of arrays (n_te x len(H))."""
    out = {}
    Xtr, Xte = f.iloc[tr_idx][cols], f.iloc[te_idx][cols]
    if xgb_on:
        P = np.zeros((len(te_idx), len(H))); models = {}
        for j, h in enumerate(H):
            m = fit_xgb(Xtr, y.iloc[tr_idx][h], seed); P[:, j] = m.predict(Xte); models[h] = m
        out['XGBoost'] = P; out['_xgb'] = models
    sc = Scaler(f.iloc[tr_idx][scols]); F = sc(f[scols])
    ym, ys = y.iloc[tr_idx].mean().values, y.iloc[tr_idx].std().values
    Ytr = (y.iloc[tr_idx].values - ym) / ys
    Str, Ste = make_seq(F, tr_idx, L), make_seq(F, te_idx, L)
    for k in kinds:
        net = train_seq(Str, Ytr, k, hidden, lr, seed)
        name = {'lstm': 'LSTM', 'gru': 'GRU', 'trf': 'Transformer'}[k]
        out[name] = predict_seq(net, Ste) * ys + ym
        if k == 'lstm':
            out['_lstm_net'] = net; out['_Ste'] = Ste; out['_ys'] = (ym, ys)
    return out


def run(s):
    t_start = time.time()
    f, y, r = load(s)
    cols, scols = feat_cols(s), seq_cols(s)
    ok = f[cols].notna().all(1) & (np.arange(len(f)) >= SEQ + 20)
    first = f.index[ok.values][0]
    pos = pd.Series(np.arange(len(f)), index=f.index)
    rows, shap_rows, w_rows = [], [], []
    bases = ['XGBoost', 'LSTM', 'GRU', 'Transformer']
    for i, t0 in enumerate(ORIGINS):
        nxt = ORIGINS[i + 1] if i + 1 < len(ORIGINS) else END + pd.Timedelta(days=1)
        cut = t0 - pd.Timedelta(days=max(H))
        tr = pos[(f.index >= first) & (f.index <= cut)].values
        te = pos[(f.index >= t0) & (f.index < nxt)].values
        # inner split for stacking weights
        icut = cut - pd.Timedelta(days=META_WIN)
        itr = pos[(f.index >= first) & (f.index <= icut - pd.Timedelta(days=max(H)))].values
        iva = pos[(f.index > icut) & (f.index <= cut)].values
        inner = fit_block(s, f, y, itr, iva, cols, scols)
        full = fit_block(s, f, y, tr, te, cols, scols)
        yv = y.iloc[iva].values
        P = {b: full[b] for b in bases}
        P['Stacking'] = np.zeros_like(full['XGBoost']); P['Mean combination'] = np.mean([full[b] for b in bases], 0)
        for j, h in enumerate(H):
            Z = np.column_stack([inner[b][:, j] for b in bases])
            meta = LinearRegression(fit_intercept=False, positive=True).fit(Z, yv[:, j])
            P['Stacking'][:, j] = np.column_stack([full[b][:, j] for b in bases]) @ meta.coef_
            w_rows.append(dict(series=s, origin=t0.date(), h=h, **dict(zip(bases, meta.coef_.round(4)))))
        # MC dropout on the LSTM + residual scale from the inner window
        ym, ys = full['_ys']
        draws = predict_seq(full['_lstm_net'], full['_Ste'], mc=MC) * ys + ym
        res_sd = np.sqrt(((inner['LSTM'] - yv) ** 2).mean(0))
        # TreeSHAP on the XGBoost model (h = 1)
        m1 = full['_xgb'][1]
        contrib = m1.get_booster().predict(xgb.DMatrix(f.iloc[te][cols]), pred_contribs=True)[:, :-1]
        shap_rows.append(pd.Series(np.abs(contrib).mean(0), index=cols, name=str(t0.date())))
        for j, h in enumerate(H):
            for n, ix in enumerate(te):
                d = f.index[ix]
                rows.append(dict(series=s, date=d.date(), h=h, y=y.iloc[ix][h], RW=0.0,
                                 **{m: P[m][n, j] for m in P},
                                 mc_mean=draws[:, n, j].mean(), mc_sd=draws[:, n, j].std(), res_sd=res_sd[j],
                                 hist_sd=r.iloc[ix - 59:ix + 1].std() * np.sqrt(h)))
        print(f'{s} origin {t0.date()} n_train={len(tr)} n_test={len(te)}  {time.time()-t_start:.0f}s', flush=True)
    fc = pd.DataFrame(rows); fc.to_csv(f'{OUT}/fc_{s}.csv', index=False)
    pd.DataFrame(shap_rows).to_csv(f'{OUT}/shap_{s}.csv')
    pd.DataFrame(w_rows).to_csv(f'{OUT}/weights_{s}.csv', index=False)


if __name__ == '__main__':
    for s in sys.argv[1:]:
        run(s)
