from common import *
T = os.path.join(OUT, 'tables')


def U(g):
    g = g.dropna(subset=['y'])
    return np.sqrt(((g.pred - g.y) ** 2).mean()) / np.sqrt((g.y ** 2).mean())


ab = pd.read_csv(f'{OUT}/robust_ablation.csv')
t = ab.groupby(['model', 'series', 'variant']).apply(U).unstack('variant')
order = ['Full', 'No volatility', 'No moving average', 'No volume', 'No local-market block', 'Returns only']
t = t[order]; t.to_csv(f'{T}/ablation.csv'); print(t.round(3).to_string())

lg = pd.read_csv(f'{OUT}/robust_lags.csv')
t = lg.groupby(['model', 'series', 'setting']).apply(U).unstack('setting'); t.to_csv(f'{T}/lags.csv'); print(t.round(3).to_string())

hp = pd.read_csv(f'{OUT}/robust_hyper.csv')
t = hp.groupby(['series', 'hidden', 'lr', 'L']).apply(U).unstack('series'); t.to_csv(f'{T}/hyper.csv'); print(t.round(3).to_string())

sd = pd.read_csv(f'{OUT}/robust_seeds.csv')
t = sd.groupby(['series', 'seed']).apply(U).unstack('seed')
t['mean'], t['sd'] = t.mean(1), t.iloc[:, :5].std(1); t.to_csv(f'{T}/seeds.csv'); print(t.round(4).to_string())

tr = pd.read_csv(f'{OUT}/robust_transfer.csv')
t = tr.groupby(['model', 'target', 'trained_on']).apply(U).unstack('model'); t.to_csv(f'{T}/transfer.csv'); print(t.round(3).to_string())
