"""Shared data, features, splits and models for the APFM revision.

Everything is forecast in daily log-return units (percent), so errors are comparable
across coins and currencies. Forecast origin t uses only information available at the
local close of day t.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, xgboost as xgb, os, warnings
warnings.filterwarnings('ignore')
torch.set_num_threads(2)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, 'panel')
OUT = os.path.join(ROOT, 'results'); os.makedirs(OUT, exist_ok=True)
SERIES = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'BTC-KRW', 'ETH-KRW', 'BTC-JPY', 'ETH-JPY', 'LTC-JPY',
          'BTC-INR', 'ETH-INR', 'LTC-INR']
H = [1, 3, 7]
TEST_START = pd.Timestamp('2022-01-01'); END = pd.Timestamp('2024-02-14')
ORIGINS = pd.to_datetime(['2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01', '2024-01-01'])
META_WIN = 180          # days of inner out-of-sample predictions used to fit the stacking weights
SEQ = 20                # default look-back for sequence models
GROUPS = {
    'ret': ['r0', 'r1', 'r2', 'r3', 'r4'],
    'ma': ['ma5', 'ma20'],
    'vol': ['vol5', 'vol20', 'range'],
    'volume': ['lvol'],
    'local': ['prem', 'dprem', 'r_usd'],     # Asia-Pacific series only
}
SEQ_BASE = ['r0', 'ma5', 'ma20', 'vol5', 'vol20', 'range', 'lvol']   # per-day inputs for sequence models


def load(s):
    d = pd.read_csv(os.path.join(PANEL, f'{s}.csv'), index_col=0, parse_dates=True)
    f = pd.DataFrame(index=d.index)
    r = 100 * np.log(d.close).diff()
    for k in range(5):
        f[f'r{k}'] = r.shift(k)
    f['ma5'] = 100 * np.log(d.close / d.close.rolling(5).mean())
    f['ma20'] = 100 * np.log(d.close / d.close.rolling(20).mean())
    f['vol5'] = r.rolling(5).std(); f['vol20'] = r.rolling(20).std()
    f['range'] = 100 * np.log(d.high / d.low)
    v = np.log1p(d.volume.clip(lower=0)); f['lvol'] = v - v.rolling(20).mean()
    if 'premium' in d:
        coin, cur = s.split('-')
        u = pd.read_csv(os.path.join(PANEL, f'{coin}-USD.csv'), index_col=0, parse_dates=True)
        ru = 100 * np.log(u.close).diff().reindex(d.index)
        lag = 1 if cur in ('KRW', 'JPY') else 0      # latest USD close known at the local close
        f['prem'] = d.premium; f['dprem'] = d.premium.diff(); f['r_usd'] = ru.shift(lag)
    y = pd.DataFrame({h: r.shift(-1).rolling(h).sum().shift(-(h - 1)) for h in H}, index=d.index)
    return f, y, r


def feat_cols(s, drop=()):
    cols = []
    for g, c in GROUPS.items():
        if g == 'local' and not s.endswith(('KRW', 'JPY', 'INR')):
            continue
        if g in drop:
            continue
        cols += c
    return cols


def seq_cols(s, drop=()):
    m = {'ma': ['ma5', 'ma20'], 'vol': ['vol5', 'vol20', 'range'], 'volume': ['lvol']}
    cols = [c for c in SEQ_BASE if not any(c in m[g] for g in drop if g in m)]
    if s.endswith(('KRW', 'JPY', 'INR')) and 'local' not in drop:
        cols += ['prem', 'dprem', 'r_usd']
    return cols


def windows(t0, nxt):
    """train index: origins whose 7-day target is known before t0; test: [t0, nxt)."""
    return t0 - pd.Timedelta(days=max(H)), nxt


# ---------------------------------------------------------------- tree model
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=2, verbosity=0)


def fit_xgb(X, y, seed=0):
    m = xgb.XGBRegressor(random_state=seed, **XGB_PARAMS); m.fit(X, y); return m


# ---------------------------------------------------------------- sequence models
class RNN(nn.Module):
    def __init__(self, k, hidden=32, cell='lstm', p=0.2):
        super().__init__()
        self.rnn = (nn.LSTM if cell == 'lstm' else nn.GRU)(k, hidden, batch_first=True)
        self.drop = nn.Dropout(p); self.out = nn.Linear(hidden, len(H))

    def forward(self, x):
        h, _ = self.rnn(x); return self.out(self.drop(h[:, -1]))


class TRF(nn.Module):
    def __init__(self, k, d=32, heads=2, p=0.2, L=SEQ):
        super().__init__()
        self.inp = nn.Linear(k, d); self.pos = nn.Parameter(torch.zeros(1, 64, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, p, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 1); self.drop = nn.Dropout(p); self.out = nn.Linear(d, len(H))

    def forward(self, x):
        z = self.inp(x) + self.pos[:, :x.shape[1]]
        return self.out(self.drop(self.enc(z)[:, -1]))


def make_seq(F, idx, L):
    """F: standardized feature DataFrame (full history); idx: positions of origins."""
    a = F.values.astype(np.float32)
    return np.stack([a[i - L + 1:i + 1] for i in idx])


def train_seq(Xs, Y, kind='lstm', hidden=32, lr=1e-3, seed=0, max_ep=60, patience=6):
    torch.manual_seed(seed); np.random.seed(seed)
    k = Xs.shape[2]
    net = TRF(k, hidden) if kind == 'trf' else RNN(k, hidden, kind)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Xs); nv = max(60, int(0.15 * n))
    Xt, Yt = torch.tensor(Xs[:-nv]), torch.tensor(Y[:-nv], dtype=torch.float32)
    Xv, Yv = torch.tensor(Xs[-nv:]), torch.tensor(Y[-nv:], dtype=torch.float32)
    best, state, bad = 1e9, None, 0
    for ep in range(max_ep):
        net.train(); perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 64):
            j = perm[b:b + 64]; opt.zero_grad()
            loss = ((net(Xt[j]) - Yt[j]) ** 2).mean(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            v = ((net(Xv) - Yv) ** 2).mean().item()
        if v < best - 1e-5:
            best, state, bad = v, {kk: vv.clone() for kk, vv in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state); net.eval(); return net


def predict_seq(net, Xs, mc=0):
    X = torch.tensor(Xs)
    if mc:
        net.train()           # keep dropout active (Monte Carlo dropout)
        with torch.no_grad():
            draws = np.stack([net(X).numpy() for _ in range(mc)])
        net.eval(); return draws
    with torch.no_grad():
        return net(X).numpy()


class Scaler:
    def __init__(self, df):
        self.m = df.mean(); self.s = df.std().replace(0, 1)

    def __call__(self, df):
        return ((df - self.m) / self.s).fillna(0)
