"""Train gold_scar's direction model with the repo's own feature code (autotrader.options.ml.features).

Needs torch (CUDA), scikit-learn and the data from download.py (data/research/gold_scar/). Writes
walk-forward OOS predictions from 2019, the last three years' fold weights for honest replays (folds/), and the shipped
weights -> autotrader/options/ml/gold_scar.npz (ridge on all data; transformers trained through last year,
early-stopped on this year).

    uv run --with torch --with scikit-learn python research/gold_scar/train.py [seeds=3]
"""
import sys, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.linear_model import Ridge
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from autotrader.options.ml import features as F
M = REPO / 'data' / 'research' / 'gold_scar'
OUT = REPO / 'autotrader' / 'options' / 'ml' / 'gold_scar.npz'
KS = [2, 5, 8, 11, 17, 23, 29, 35, 41, 47, 53, 59]
EXIT = 70
dev = 'cuda'
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def raw(sym):
    df = pd.read_parquet(M / 'bars' / f'{sym}.parquet')
    return df.rename(columns=dict(o='open', h='high', l='low', c='close', v='volume'))


def load():
    f = M / 'prod_arrays.pkl'
    if f.exists():
        return pickle.loads(f.read_bytes())
    B = {s: raw(s) for s in F.SYMBOLS}
    days, _ = F.sessions(B['SPY'])
    A = {s: F.sessions(B[s], days)[1] for s in F.SYMBOLS}
    # VWAP of each bar (Alpaca) for the realistic entry/exit prices of the target
    VW = {}
    for u in F.SYMBOLS:
        df = B[u].copy(); df['close'] = df['vw']
        VW[u] = F.sessions(df, days)[1]['c']
    full = np.ones(len(days), bool)
    for u in F.TARGETS:  # whole regular sessions only (drops half days and broken data)
        et = B[u].index.tz_convert(F.ET); m = et.hour * 60 + et.minute
        cnt = pd.Series(1, index=pd.DatetimeIndex(et.date)[(m >= 570) & (m < 960)]).groupby(level=0).size()
        full &= cnt.reindex(days).fillna(0).to_numpy() >= 70
    out = (days, A, VW, full)
    f.write_bytes(pickle.dumps(out))
    return out


days, A, VW, full = load()
def prior(name):
    d = pd.read_csv(M / 'cboe' / f'{name}.csv'); d.columns = [c.upper() for c in d.columns]
    d['DATE'] = pd.to_datetime(d['DATE']); col = 'CLOSE' if 'CLOSE' in d else d.columns[-1]
    s = d.set_index('DATE')[col].astype(float).sort_index()
    i = s.index.searchsorted(days, side='left') - 1  # last close strictly before each session
    return np.where(i >= 0, s.to_numpy()[np.maximum(i, 0)], np.nan)


vix = prior('VIX')
sig = F.expected_move(vix)
nd = len(days)
years = days.year.to_numpy()
LAST = int(years[-1])
ok = full & np.r_[False, full[:-1]] & np.isfinite(vix)  # today and yesterday complete
print('days', nd, 'usable', ok.sum(), days[0].date(), days[-1].date(), flush=True)

# targets: next-bar VWAP -> 15:20-15:25 VWAP, in expected-move units for the remaining time
T = np.full((nd, F.SLOTS, len(F.SYMBOLS)), np.nan, np.float32)
for j, u in enumerate(F.SYMBOLS):
    for k in range(60):
        T[:, k, j] = np.log(VW[u][:, EXIT] / VW[u][:, k + 1]) / (sig * np.sqrt((EXIT - k) / 78))
T = np.clip(T, -4, 4)

# ---------------- ridge ----------------
rows = []
for k in KS:
    for j, u in enumerate(F.TARGETS):
        X = F.ridge_features(A, u, k, sig)
        rows.append((np.where(ok)[0], k, j, X[ok], T[ok, k, j]))
RI = np.concatenate([r[0] for r in rows]); RK = np.concatenate([np.full(len(r[0]), r[1]) for r in rows])
RJ = np.concatenate([np.full(len(r[0]), r[2]) for r in rows])
RX = np.concatenate([r[3] for r in rows]); RY = np.concatenate([r[4] for r in rows])
fin = np.isfinite(RY)
RI, RK, RJ, RX, RY = RI[fin], RK[fin], RJ[fin], RX[fin], RY[fin]
RYc = np.clip(RY, -3, 3)
ry = years[RI]
ridge_oos = np.full(len(RY), np.nan)


def fit_ridge(m):
    mu, sd = RX[m].mean(0), RX[m].std(0) + 1e-9
    Z = np.clip((RX[m] - mu) / sd, -6, 6)
    r = Ridge(1000.0).fit(Z, RYc[m])
    return mu, sd, r


for Y in range(2019, LAST + 1):
    mu, sd, r = fit_ridge(ry < Y)
    te = ry == Y
    ridge_oos[te] = r.predict(np.clip((RX[te] - mu) / sd, -6, 6))
te = ~np.isnan(ridge_oos)
print('ridge OOS IC', np.corrcoef(ridge_oos[te], RY[te])[0, 1].round(4),
      {Y: np.corrcoef(ridge_oos[ry == Y], RY[ry == Y])[0, 1].round(3) for Y in range(2019, LAST + 1)}, flush=True)
mu, sd, rfin = fit_ridge(np.ones(len(RY), bool))

# ---------------- transformer ----------------
RAW = F.bar_raw(A)
ST = F.static_inputs(days, vix)
torch.backends.cuda.matmul.allow_tf32 = True


class Net(nn.Module):
    def __init__(s, d=64, L=3, H=4, p=0.2):
        super().__init__()
        s.inp = nn.Linear(F.BAR_DIM + F.STATIC_DIM, d)
        s.pos = nn.Parameter(torch.zeros(2 * F.SLOTS, d))
        layer = nn.TransformerEncoderLayer(d, H, 4 * d, p, batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        s.out = nn.Linear(d, len(F.SYMBOLS))
        s.register_buffer('mask', torch.triu(torch.full((2 * F.SLOTS, 2 * F.SLOTS), float('-inf')), 1))

    def forward(s, x, st):
        st = st[:, None, :].expand(-1, x.shape[1], -1)
        h = s.inp(torch.cat([x, st], -1)) + s.pos
        return s.out(s.enc(h, mask=s.mask))[:, F.SLOTS:]


def batches(idx, bs, shuffle):
    idx = np.array(idx)
    if shuffle:
        idx = np.random.permutation(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        yield (torch.tensor(F.window(RAW, b, sig[b]), device=dev), torch.tensor(ST[b], device=dev),
               torch.tensor(T[b], device=dev))


def loss_fn(p, t):
    m = ~torch.isnan(t); m[:, 60:] = False
    return ((p - torch.nan_to_num(t)) ** 2 * m).sum() / m.sum()


def fit(tr, va, seed, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=0.05)
    best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        for x, s, t in batches(tr, 64, True):
            opt.zero_grad(); l = loss_fn(net(x, s), t); l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            vl = np.mean([loss_fn(net(x, s), t).item() for x, s, t in batches(va, 256, False)])
        if vl < best - 1e-4:
            best, state, bad = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(state)
    return net.eval()


def predict(net, idx):
    out = []
    with torch.no_grad():
        for x, s, t in batches(idx, 256, False):
            out.append(net(x, s).cpu().numpy())
    return np.concatenate(out)


tfm_day = np.full((nd, F.SLOTS, len(F.SYMBOLS)), np.nan, np.float32)
FOLDS = {}
for Y in range(2019, LAST + 1):
    tr = np.where(ok & (years < Y - 1))[0]; va = np.where(ok & (years == Y - 1))[0]; te = np.where(ok & (years == Y))[0]
    nets = [fit(tr, va, sd_) for sd_ in range(NS)]
    tfm_day[te] = np.mean([predict(n_, te) for n_ in nets], 0)
    if Y >= LAST - 2:
        FOLDS[Y] = nets
    print('tfm fold', Y, flush=True)
tfm_oos = tfm_day[RI, RK, RJ]
m = ~np.isnan(tfm_oos)
print('tfm OOS IC', np.corrcoef(tfm_oos[m], RY[m])[0, 1].round(4),
      {Y: np.corrcoef(tfm_oos[m & (ry == Y)], RY[m & (ry == Y)])[0, 1].round(3) for Y in range(2019, LAST + 1)}, flush=True)
final = [fit(np.where(ok & (years < LAST))[0], np.where(ok & (years == LAST))[0], 100 + s_) for s_ in range(NS)]

# ---------------- scales, export, OOS table ----------------
rsd = np.ones(F.SLOTS); tsd = np.ones(F.SLOTS)
for k in KS:
    mk = (RK == k) & ~np.isnan(ridge_oos) & ~np.isnan(tfm_oos)
    rsd[k], tsd[k] = ridge_oos[mk].std(), tfm_oos[mk].std()
for k in range(F.SLOTS):  # slots not in KS: nearest trained slot
    if k not in KS:
        kk = min(KS, key=lambda x: abs(x - k)); rsd[k], tsd[k] = rsd[kk], tsd[kk]
W = dict(ridge_mu=mu, ridge_sd=sd, ridge_coef=rfin.coef_, ridge_b=np.array(rfin.intercept_), ridge_pred_sd=rsd,
         tfm_pred_sd=tsd, tfm_layers=np.array(3), tfm_heads=np.array(4), tfm_seeds=np.array(NS))
for i, net in enumerate(final):
    for kname, v in net.state_dict().items():
        if kname != 'mask':
            W[f't{i}/{kname}'] = v.cpu().numpy()
np.savez_compressed(OUT, **{k: (np.asarray(v, np.float32) if np.asarray(v).dtype.kind == 'f' else np.asarray(v))
                            for k, v in W.items()})
def export(path, mu_, sd_, r_, nets, rs_, ts_):
    W_ = dict(ridge_mu=mu_, ridge_sd=sd_, ridge_coef=r_.coef_, ridge_b=np.array(r_.intercept_), ridge_pred_sd=rs_,
              tfm_pred_sd=ts_, tfm_layers=np.array(3), tfm_heads=np.array(4), tfm_seeds=np.array(len(nets)))
    for i_, n_ in enumerate(nets):
        for kn, v in n_.state_dict().items():
            if kn != 'mask':
                W_[f't{i_}/{kn}'] = v.cpu().numpy()
    np.savez_compressed(path, **{k_: (np.asarray(v, np.float32) if np.asarray(v).dtype.kind == 'f' else np.asarray(v))
                                 for k_, v in W_.items()})


(M / 'folds').mkdir(exist_ok=True)
for Y, nets in FOLDS.items():  # walk-forward weights for the realistic replay; scales from earlier OOS years only
    mu_, sd_, r_ = fit_ridge(ry < Y)
    rs_, ts_ = np.ones(F.SLOTS), np.ones(F.SLOTS)
    for k in KS:
        mk = (RK == k) & (ry < Y) & ~np.isnan(ridge_oos) & ~np.isnan(tfm_oos)
        rs_[k], ts_[k] = ridge_oos[mk].std(), tfm_oos[mk].std()
    for k in range(F.SLOTS):
        if k not in KS:
            kk = min(KS, key=lambda x: abs(x - k)); rs_[k], ts_[k] = rs_[kk], ts_[kk]
    export(M / 'folds' / f'gold_scar_{Y}.npz', mu_, sd_, r_, nets, rs_, ts_)
oos = pd.DataFrame(dict(u=np.array(F.TARGETS)[RJ], date=days[RI], k=RK, tgt_n=RY, ridge=ridge_oos, tfm=tfm_oos))
oos['pred'] = (oos.ridge / oos.k.map(dict(enumerate(rsd))) + oos.tfm / oos.k.map(dict(enumerate(tsd)))) / 2
oos.to_parquet(M / 'prod_oos.parquet')
oos['year'] = oos.date.dt.year
o = oos.dropna()
print('ENS OOS IC', np.corrcoef(o.pred, o.tgt_n)[0, 1].round(4),
      o.groupby('year').apply(lambda g: np.corrcoef(g.pred, g.tgt_n)[0, 1]).round(3).to_dict(), flush=True)
print('saved', OUT, OUT.stat().st_size)
# check: numpy forward == torch forward
from autotrader.options.ml import model as Mdl
Mdl.weights.cache_clear()
Wn = Mdl.weights()
i = np.where(ok)[0][-5]
x = F.window(RAW, np.array([i]), sig[[i]])[0]
with torch.no_grad():
    tt = final[0](torch.tensor(x[None], device=dev), torch.tensor(ST[[i]], device=dev)).cpu().numpy()[0]
nn_ = Mdl.transformer(Wn, 't0/', x.astype(np.float64), ST[i].astype(np.float64), 3, 4)[F.SLOTS:]
print('numpy vs torch max abs diff', np.abs(nn_ - tt).max())
