"""
run_fair_comparison.py  --  drop into the repo root (next to run_pipeline.py) and run:

    python run_fair_comparison.py                      # 5 seeds, real DeepMIMO data
    python run_fair_comparison.py --seeds 0 1 2 3 4 --epochs 40

What it fixes relative to run_pipeline.py / lstm_baseline.py / train_gru_baseline.py:
  1. Split is by TRAJECTORY (187/38/25 of 250), not by shuffled overlapping window.
  2. SNN, LSTM and GRU are trained on the SAME data (N_STEPS=100, SEQ_LEN=20,
     T_INDEX=0, TX_INDEX=0) and the SAME split, per seed.
  3. All three models are trained per-timestep (the released GRU only predicted the
     last step of each window) and are scored with ONE protocol: causal, non-overlapping
     20-step chunks with fresh state, on held-out test trajectories only.
  4. Top-k is reported with the paper's definition (ground truth inside the model's
     k best logits) and, for reference, the old code definition (model's top-1 inside
     the ground-truth top-k).
  5. Ping-pong rate is computed as defined in Sec. V-A.
  6. Adds two reference rows: Random and "Reactive" (beam = previous step's best beam).
  7. Logs firing rate of BOTH LIF layers on the test set, so the AC count in Table II
     can be recomputed per layer.
Existing files are not modified. Outputs go to --out.
"""
import os, sys, csv, json, argparse, contextlib, io, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepmimo_loader import (load_deepmimo_multifile, DeepMIMODataset, _synthetic_grid,
                             _synthesize_channels, _generate_dft_codebook, build_file_index, _read_mat)
import trajectory_generator as _tg
from trajectory_generator import generate_trajectories, trajectories_to_sequences
from snn_model import build_model
from trainer import train as train_snn
from lstm_baseline import LSTMBeamTracker

# ---- same settings as run_pipeline.py ----------------------------------------------------
DATA_DIR = r"G:\Shree\6G Beam Switching enabled by SNN\6G Dataset creation\deepmimo_scenarios\O1_140"
N_TRAJ, N_STEPS, SEQ_LEN, STRIDE = 250, 100, 20, 5
N_BEAMS, LR, BATCH = 64, 5e-4, 32
T_INDEX, TX_INDEX = 0, 0
LAMBDA_SPK, LAMBDA_TOPK = 1e-3, 0.1
SNR_DB = 20.0                      # fixed reference SNR used by the SE metric in trainer.py
# ------------------------------------------------------------------------------------------


class GRUSeq(nn.Module):
    """Same parameters as GRUBeamTracker (1 layer, 128 units) but reads out EVERY timestep."""
    def __init__(self, input_dim=10, hidden_dim=128, output_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)


# ---------------------------------------------------------------- data
# trajectory_generator calls compute_beam_gains(ds.channels, codebook) once PER TRAJECTORY, with a python
# loop over users x beams. Harmless for 21 users, ~30+ min for 10k users. Same maths, vectorised + cached.
_gain_cache = {}
def _fast_gains(H, codebook):
    key = (id(H), H.shape, codebook.shape)
    if key not in _gain_cache:
        _gain_cache[key] = np.abs(np.einsum('urt,bt->urb', H[..., 0], codebook)).sum(axis=1)
    return _gain_cache[key]
_tg.compute_beam_gains = _fast_gains


def load_subsampled(data_dir, t_index, tx_index, users_per_file, cache_dir, seed=0, probe=False):
    """The original loader keeps ONLY THE FIRST UE of every row file (21 UEs in total). Each row file holds
    ~500k UEs, so this draws `users_per_file` random UEs that have at least one propagation path from every
    row file of the chosen (snapshot, BS)."""
    index = build_file_index(data_dir)
    t_key = sorted(index['rx_pos'])[min(t_index, len(index['rx_pos']) - 1)]
    tx_key = sorted(index['rx_pos'][t_key])[min(tx_index, len(index['rx_pos'][t_key]) - 1)]
    rows = sorted(index['rx_pos'][t_key][tx_key])
    params = ['power', 'delay', 'phase', 'aoa_az', 'aoa_el', 'aod_az', 'aod_el']
    MAXP = 25
    f = lambda p, r: index[p][t_key][tx_key][r]
    if probe:
        for p in ['rx_pos', 'tx_pos'] + params:
            a = _read_mat(f(p, rows[0]))
            print(f"  {p:8s} shape={None if a is None else a.shape}")
        sys.exit(0)
    cache = os.path.join(cache_dir, f"ue_subsample_t{t_key}_tx{tx_key}_n{users_per_file}_s{seed}.npz")
    if os.path.exists(cache):
        z = np.load(cache); print(f"[subsample] loaded cache {cache}")
        pos, tx, par = z['pos'], z['tx'], {p: z[p] for p in params}
    else:
        rs = np.random.RandomState(seed)
        pos_l, par_l, tx = [], {p: [] for p in params}, np.zeros(3)
        for r in rows:
            pos = np.atleast_2d(_read_mat(f('rx_pos', r)))          # a one-UE file comes back as shape (3,)
            if pos.shape[1] != 3 and pos.shape[0] == 3: pos = pos.T
            N = len(pos)
            arr = {}
            for p in params:
                a = _read_mat(f(p, r)); assert a is not None, f"could not read {p} row {r}"
                a = np.atleast_2d(a)
                if a.shape[0] != N and a.shape[1] == N: a = a.T
                assert a.shape[0] == N, f"{p} row {r}: shape {a.shape} does not match {N} UEs"
                arr[p] = a
            pw = arr['power']
            valid = np.where((np.isfinite(pw) & (pw != 0)).any(axis=1) & np.isfinite(pos).all(axis=1))[0]
            if len(valid) == 0:
                print(f"[subsample] row {r:03d}: {N} UEs, none with a path -> skipped"); continue
            sel = np.sort(rs.choice(valid, min(users_per_file, len(valid)), replace=False))
            print(f"[subsample] row {r:03d}: {N} UEs, {len(valid)} with >=1 path, kept {len(sel)}")
            pos_l.append(pos[sel, :3])
            for p in params:
                a = arr[p][sel][:, :MAXP]
                par_l[p].append(np.pad(a, ((0, 0), (0, MAXP - a.shape[1]))))
        tx_arr = _read_mat(f('tx_pos', rows[0]))
        tx = np.atleast_1d(tx_arr).flatten()[:3] if tx_arr is not None else tx
        pos = np.concatenate(pos_l); par = {p: np.concatenate(par_l[p]) for p in params}
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, pos=pos, tx=tx, **par)
    ds = DeepMIMODataset(n_beams=N_BEAMS)
    ds.user_locations, ds.tx_location = pos, tx
    ds.path_power, ds.path_delay, ds.path_phase = par['power'], par['delay'], par['phase']
    ds.aoa_az, ds.aoa_el, ds.aod_az, ds.aod_el = par['aoa_az'], par['aoa_el'], par['aod_az'], par['aod_el']
    ds.num_paths = (np.isfinite(ds.path_power) & (ds.path_power != 0)).sum(axis=1)
    ds.beam_codebook = _generate_dft_codebook(N_BEAMS, 64)
    print(f"[subsample] {ds.n_users} UEs, avg paths {ds.num_paths.mean():.1f}, "
          f"x=[{pos[:,0].min():.0f},{pos[:,0].max():.0f}] y=[{pos[:,1].min():.0f},{pos[:,1].max():.0f}]")
    return ds

def load_data(synthetic, data_dir, n_traj, users_per_file=0, cache_dir='.', probe=False):
    if synthetic or not os.path.isdir(data_dir):
        print("[data] synthetic demo dataset (smoke test only -- NOT for paper numbers)")
        rs = np.random.RandomState(42)
        n_users, n_p = 400, 5
        ds = DeepMIMODataset(n_beams=N_BEAMS)
        ds.user_locations = _synthetic_grid(n_users)
        ds.path_power = rs.uniform(-90, -50, (n_users, n_p))
        ds.path_delay = rs.exponential(5e-9, (n_users, n_p))
        ds.path_phase = rs.uniform(-np.pi, np.pi, (n_users, n_p))
        ds.aod_az = rs.uniform(-60, 60, (n_users, n_p)); ds.aod_el = rs.uniform(-30, 30, (n_users, n_p))
        ds.aoa_az = rs.uniform(-60, 60, (n_users, n_p)); ds.aoa_el = rs.uniform(-30, 30, (n_users, n_p))
        ds.num_paths = np.full(n_users, n_p)
        ds.beam_codebook = _generate_dft_codebook(N_BEAMS, 64)
        ds.channels = _synthesize_channels(ds, N_rx=4, N_tx=64)
    elif users_per_file > 0 or probe:
        ds = load_subsampled(data_dir, T_INDEX, TX_INDEX, users_per_file, cache_dir, probe=probe)
    else:
        ds = load_deepmimo_multifile(data_dir, t_index=T_INDEX, tx_index=TX_INDEX, n_beams=N_BEAMS)
    # same NaN/Inf scrub as run_pipeline.py
    ds.user_locations = np.nan_to_num(ds.user_locations, nan=0.0, posinf=0.0, neginf=0.0)
    ds.tx_location = np.nan_to_num(ds.tx_location, nan=0.0, posinf=0.0, neginf=0.0)
    ds.path_power = np.nan_to_num(ds.path_power, nan=-200.0, posinf=-200.0, neginf=-200.0)
    for a in ['path_delay', 'path_phase', 'aod_az', 'aod_el', 'aoa_az', 'aoa_el']:
        setattr(ds, a, np.nan_to_num(getattr(ds, a), nan=0.0, posinf=0.0, neginf=0.0))
    ds.channels = _synthesize_channels(ds, N_rx=4, N_tx=64)
    ds.channels = np.nan_to_num(ds.channels, nan=0.0, posinf=0.0, neginf=0.0)
    trajs = generate_trajectories(ds, n_trajectories=n_traj, n_steps=N_STEPS, dt=0.5, top_k=5, seed=42)
    X, y, yk = trajectories_to_sequences(trajs, seq_len=SEQ_LEN, stride=STRIDE)
    per = len(range(0, N_STEPS - SEQ_LEN, STRIDE))          # windows per trajectory (16)
    assert len(X) == per * n_traj, "window ordering assumption broken"
    return trajs, X, y, yk, per, ds.n_users


def split_ids(n_traj, seed):
    perm = np.random.RandomState(seed).permutation(n_traj)
    n_te, n_va = int(round(0.10 * n_traj)), int(round(0.15 * n_traj))
    return perm[n_te + n_va:], perm[n_te:n_te + n_va], perm[:n_te]      # train, val, test


def make_loader(X, y, yk, ids, per, shuffle, seed):
    idx = np.concatenate([np.arange(t * per, (t + 1) * per) for t in ids])
    ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]), torch.from_numpy(yk[idx]))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, generator=g if shuffle else None)


# ---------------------------------------------------------------- training
def train_baseline(model, tr, va, device, epochs):
    """Baselines as released (AdamW lr 5e-4, plain CE) + best-val-loss checkpointing, like the SNN."""
    model.to(device)
    opt, ce = optim.AdamW(model.parameters(), lr=LR), nn.CrossEntropyLoss()
    best, best_state = float('inf'), None
    for _ in range(epochs):
        model.train()
        for xb, yb, _ in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            ce(model(xb).reshape(-1, N_BEAMS), yb.reshape(-1)).backward()
            opt.step()
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yb, _ in va:
                xb, yb = xb.to(device), yb.to(device)
                tot += ce(model(xb).reshape(-1, N_BEAMS), yb.reshape(-1)).item() * yb.numel(); n += yb.numel()
        if tot / n < best:
            best, best_state = tot / n, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def train_snn_model(tr, va, device, epochs, path):
    model = build_model(n_features=10, n_beams=N_BEAMS, device=device)
    with contextlib.redirect_stdout(io.StringIO()):
        train_snn(model, tr, va, n_epochs=epochs, lr=LR, device=device, patience=10,
                  save_path=path, lambda_spk=LAMBDA_SPK, lambda_topk=LAMBDA_TOPK)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def run_model(model, kind, traj, device):
    """Causal inference on one trajectory: non-overlapping SEQ_LEN chunks, fresh state per chunk."""
    model.eval()
    x = torch.tensor(traj.channel_features, dtype=torch.float32, device=device).unsqueeze(0)
    outs = []
    for s in range(0, traj.n_steps, SEQ_LEN):
        o = model(x[:, s:s + SEQ_LEN])
        outs.append((o[0] if kind == 'snn' else o)[0].cpu().numpy())
    return np.concatenate(outs, axis=0)                      # [T, n_beams]


def se_of(traj, beams):
    g = traj.beam_gains
    gn = g[np.arange(traj.n_steps), beams] / (g.max(axis=1) + 1e-12)
    return float(np.log2(1 + 10 ** (SNR_DB / 10) * gn).mean())


def score(trajs, preds, logits=None):
    """preds: list of [T] beam arrays; logits: optional list of [T, n_beams]."""
    c1, c3, c5, o3, o5, se, sw, pp = [], [], [], [], [], [], 0, 0
    steps = 0
    for i, tr in enumerate(trajs):
        p, gt = preds[i], tr.beam_indices
        c1.append(p == gt)
        o3.append((p[:, None] == tr.top_k_beams[:, :3]).any(1))        # old code definition
        o5.append((p[:, None] == tr.top_k_beams[:, :5]).any(1))
        if logits is not None:
            order = np.argsort(-logits[i], axis=1)
            c3.append((order[:, :3] == gt[:, None]).any(1))            # paper definition
            c5.append((order[:, :5] == gt[:, None]).any(1))
        se.append(se_of(tr, p))
        sw += int(np.sum(np.diff(p) != 0))
        pp += int(np.sum((p[1:-1] != p[:-2]) & (p[2:] == p[:-2])))     # switch reversed next step
        steps += tr.n_steps
    out = dict(top1=100 * np.concatenate(c1).mean(),
               top3_oldDef=100 * np.concatenate(o3).mean(), top5_oldDef=100 * np.concatenate(o5).mean(),
               se=float(np.mean(se)), switches_per_traj=sw / len(trajs),
               switch_rate_per_step=sw / steps, pingpong_rate=pp / max(sw, 1))
    if logits is not None:
        out['top3'] = 100 * np.concatenate(c3).mean()
        out['top5'] = 100 * np.concatenate(c5).mean()
    return out


def eval_learned(model, kind, trajs, device):
    counts = {}
    hooks = []
    if kind == 'snn':
        for name in ('lif1', 'lif2'):
            counts[name] = [0.0, 0]
            def hook(m, i, o, name=name):
                s = o[0] if isinstance(o, tuple) else o
                counts[name][0] += float(s.sum()); counts[name][1] += s.numel()
            hooks.append(getattr(model, name).register_forward_hook(hook))
    logits = [run_model(model, kind, t, device) for t in trajs]
    for h in hooks: h.remove()
    res = score(trajs, [l.argmax(1) for l in logits], logits)
    if kind == 'snn':
        r1, r2 = counts['lif1'][0] / counts['lif1'][1], counts['lif2'][0] / counts['lif2'][1]
        ac = r1 * 256 * 128 + r2 * 128 * 64          # fc2 is driven by layer-1 spikes, fc_out by layer-2 spikes
        mac = 10 * 256                               # fc1 sees analog input -> MAC
        res.update(fr_layer1=100 * r1, fr_layer2=100 * r2, ac_ops=ac, mac_ops=mac,
                   energy_vs_dense_pct=100 * (mac + ac / 5.0) / 43520)   # 5x MAC/AC ratio as in the paper
    return res


def eval_reference(trajs, seed, fixed_beam):
    rs = np.random.RandomState(1000 + seed)
    gts = [t.beam_indices for t in trajs]
    return {
        'Oracle': score(trajs, gts),
        'Reactive (prev-step best beam)': score(trajs, [np.concatenate([[g[0]], g[:-1]]) for g in gts]),
        'Fixed beam (train-majority)': score(trajs, [np.full(t.n_steps, fixed_beam) for t in trajs]),
        'Random': score(trajs, [rs.randint(0, N_BEAMS, t.n_steps) for t in trajs]),
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--n_traj', type=int, default=N_TRAJ)
    ap.add_argument('--data_dir', default=DATA_DIR)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--users_per_file', type=int, default=500,
                    help='UEs drawn from each row file (21 files). 0 = original loader (first UE of each file only)')
    ap.add_argument('--probe', action='store_true', help='print array shapes of the first row file and exit')
    ap.add_argument('--out', default='fair_comparison_results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trajs, X, y, yk, per, n_ue = load_data(a.synthetic, a.data_dir, a.n_traj, a.users_per_file, a.out, a.probe)
    lab = np.concatenate([t.beam_indices for t in trajs])
    cnt = np.bincount(lab, minlength=N_BEAMS)
    print(f"[diag] UE locations in the loaded dataset: {n_ue} | distinct best-beam labels: {(cnt > 0).sum()} of {N_BEAMS}"
          f" | most common beam = {100 * cnt.max() / cnt.sum():.1f}% of all steps")
    rows = []
    for seed in a.seeds:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        tr_ids, va_ids, te_ids = split_ids(a.n_traj, seed)
        tr = make_loader(X, y, yk, tr_ids, per, True, seed)
        va = make_loader(X, y, yk, va_ids, per, False, seed)
        test_trajs = [trajs[i] for i in te_ids]
        print(f"\n[seed {seed}] trajectories train/val/test = {len(tr_ids)}/{len(va_ids)}/{len(te_ids)}")

        models = {
            'REAP-6G (SNN)': ('snn', train_snn_model(tr, va, device, a.epochs, os.path.join(a.out, f'snn_seed{seed}.pt'))),
            'LSTM': ('rnn', train_baseline(LSTMBeamTracker(input_dim=10, output_dim=N_BEAMS), tr, va, device, a.epochs)),
            'GRU': ('rnn', train_baseline(GRUSeq(input_dim=10, output_dim=N_BEAMS), tr, va, device, a.epochs)),
        }
        results = {n: eval_learned(m, k, test_trajs, device) for n, (k, m) in models.items()}
        fixed = int(np.bincount(np.concatenate([trajs[i].beam_indices for i in tr_ids]), minlength=N_BEAMS).argmax())
        results.update(eval_reference(test_trajs, seed, fixed))
        for name, r in results.items():
            rows.append(dict(seed=seed, model=name, **r))
            print(f"  {name:32s} top1={r['top1']:6.2f}  SE={r['se']:.3f}  switches/traj={r['switches_per_traj']:.2f}"
                  f"  pingpong={r['pingpong_rate']:.3f}")
        print(f"  ({time.time() - t0:.0f}s)")

    # ---- save + summarise
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('seed', 'model'), k))
    with open(os.path.join(a.out, 'per_seed.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    order = ['REAP-6G (SNN)', 'LSTM', 'GRU', 'Reactive (prev-step best beam)', 'Fixed beam (train-majority)', 'Random', 'Oracle']
    cols = [('top1', 'Top-1 %'), ('top3', 'Top-3 %'), ('top5', 'Top-5 %'), ('se', 'SE b/s/Hz'),
            ('switches_per_traj', 'Switches/traj'), ('pingpong_rate', 'Ping-pong'),
            ('fr_layer1', 'FR L1 %'), ('fr_layer2', 'FR L2 %'), ('ac_ops', 'AC ops'), ('energy_vs_dense_pct', 'Energy % of dense')]
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    lines = [f"Held-out test trajectories, mean ± std over seeds {a.seeds}", "",
             "| Model | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m in order:
        rs_ = [r for r in rows if r['model'] == m]
        cells = []
        for k, _ in cols:
            v = [r[k] for r in rs_ if k in r]
            cells.append("–" if not v else f"{np.mean(v):.3f} ± {sd(v):.3f}" if k in ('se', 'pingpong_rate')
                         else f"{np.mean(v):.2f} ± {sd(v):.2f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    snn = {r['seed']: r['top1'] for r in rows if r['model'] == 'REAP-6G (SNN)'}
    for b in ('LSTM', 'GRU'):
        d = [snn[r['seed']] - r['top1'] for r in rows if r['model'] == b]
        lines.append(f"\nTop-1 (SNN - {b}), paired by seed: {np.mean(d):+.2f} ± {sd(d):.2f} points")
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write(txt + "\n")
    json.dump(vars(a) | dict(N_STEPS=N_STEPS, SEQ_LEN=SEQ_LEN, T_INDEX=T_INDEX, TX_INDEX=TX_INDEX),
              open(os.path.join(a.out, 'config.json'), 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()