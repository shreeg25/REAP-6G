"""
make_figures.py -- regenerate Fig. 2 and Fig. 3 for the NCC submission.

Run from the repo root:

    python make_figures.py --data_dir "F:\\Shree\\6G Beam Switching enabled by SNN\\6G Dataset creation\\deepmimo_scenarios\\O1_140" --csv fair_matched/per_seed.csv --cache_dir fair_results

Writes into --fig_dir (default: figures/):
    1_trajectory_diagram_trajectories.png   Fig. 2a -- 250 trajectories over the sampled UE grid
    1_trajectory_diagram_beamswitches.png   Fig. 2b -- beam changes per trajectory
    3_results_accuracy.png                  Fig. 3a -- Top-1 per model, one dot per seed
    3_results_se.png                        Fig. 3b -- SE against the three reference rules
    3_results_firingrate.png                Fig. 3c -- per-layer firing rate across seeds
    3_results_cost.png                      Fig. 3d -- accuracy against MAC-equivalent ops

Fig. 2 needs the dataset (it reuses run_fair_comparison's cached UE subsample, so pass the
same --cache_dir you used there). Fig. 3 needs only the CSV: use --skip_fig2 if the dataset
is not to hand.
"""
import os, argparse, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_fair_comparison as R

# IEEE two-column: ~3.5 in per column. Keep fonts >= 7pt so they survive the shrink.
plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
    'figure.dpi': 400, 'savefig.dpi': 400, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_SNN, C_LSTM, C_GRU, C_REF = '#1f4e79', '#c0504d', '#e8a33d', '#7f7f7f'


def read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({k: (v if k == 'model' else (float(v) if v not in ('', '-', None) else np.nan))
                         for k, v in r.items()})
    return rows


def col(rows, model, key):
    return np.array([r[key] for r in rows if r['model'] == model and not np.isnan(r.get(key, np.nan))])


# ----------------------------------------------------------------- Fig. 2
def fig2(args):
    """Fig. 2a keeps the original layout: trajectory points shaded by normalised best-beam gain,
    BS marker, start/end markers, one colour per mobility pattern. Fig. 2b defaults to the original
    per-trajectory stem plot; --hist_switches draws a histogram instead."""
    trajs, X, y, yk, per, n_ue = R.load_data(False, args.data_dir, args.n_traj,
                                             args.users_per_file, args.cache_dir, False)
    ds = R.load_subsampled(args.data_dir, R.T_INDEX, R.TX_INDEX, args.users_per_file, args.cache_dir)
    labels = [t.mobility_type for t in trajs]
    uniq = sorted(set(labels))
    cmap = plt.get_cmap('tab10')

    # ---- (a) trajectories over a gain-shaded background
    # All 250 trajectories at once is unreadable, so the gain field is drawn from every point while
    # only a stratified subset of trajectories is traced (args.show_traj, evenly over patterns).
    pts = np.concatenate([t.positions[:, :2] for t in trajs])
    g = np.concatenate([t.beam_gains[np.arange(t.n_steps), t.beam_indices] for t in trajs])
    g = g / (g.max() + 1e-12)
    rs = np.random.RandomState(0)
    per_pat = len(trajs) if args.show_traj == 0 else max(1, args.show_traj // len(uniq))
    shown = []
    for u in uniq:
        idx = [k for k, l in enumerate(labels) if l == u]
        shown += list(rs.choice(idx, min(per_pat, len(idx)), replace=False))
    shown = sorted(shown)

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=g, s=0.8, cmap='viridis', alpha=0.22, linewidths=0,
                    rasterized=True)
    for i_ in shown:
        tr = trajs[i_]
        ax.plot(tr.positions[:, 0], tr.positions[:, 1], lw=0.8, alpha=0.9,
                color=cmap(uniq.index(labels[i_]) % 10), solid_capstyle='round')
        ax.plot(tr.positions[0, 0], tr.positions[0, 1], 'o', ms=2.4, mfc='white',
                mec='black', mew=0.4, zorder=4)
        ax.plot(tr.positions[-1, 0], tr.positions[-1, 1], 's', ms=2.2, color='black', zorder=4)
    ax.scatter([ds.tx_location[0]], [ds.tx_location[1]], marker='^', s=60, color='#c0504d',
               edgecolors='white', linewidths=0.7, zorder=6, label='Base station')
    ax.plot([], [], 'o', ms=3, mfc='white', mec='black', mew=0.5, label='start')
    ax.plot([], [], 's', ms=3, color='black', label='end')
    for j_, u in enumerate(uniq):
        ax.plot([], [], lw=1.4, color=cmap(j_ % 10), label=u.replace('_', ' '))
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label('Normalised best beam gain', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.solids.set_alpha(1.0)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.legend(ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.24),
              handletextpad=0.35, columnspacing=0.8, fontsize=6.0)
    ax.grid(alpha=0.12)
    fig.savefig(os.path.join(args.fig_dir, '1_trajectory_diagram_trajectories.png'))
    plt.close(fig)
    print(f"[fig2a] traced {len(shown)} of {len(trajs)} trajectories "
          f"({per_pat} per pattern); gain field from all {len(pts)} points")

    # ---- (b) beam changes per trajectory, bars coloured by mobility pattern
    sw = np.array([int(np.sum(np.diff(t.beam_indices) != 0)) for t in trajs])
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    if args.hist_switches:
        ax.hist(sw, bins=np.arange(-0.5, sw.max() + 1.5), color=C_SNN,
                edgecolor='white', linewidth=0.5)
        ax.axvline(sw.mean(), color='black', ls='--', lw=1.1, label=f'Mean = {sw.mean():.1f}')
        ax.set_xlabel('Beam changes per trajectory'); ax.set_ylabel('# trajectories')
    else:
        bar_c = [cmap(uniq.index(l) % 10) for l in labels]
        ax.bar(np.arange(len(sw)), sw, width=0.9, color=bar_c, linewidth=0)
        ax.axhline(sw.mean(), color='black', ls='--', lw=1.1, label=f'Mean = {sw.mean():.1f}')
        ax.set_xlabel('Trajectory index'); ax.set_ylabel('# beam switches')
        ax.set_xlim(-3, len(sw) + 2); ax.set_ylim(0, sw.max() * 1.12)
    if args.panel_titles:
        ax.set_title('Beam switch count per trajectory')
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='0.7', loc='upper right')
    fig.savefig(os.path.join(args.fig_dir, '1_trajectory_diagram_beamswitches.png'))
    plt.close(fig)
    print(f"[fig2] {len(trajs)} trajectories, {n_ue} UEs; beam changes mean {sw.mean():.2f}, "
          f"max {sw.max()}, zero-switch trajectories {int((sw == 0).sum())}")
    return sw


# ----------------------------------------------------------------- Fig. 3
def fig3(rows, fig_dir):
    models = ['REAP-6G (SNN)', 'LSTM', 'GRU']
    short = {'REAP-6G (SNN)': 'REAP-6G', 'LSTM': 'LSTM', 'GRU': 'GRU'}
    cols = [C_SNN, C_LSTM, C_GRU]
    rng = np.random.RandomState(0)

    # (a) Top-1, bar + per-seed dots
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for i, m in enumerate(models):
        v = col(rows, m, 'top1')
        ax.bar(i, v.mean(), yerr=v.std(ddof=1), width=0.6, color=cols[i],
               error_kw=dict(lw=0.9, capsize=3))
        ax.scatter(np.full_like(v, i) + rng.uniform(-0.13, 0.13, len(v)), v,
                   s=7, color='black', zorder=3, alpha=0.65, linewidths=0)
    react = col(rows, 'Reactive (prev-step best beam)', 'top1')
    ax.axhline(react.mean(), color=C_REF, ls='--', lw=1.1)
    ax.text(2.42, react.mean() + 0.5, f'reactive {react.mean():.1f}%', ha='right',
            fontsize=6.5, color=C_REF)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([short[m] for m in models])
    ax.set_ylabel('Top-1 accuracy (%)')
    lo = min(min(col(rows, m, 'top1').min() for m in models), react.mean()) - 4
    ax.set_ylim(max(0, lo), 101)
    fig.savefig(os.path.join(fig_dir, '3_results_accuracy.png')); plt.close(fig)

    # (b) SE, learned models against the reference rules
    names = models + ['Fixed beam (train-majority)', 'Reactive (prev-step best beam)', 'Oracle']
    lab = [short.get(n, n.split(' (')[0]) for n in names]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    means = [col(rows, n, 'se').mean() for n in names]
    lo, hi = min(means), max(means)
    pad = max(0.25, 0.12 * (hi - lo))
    ylo, yhi = max(0.0, lo - pad), hi + pad
    for i, n in enumerate(names):
        v = col(rows, n, 'se')
        ax.bar(i, v.mean(), yerr=(v.std(ddof=1) if len(v) > 1 else 0), width=0.62,
               color=cols[i] if i < 3 else C_REF, error_kw=dict(lw=0.9, capsize=3))
        ax.annotate(f'{v.mean():.2f}', (i, min(v.mean() + 0.02 * (yhi - ylo), yhi)),
                    ha='center', fontsize=6.3, annotation_clip=True)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(lab, rotation=20, ha='right')
    ax.set_ylabel('Spectral efficiency (b/s/Hz)'); ax.set_ylim(ylo, yhi)
    fig.savefig(os.path.join(fig_dir, '3_results_se.png')); plt.close(fig)

    # (c) per-layer firing rate
    f1, f2 = col(rows, 'REAP-6G (SNN)', 'fr_layer1'), col(rows, 'REAP-6G (SNN)', 'fr_layer2')
    seeds = col(rows, 'REAP-6G (SNN)', 'seed')
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(seeds, f1, 'o-', ms=3.5, lw=1.1, color=C_SNN, label='LIF layer 1 (256)')
    ax.plot(seeds, f2, 's-', ms=3.5, lw=1.1, color=C_GRU, label='LIF layer 2 (128)')
    ax.axhline(15, color=C_REF, ls=':', lw=1.0)
    ax.text(seeds.max(), 16, 'loss target 15%', ha='right', fontsize=6.5, color=C_REF)
    ax.set_xlabel('Seed'); ax.set_ylabel('Mean firing rate (%)')
    ax.set_ylim(0, max(60, f1.max() * 1.25)); ax.set_xticks(seeds.astype(int)); ax.legend(frameon=False, loc='lower right')
    fig.savefig(os.path.join(fig_dir, '3_results_firingrate.png')); plt.close(fig)

    # (d) accuracy against cost. MAC-equivalents, AC = MAC/5, matching Table III.
    ac = col(rows, 'REAP-6G (SNN)', 'ac_ops').mean()
    pts = [('REAP-6G', 2560 + 448 + ac / 5, col(rows, 'REAP-6G (SNN)', 'top1'), C_SNN, 'o'),
           ('LSTM', 209920, col(rows, 'LSTM', 'top1'), C_LSTM, 's'),
           ('GRU', 61184, col(rows, 'GRU', 'top1'), C_GRU, '^')]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for name, ops, acc, c, mk in pts:
        ax.errorbar(ops, acc.mean(), yerr=acc.std(ddof=1), fmt=mk, ms=6, color=c,
                    capsize=3, lw=1.0)
        ax.annotate(name, (ops, acc.mean()), textcoords='offset points', xytext=(0, 9),
                    ha='center', fontsize=7)
    ax.set_xscale('log')
    ax.set_xlabel('MAC-equivalent operations per step (log scale)')
    ax.set_ylabel('Top-1 accuracy (%)')
    accs = np.concatenate([p[2] for p in pts]); opsv = [p[1] for p in pts]
    ax.set_ylim(accs.min() - 4, min(100.5, accs.max() + 4))
    ax.set_xlim(min(opsv) * 0.45, max(opsv) * 2.5)
    fig.savefig(os.path.join(fig_dir, '3_results_cost.png')); plt.close(fig)
    print(f"[fig3] REAP-6G {col(rows,'REAP-6G (SNN)','top1').mean():.2f}% at "
          f"{2560 + 448 + ac/5:.0f} MAC-eq; LSTM 209920; GRU 61184")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='fair_matched/per_seed.csv')
    ap.add_argument('--fig_dir', default='figures')
    ap.add_argument('--data_dir', default=R.DATA_DIR)
    ap.add_argument('--cache_dir', default='fair_results')
    ap.add_argument('--users_per_file', type=int, default=500)
    ap.add_argument('--n_traj', type=int, default=R.N_TRAJ)
    ap.add_argument('--skip_fig2', action='store_true')
    ap.add_argument('--show_traj', type=int, default=25,
                    help='how many trajectories to trace in Fig. 2a (0 = all); the gain field always uses all')
    ap.add_argument('--panel_titles', action='store_true',
                    help='draw a title inside each panel (IEEE style normally leaves this to the caption)')
    ap.add_argument('--hist_switches', action='store_true',
                    help='Fig. 2b as a histogram instead of the per-trajectory bars')
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    if not a.skip_fig2:
        fig2(a)
    fig3(read_csv(a.csv), a.fig_dir)
    print(f"[done] figures written to {os.path.abspath(a.fig_dir)}")


if __name__ == '__main__':
    main()