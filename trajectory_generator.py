"""
Multi-User Trajectory Generator
Generates 50+ realistic UE mobility trajectories over the DeepMIMO spatial grid.
Supports: linear, random-walk, L-shaped, circular, and highway mobility patterns.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from deepmimo_loader import DeepMIMODataset, compute_beam_gains, get_optimal_beams


@dataclass
class UserTrajectory:
    """Single UE trajectory with beam labels"""
    user_id: int
    positions: np.ndarray          # [T, 3]
    beam_indices: np.ndarray       # [T] optimal beam at each step
    top_k_beams: np.ndarray        # [T, K] top-K beams at each step
    beam_gains: np.ndarray         # [T, n_beams]
    channel_features: np.ndarray   # [T, F] extracted features for SNN input
    velocity: float                # m/s
    mobility_type: str
    n_steps: int


def generate_trajectories(
    ds: DeepMIMODataset,
    n_trajectories: int = 50,
    n_steps: int = 100,
    dt: float = 0.5,              # seconds per step
    top_k: int = 5,
    seed: int = 42
) -> List[UserTrajectory]:
    """
    Generate n_trajectories multi-user trajectories.
    Each trajectory is a UE moving over the DeepMIMO spatial grid.
    """
    np.random.seed(seed)
    trajectories = []

    locs = ds.user_locations        # [N_users, 3]
    x_min, x_max = locs[:, 0].min(), locs[:, 0].max()
    y_min, y_max = locs[:, 1].min(), locs[:, 1].max()

    mobility_types = ['linear', 'random_walk', 'L_shaped', 'circular', 'highway']
    velocities = {'pedestrian': 1.5, 'vehicle': 15.0, 'high_speed': 60.0}

    print(f"\n[Trajectory] Generating {n_trajectories} trajectories × {n_steps} steps")
    print(f"[Trajectory] Grid: x=[{x_min:.1f},{x_max:.1f}] y=[{y_min:.1f},{y_max:.1f}]")

    for i in range(n_trajectories):
        mob_type = mobility_types[i % len(mobility_types)]

        # Assign velocity profile
        if i % 5 == 0:
            v = velocities['high_speed'] + np.random.randn() * 5
        elif i % 3 == 0:
            v = velocities['vehicle'] + np.random.randn() * 2
        else:
            v = velocities['pedestrian'] + np.random.randn() * 0.3

        # Generate 2D positions
        positions_2d = _generate_path(
            mob_type, n_steps, v, dt,
            x_min, x_max, y_min, y_max
        )

        # Add z=1.5m (UE height)
        z = np.full((n_steps, 1), 1.5)
        positions = np.concatenate([positions_2d, z], axis=1)  # [T, 3]

        # Interpolate channel features at each position
        beam_gains, beam_labels, topk_beams, ch_features = _interpolate_channels(
            positions, ds, top_k
        )

        traj = UserTrajectory(
            user_id=i,
            positions=positions,
            beam_indices=beam_labels,
            top_k_beams=topk_beams,
            beam_gains=beam_gains,
            channel_features=ch_features,
            velocity=v,
            mobility_type=mob_type,
            n_steps=n_steps
        )
        trajectories.append(traj)

        if (i + 1) % 10 == 0:
            switches = np.sum(np.diff(beam_labels) != 0)
            print(f"  [{i+1:3d}/{n_trajectories}] {mob_type:12s} v={v:5.1f}m/s  beam_switches={switches}")

    print(f"[Trajectory] ✓ Generated {len(trajectories)} trajectories")
    return trajectories


def _generate_path(
    mob_type: str, n_steps: int, velocity: float, dt: float,
    x_min: float, x_max: float, y_min: float, y_max: float
) -> np.ndarray:
    """Generate 2D trajectory path [n_steps, 2]"""

    # Random start position
    x0 = np.random.uniform(x_min + 5, x_max - 5)
    y0 = np.random.uniform(y_min + 5, y_max - 5)
    step_len = velocity * dt

    if mob_type == 'linear':
        angle = np.random.uniform(0, 2 * np.pi)
        xs = np.clip(x0 + np.arange(n_steps) * step_len * np.cos(angle), x_min, x_max)
        ys = np.clip(y0 + np.arange(n_steps) * step_len * np.sin(angle), y_min, y_max)

    elif mob_type == 'random_walk':
        pos = np.zeros((n_steps, 2))
        pos[0] = [x0, y0]
        for t in range(1, n_steps):
            angle = np.random.uniform(0, 2 * np.pi)
            pos[t, 0] = np.clip(pos[t-1, 0] + step_len * np.cos(angle), x_min, x_max)
            pos[t, 1] = np.clip(pos[t-1, 1] + step_len * np.sin(angle), y_min, y_max)
        return pos

    elif mob_type == 'L_shaped':
        turn = n_steps // 2
        angle1 = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2])
        angle2 = angle1 + np.pi/2
        x1 = x0 + np.arange(turn) * step_len * np.cos(angle1)
        y1 = y0 + np.arange(turn) * step_len * np.sin(angle1)
        x2 = x1[-1] + np.arange(n_steps - turn) * step_len * np.cos(angle2)
        y2 = y1[-1] + np.arange(n_steps - turn) * step_len * np.sin(angle2)
        xs = np.clip(np.concatenate([x1, x2]), x_min, x_max)
        ys = np.clip(np.concatenate([y1, y2]), y_min, y_max)

    elif mob_type == 'circular':
        radius = np.random.uniform(10, 30)
        cx = np.clip(x0, x_min + radius, x_max - radius)
        cy = np.clip(y0, y_min + radius, y_max - radius)
        total_angle = (n_steps * step_len) / radius
        angles = np.linspace(0, total_angle, n_steps)
        xs = cx + radius * np.cos(angles)
        ys = cy + radius * np.sin(angles)

    elif mob_type == 'highway':
        # Straight line along x-axis with small lateral drift
        xs = np.linspace(x_min + 5, x_max - 5, n_steps)
        ys = y0 + np.cumsum(np.random.randn(n_steps) * 0.2)
        ys = np.clip(ys, y_min, y_max)

    else:
        xs = np.full(n_steps, x0)
        ys = np.full(n_steps, y0)

    return np.stack([xs, ys], axis=1)


def _interpolate_channels(
    positions: np.ndarray,
    ds: DeepMIMODataset,
    top_k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolates large-scale channel features using IDW. 
    Predictive SNN tracks the macro-envelope, ignoring sub-millisecond small-scale fading.
    """
    locs2d = ds.user_locations[:, :2]   
    pos2d  = positions[:, :2]           
    T = len(positions)
    n_beams = ds.n_beams
    codebook = ds.beam_codebook

    all_gains    = np.zeros((T, n_beams))
    best_beams   = np.zeros(T, dtype=int)
    topk_beams   = np.zeros((T, top_k), dtype=int)
    features     = np.zeros((T, _feature_dim(ds)))

    dataset_gains = compute_beam_gains(ds.channels, codebook)  

    for t in range(T):
        dists = np.linalg.norm(locs2d - pos2d[t], axis=1)
        k_nn = min(4, len(dists))
        nn_idx = np.argsort(dists)[:k_nn]
        nn_d   = dists[nn_idx] + 1e-6

        # 1. Pure Large-Scale Fading (IDW Interpolation)
        weights = 1.0 / (nn_d ** 2)
        weights /= weights.sum()
        gains_t = (weights[:, None] * dataset_gains[nn_idx]).sum(axis=0)

        all_gains[t]  = gains_t
        best_beams[t] = np.argmax(gains_t)
        topk_beams[t] = np.argsort(-gains_t)[:top_k]

        features[t] = _extract_features(nn_idx[0], ds, gains_t, positions[t])

    return all_gains, best_beams, topk_beams, features


def _feature_dim(ds: DeepMIMODataset) -> int:
    """Feature vector dimension: RSS + angle_spread + delay_spread + position_2d + top5_gains"""
    return 1 + 1 + 1 + 2 + 5   # = 10


def _extract_features(user_idx: int, ds: DeepMIMODataset,
                       beam_gains: np.ndarray, position: np.ndarray) -> np.ndarray:
    """
    Extract channel features for SNN input encoding:
    [RSS_dB, AoD_spread, delay_spread, x_norm, y_norm, top5_gain_norm]
    """
    # RSS (received signal strength)
    rss_db = 10 * np.log10(np.max(beam_gains) + 1e-12)

    # AoD angular spread
    if ds.aod_az.size > user_idx:
        az = ds.aod_az[user_idx]
        az = az[az != 0]
        az_spread = np.std(az) if len(az) > 1 else 0.0
    else:
        az_spread = 0.0

    # Delay spread
    if ds.path_delay.size > user_idx:
        td = ds.path_delay[user_idx]
        td = td[td != 0]
        delay_spread = np.std(td) if len(td) > 1 else 0.0
    else:
        delay_spread = 0.0

    # Position (normalized)
    locs = ds.user_locations
    x_norm = (position[0] - locs[:, 0].min()) / (np.ptp(locs[:, 0]) + 1e-6)
    y_norm = (position[1] - locs[:, 1].min()) / (np.ptp(locs[:, 1]) + 1e-6)

    # Top-5 beam gains (normalized)
    top5 = np.sort(beam_gains)[::-1][:5]
    top5_norm = top5 / (top5.sum() + 1e-12)

    features = np.array([
        np.clip((rss_db + 120) / 120, 0, 1),   # normalize RSS to [0,1]
        np.clip(az_spread / 90, 0, 1),
        np.clip(delay_spread * 1e9, 0, 1),      # ns scale
        x_norm,
        y_norm,
        *top5_norm
    ])
    return features


def trajectories_to_sequences(
    trajectories: List[UserTrajectory],
    seq_len: int = 20,
    stride: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice trajectories into overlapping sequences for R-SNN training.
    Returns:
        X:      [N_seq, seq_len, n_features]
        y:      [N_seq, seq_len] beam index labels
        y_topk: [N_seq, seq_len, K] top-K labels
    """
    X_all, y_all, yk_all = [], [], []

    for traj in trajectories:
        T = traj.n_steps
        for start in range(0, T - seq_len, stride):
            end = start + seq_len
            X_all.append(traj.channel_features[start:end])
            y_all.append(traj.beam_indices[start:end])
            yk_all.append(traj.top_k_beams[start:end])

    X = np.stack(X_all, axis=0).astype(np.float32)
    y = np.stack(y_all, axis=0).astype(np.int64)
    yk = np.stack(yk_all, axis=0).astype(np.int64)

    print(f"[Trajectory] Sequences: X={X.shape}  y={y.shape}  y_topk={yk.shape}")
    return X, y, yk