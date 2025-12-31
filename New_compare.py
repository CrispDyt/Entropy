import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

USE_SINKHORN_3D = True
try:
    import torch
    from geomloss import SamplesLoss
except Exception:
    USE_SINKHORN_3D = False


def lorenz_dynamics(xyz):
    s, r, b = 10.0, 28.0, 2.667
    x, y, z = xyz
    return np.array([s * (y - x), r * x - y - x * z, x * y - b * z], dtype=np.float64)

def generate_ground_truth_from_start_point(x0, steps, dt=0.01):
    y = np.zeros((steps, 3), dtype=np.float64)
    y[0] = x0.astype(np.float64)
    curr = x0.astype(np.float64)
    for i in range(steps - 1):
        curr = curr + lorenz_dynamics(curr) * dt
        y[i + 1] = curr
    return y

def load_traj(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return np.asarray(data[4], dtype=np.float64)  # xs_orig

print("Loading data...")
traj_ours = load_traj("DualOpt_Best.p")
traj_base = load_traj("DIM_recon_short.p")

min_len = min(len(traj_ours), len(traj_base))
traj_ours = traj_ours[:min_len]
traj_base = traj_base[:min_len]
dt = 0.01

print("Generating matched GT for EACH model...")
x0_ours = traj_ours[0]
x0_base = traj_base[0]
gt_ours = generate_ground_truth_from_start_point(x0_ours, min_len, dt=dt)
gt_base = generate_ground_truth_from_start_point(x0_base, min_len, dt=dt)

print("Sanity t=0 (should be 0):")
print("  ours:", float(np.sum((traj_ours[0] - gt_ours[0])**2)))
print("  base:", float(np.sum((traj_base[0] - gt_base[0])**2)))


def w1_marginals(traj_a, traj_b):
    """W1 for x,y,z and radius r."""
    xa, ya, za = traj_a[:,0], traj_a[:,1], traj_a[:,2]
    xb, yb, zb = traj_b[:,0], traj_b[:,1], traj_b[:,2]
    ra = np.sqrt(xa*xa + ya*ya + za*za)
    rb = np.sqrt(xb*xb + yb*yb + zb*zb)
    return {
        "x": wasserstein_distance(xa, xb),
        "y": wasserstein_distance(ya, yb),
        "z": wasserstein_distance(za, zb),
        "r": wasserstein_distance(ra, rb),
    }

def speed(traj):
    v = np.diff(traj, axis=0) / dt
    return np.linalg.norm(v, axis=1)

def curvature_proxy(traj):
    """
    Simple turning/curvature proxy:
      k_t = ||v_{t+1} - v_t|| / (||v_t|| + eps)
    captures abrupt direction changes / teleport-like jumps / over-smoothing differences
    """
    v = np.diff(traj, axis=0) / dt                  # [T-1,3]
    dv = np.diff(v, axis=0) / dt                    # [T-2,3]
    vnorm = np.linalg.norm(v[:-1], axis=1) + 1e-9
    knorm = np.linalg.norm(dv, axis=1) / vnorm
    return knorm

def divergence_time(traj, gt, thresholds=(1.0, 5.0, 10.0)):
    """
    time (steps) when ||x_t - gt_t|| first exceeds threshold.
    return dict threshold->time
    """
    err = np.linalg.norm(traj - gt, axis=1)
    out = {}
    for th in thresholds:
        idx = np.argmax(err > th)
        if err[0] > th:
            out[th] = 0
        elif (err > th).any():
            out[th] = int(idx)
        else:
            out[th] = None
    return out

def short_window_mse(traj, gt, win=2000):
    e = np.sum((traj[:win] - gt[:win])**2, axis=1)
    return float(np.mean(e)), e

def log_psd_distance(x, y, dt, nfft=4096):
    """
    Compare log power spectral density (PSD) of scalar signals x(t), y(t).
    Uses rFFT. Return L2 distance between normalized log-PSD.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(len(x), len(y), nfft)
    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])
    fx = np.fft.rfft(x)
    fy = np.fft.rfft(y)
    psdx = (fx*np.conj(fx)).real + 1e-12
    psdy = (fy*np.conj(fy)).real + 1e-12
    lx = np.log(psdx)
    ly = np.log(psdy)
    lx = (lx - lx.mean()) / (lx.std() + 1e-9)
    ly = (ly - ly.mean()) / (ly.std() + 1e-9)
    return float(np.sqrt(np.mean((lx - ly)**2)))

def poincare_section(traj, z0=25.0):
    z = traj[:,2]
    hits = []
    times = []
    last_t = None
    for t in range(1, len(traj)):
        if z[t-1] < z0 and z[t] >= z0:
            hits.append(traj[t,:2].copy())
            if last_t is not None:
                times.append(t - last_t)
            last_t = t
    hits = np.array(hits) if len(hits) else np.zeros((0,2))
    times = np.array(times) if len(times) else np.zeros((0,))
    return hits, times

def sinkhorn_3d(a, b, blur=1.0, n=5000, seed=0):
    if not USE_SINKHORN_3D:
        return None
    rng = np.random.default_rng(seed)
    na = min(n, len(a))
    nb = min(n, len(b))
    ia = rng.choice(len(a), size=na, replace=False)
    ib = rng.choice(len(b), size=nb, replace=False)
    A = torch.tensor(a[ia], dtype=torch.float32)
    B = torch.tensor(b[ib], dtype=torch.float32)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur)
    return float(loss(A, B).item())

print("\n" + "="*60)
print("                 METRICS (Robust for chaos)")
print("="*60)
wins = [50, 100, 200, 500, 1000, 2000, 5000]

print("\n[1] Short-window MSE (multiple horizons)")
for win in wins:
    w = min(win, len(traj_ours), len(gt_ours), len(traj_base), len(gt_base))
    mse_ours_avg, _ = short_window_mse(traj_ours, gt_ours, win=w)
    mse_base_avg, _ = short_window_mse(traj_base, gt_base, win=w)
    print(f"  first {w:5d} steps | Ours: {mse_ours_avg:.6f} | Baseline: {mse_base_avg:.6f}")

win = min(1000, len(traj_ours), len(gt_ours), len(traj_base), len(gt_base))
mse_ours_avg, mse_ours_curve = short_window_mse(traj_ours, gt_ours, win=win)
mse_base_avg, mse_base_curve = short_window_mse(traj_base, gt_base, win=win)

div_ours = divergence_time(traj_ours, gt_ours, thresholds=(1.0, 5.0, 10.0))
div_base = divergence_time(traj_base, gt_base, thresholds=(1.0, 5.0, 10.0))
print("\n[2] Divergence time (first crossing of ||err|| > threshold)")
print("  Ours     :", div_ours)
print("  Baseline :", div_base)

im_ours = w1_marginals(traj_ours, gt_ours)
im_base = w1_marginals(traj_base, gt_base)
print("\n[3] Invariant measure: marginal W1 distances (lower is better)")
print("  Ours vs GT     :", im_ours)
print("  Baseline vs GT :", im_base)

if USE_SINKHORN_3D:
    s_ours = sinkhorn_3d(traj_ours, gt_ours, blur=2.0, n=4000, seed=1)
    s_base = sinkhorn_3d(traj_base, gt_base, blur=2.0, n=4000, seed=2)
    print("\n[4] 3D Sinkhorn (subsampled, blur=2.0)  (lower is better)")
    print(f"  Ours vs GT     : {s_ours:.6f}")
    print(f"  Baseline vs GT : {s_base:.6f}")
else:
    print("\n[4] 3D Sinkhorn skipped (torch/geomloss not available).")

vel_ours = speed(traj_ours)
vel_base = speed(traj_base)
vel_gt   = speed(gt_ours)   

wd_vel_ours = wasserstein_distance(vel_ours, vel_gt)
wd_vel_base = wasserstein_distance(vel_base, vel_gt)
print("\n[5] Local dynamics: speed distribution W1 (lower is better)")
print(f"  Ours     : {wd_vel_ours:.6f}")
print(f"  Baseline : {wd_vel_base:.6f}")

curv_ours = curvature_proxy(traj_ours)
curv_base = curvature_proxy(traj_base)
curv_gt   = curvature_proxy(gt_ours)
wd_curv_ours = wasserstein_distance(curv_ours, curv_gt)
wd_curv_base = wasserstein_distance(curv_base, curv_gt)
print("\n[6] Local dynamics: curvature/turning proxy W1 (lower is better)")
print(f"  Ours     : {wd_curv_ours:.6f}")
print(f"  Baseline : {wd_curv_base:.6f}")

psd_ours = log_psd_distance(traj_ours[:,0], gt_ours[:,0], dt=dt, nfft=8192)
psd_base = log_psd_distance(traj_base[:,0], gt_base[:,0], dt=dt, nfft=8192)
print("\n[7] Spectral distance (log-PSD of x(t), L2, lower is better)")
print(f"  Ours     : {psd_ours:.6f}")
print(f"  Baseline : {psd_base:.6f}")

hits_ours, rt_ours = poincare_section(traj_ours, z0=25.0)
hits_base, rt_base = poincare_section(traj_base, z0=25.0)
hits_gt,   rt_gt   = poincare_section(gt_ours,   z0=25.0)

def w1_hits(hits_a, hits_b):
    if len(hits_a)==0 or len(hits_b)==0:
        return None
    wx = wasserstein_distance(hits_a[:,0], hits_b[:,0])
    wy = wasserstein_distance(hits_a[:,1], hits_b[:,1])
    return float(wx), float(wy)

w_hits_ours = w1_hits(hits_ours, hits_gt)
w_hits_base = w1_hits(hits_base, hits_gt)

print("\n[8] Poincaré section z=25 upward-crossing: W1 marginals on (x,y)  (lower better)")
print("  Ours     :", w_hits_ours, f"  (nhits={len(hits_ours)})")
print("  Baseline :", w_hits_base, f"  (nhits={len(hits_base)})")

if len(rt_ours)>0 and len(rt_gt)>0:
    w_rt_ours = wasserstein_distance(rt_ours, rt_gt)
else:
    w_rt_ours = None
if len(rt_base)>0 and len(rt_gt)>0:
    w_rt_base = wasserstein_distance(rt_base, rt_gt)
else:
    w_rt_base = None

print("\n[9] Return-time distribution on Poincaré section: W1 (lower better)")
print("  Ours     :", w_rt_ours)
print("  Baseline :", w_rt_base)

print("\nDone.")

DO_PLOT = True
if DO_PLOT:
    t = np.arange(min_len) * dt

    plt.figure(figsize=(12,4))
    plt.plot(t[:win], mse_ours_curve[:win], label="ours")
    plt.plot(t[:win], mse_base_curve[:win], label="baseline")
    plt.title(f"MSE curve (first {win} steps)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(10,4))
    plt.hist(vel_gt, bins=60, density=True, alpha=0.4, label="GT")
    plt.hist(vel_base, bins=60, density=True, histtype="step", label="baseline")
    plt.hist(vel_ours, bins=60, density=True, histtype="step", linewidth=2, label="ours")
    plt.title("Speed distribution")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.hist(curv_gt, bins=60, density=True, alpha=0.4, label="GT")
    plt.hist(curv_base, bins=60, density=True, histtype="step", label="baseline")
    plt.hist(curv_ours, bins=60, density=True, histtype="step", linewidth=2, label="ours")
    plt.title("Curvature proxy distribution")
    plt.legend()
    plt.show()


