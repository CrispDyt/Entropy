import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ==========================================
# 0) Optional: geomloss Sinkhorn for 2D joint OT
# ==========================================
USE_SINKHORN_2D = True
try:
    import torch
    from geomloss import SamplesLoss
except Exception:
    USE_SINKHORN_2D = False

# ==========================================
# 1) 基础配置与加载函数
# ==========================================
def lorenz_dynamics(xyz):
    s, r, b = 10.0, 28.0, 2.667
    x, y, z = xyz
    return np.array([s * (y - x), r * x - y - x * z, x * y - b * z], dtype=np.float64)

def generate_long_gt(x0, steps, dt=0.01):
    """生成足够长的 GT 用于画背景灰线"""
    y = np.zeros((steps, 3), dtype=np.float64)
    y[0] = x0
    curr = x0.copy()
    for i in range(steps - 1):
        curr = curr + lorenz_dynamics(curr) * dt
        y[i + 1] = curr
    return y

def load_traj(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return np.asarray(data[4], dtype=np.float64)  # xs_orig

# ==========================================
# 2) 核心算法：提取局部极大值与回归时间
# ==========================================
def analyze_peaks(traj, dt=0.01):
    """
    提取 z 轴的局部极大值 z_peaks 和峰值间隔 return_times
    """
    z = traj[:, 2]
    mask = (z[1:-1] > z[:-2]) & (z[1:-1] > z[2:])
    peak_idxs = np.where(mask)[0] + 1
    z_peaks = z[peak_idxs]
    if len(peak_idxs) > 1:
        return_times = np.diff(peak_idxs) * dt
    else:
        return_times = np.array([])
    return z_peaks, return_times

def peak_pairs(z_peaks):
    """Return array of shape [N-1,2] with (z_n, z_{n+1})."""
    if len(z_peaks) < 2:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([z_peaks[:-1], z_peaks[1:]], axis=1)

# ==========================================
# 3) Quantitative metrics
# ==========================================
def sinkhorn_2d(points_a, points_b, blur=0.5, p=2, device="cpu"):
    """
    2D Sinkhorn divergence on point clouds using geomloss.
    Returns float or None if unavailable.
    """
    if not USE_SINKHORN_2D:
        return None
    if len(points_a) == 0 or len(points_b) == 0:
        return None
    A = torch.tensor(points_a, dtype=torch.float32, device=device)
    B = torch.tensor(points_b, dtype=torch.float32, device=device)
    loss = SamplesLoss(loss="sinkhorn", p=p, blur=float(blur))
    return float(loss(A, B).item())

def conditional_variance_profile(z_pairs, nbins=40, zmin=None, zmax=None, min_count=30):
    """
    Compute Var(z_{n+1} | z_n in bin) across bins.
    Returns:
      bin_centers, cond_var (NaN where insufficient), counts, and scalar weighted_avg_var.
    """
    if len(z_pairs) == 0:
        return None, None, None, None
    zn = z_pairs[:, 0]
    znp1 = z_pairs[:, 1]
    if zmin is None: zmin = float(np.min(zn))
    if zmax is None: zmax = float(np.max(zn))
    edges = np.linspace(zmin, zmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    cond_var = np.full(nbins, np.nan, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    # assign bins
    bin_id = np.clip(np.digitize(zn, edges) - 1, 0, nbins - 1)
    for b in range(nbins):
        idx = np.where(bin_id == b)[0]
        counts[b] = len(idx)
        if counts[b] >= min_count:
            cond_var[b] = float(np.var(znp1[idx], ddof=0))

    # scalar: weighted average of conditional variance over valid bins
    valid = np.isfinite(cond_var)
    if np.any(valid):
        w = counts[valid].astype(np.float64)
        weighted_avg_var = float(np.sum(w * cond_var[valid]) / np.sum(w))
    else:
        weighted_avg_var = None

    return centers, cond_var, counts, weighted_avg_var

def mutual_information_hist2d(z_pairs, nbins=60, zmin=None, zmax=None, eps=1e-12):
    """
    Discretized mutual information I(Zn; Z_{n+1}) using 2D histogram (in nats).
    Returns MI or None if empty.
    """
    if len(z_pairs) == 0:
        return None
    x = z_pairs[:, 0]
    y = z_pairs[:, 1]
    if zmin is None: zmin = float(min(x.min(), y.min()))
    if zmax is None: zmax = float(max(x.max(), y.max()))
    # Joint histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=[[zmin, zmax], [zmin, zmax]])
    Pxy = H / (H.sum() + eps)
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)

    # MI = sum Pxy log(Pxy/(PxPy))
    ratio = (Pxy + eps) / (Px @ Py + eps)
    mi = float(np.sum(Pxy * np.log(ratio)))
    return mi

# ==========================================
# 4) 主执行逻辑
# ==========================================
print("Loading data...")
traj_ours = load_traj("Joint_Lifted_recon_new.p")
traj_base = load_traj("DIM_recon_short.p")

# 截断到相同长度，保证公平对比
min_len = min(len(traj_ours), len(traj_base))
traj_ours = traj_ours[:min_len]
traj_base = traj_base[:min_len]

dt = 0.01

# 生成一个长 GT 作为参照（吸引子统计足够长更稳）
print("Generating Ground Truth reference...")
x0_ref = traj_ours[0]
gt_long = generate_long_gt(x0_ref, steps=30000, dt=dt)

# --- burn-in: drop transients before peak analysis (recommended)
burn_in = 2000
traj_ours_b = traj_ours[burn_in:]
traj_base_b = traj_base[burn_in:]
gt_long_b   = gt_long[burn_in:]

# 分析 peaks
print("Analyzing peaks...")
z_peaks_gt,   rt_gt   = analyze_peaks(gt_long_b, dt=dt)
z_peaks_base, rt_base = analyze_peaks(traj_base_b, dt=dt)
z_peaks_ours, rt_ours = analyze_peaks(traj_ours_b, dt=dt)

pairs_gt   = peak_pairs(z_peaks_gt)
pairs_base = peak_pairs(z_peaks_base)
pairs_ours = peak_pairs(z_peaks_ours)

# ==========================================
# 5) Quantitative metrics on return map (conditional structure)
# ==========================================
print("\n" + "="*70)
print("QUANTITATIVE METRICS FOR PEAK-TO-PEAK CONDITIONAL STRUCTURE")
print("="*70)

# Set common plotting/metric range (optional but stabilizes hist/MI)
zmin, zmax = 28.0, 48.0

# (A) 2D Sinkhorn distance between return-map point clouds
if USE_SINKHORN_2D:
    dev = "cuda" if (torch.cuda.is_available()) else "cpu"
    # blur can be tuned; 0.5 ~ 2.0 typical depending on scaling
    s_base = sinkhorn_2d(pairs_base, pairs_gt, blur=0.8, p=2, device=dev)
    s_ours = sinkhorn_2d(pairs_ours, pairs_gt, blur=0.8, p=2, device=dev)
    print(f"[A] 2D Sinkhorn on (z_n,z_{{n+1}}) vs GT (lower better), blur=0.8")
    print(f"    Baseline: {s_base:.6f}")
    print(f"    Ours    : {s_ours:.6f}")
else:
    print("[A] 2D Sinkhorn skipped (torch/geomloss not available).")

# (B) Conditional variance profile: Var(z_{n+1} | z_n in bin)
cent_gt, var_gt, cnt_gt, wvar_gt = conditional_variance_profile(
    pairs_gt, nbins=40, zmin=zmin, zmax=zmax, min_count=30
)
cent_b, var_b, cnt_b, wvar_b = conditional_variance_profile(
    pairs_base, nbins=40, zmin=zmin, zmax=zmax, min_count=30
)
cent_o, var_o, cnt_o, wvar_o = conditional_variance_profile(
    pairs_ours, nbins=40, zmin=zmin, zmax=zmax, min_count=30
)

# Scalar comparisons: (i) weighted avg conditional variance, (ii) L2 distance to GT profile
def profile_l2_to_gt(var_model, var_gt):
    if var_model is None or var_gt is None:
        return None
    mask = np.isfinite(var_model) & np.isfinite(var_gt)
    if not np.any(mask):
        return None
    return float(np.sqrt(np.mean((var_model[mask] - var_gt[mask])**2)))

l2_base = profile_l2_to_gt(var_b, var_gt)
l2_ours = profile_l2_to_gt(var_o, var_gt)

print("\n[B] Conditional variance Var(z_{n+1} | z_n in bin)")
print("    Scalar 1: weighted avg conditional variance (lower = sharper return map)")
print(f"    GT      : {wvar_gt:.6f}" if wvar_gt is not None else "    GT      : None")
print(f"    Baseline: {wvar_b:.6f}"  if wvar_b  is not None else "    Baseline: None")
print(f"    Ours    : {wvar_o:.6f}"  if wvar_o  is not None else "    Ours    : None")
print("    Scalar 2: L2 distance of variance profile to GT (lower better)")
print(f"    Baseline: {l2_base:.6f}" if l2_base is not None else "    Baseline: None")
print(f"    Ours    : {l2_ours:.6f}" if l2_ours is not None else "    Ours    : None")

# (C) Mutual information I(z_n; z_{n+1}) via histogram
mi_gt   = mutual_information_hist2d(pairs_gt,   nbins=60, zmin=zmin, zmax=zmax)
mi_base = mutual_information_hist2d(pairs_base, nbins=60, zmin=zmin, zmax=zmax)
mi_ours = mutual_information_hist2d(pairs_ours, nbins=60, zmin=zmin, zmax=zmax)

print("\n[C] Mutual Information I(z_n; z_{n+1}) (hist2d, nats; higher = stronger dependence)")
print(f"    GT      : {mi_gt:.6f}"   if mi_gt   is not None else "    GT      : None")
print(f"    Baseline: {mi_base:.6f}" if mi_base is not None else "    Baseline: None")
print(f"    Ours    : {mi_ours:.6f}" if mi_ours is not None else "    Ours    : None")

# ==========================================
# 6) Visualization: original 2x return map + return-time + add metric text
# ==========================================
plt.style.use('seaborn-v0_8-paper')
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])

# --- Row 1: Return Map (z_n vs z_{n+1})
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(pairs_gt[:,0], pairs_gt[:,1], s=10, c='gray', alpha=0.15, label='Ground Truth')
ax1.scatter(pairs_base[:,0], pairs_base[:,1], s=10, c='#d62728', alpha=0.6, label='Baseline (Greenhouse)')
ax1.set_title("Baseline: Fuzzy Return Map = Decoupled Dynamics", fontsize=14, fontweight='bold')
ax1.set_xlabel("$z_n$ (Current Peak)", fontsize=12)
ax1.set_ylabel("$z_{n+1}$ (Next Peak)", fontsize=12)
ax1.legend(loc="upper left")
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_xlim(zmin, zmax)
ax1.set_ylim(zmin, zmax)

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(pairs_gt[:,0], pairs_gt[:,1], s=10, c='gray', alpha=0.15, label='Ground Truth')
ax2.scatter(pairs_ours[:,0], pairs_ours[:,1], s=10, c='#1f77b4', alpha=0.6, label='Ours (Joint Lifted OT)')
ax2.set_title("Ours: Sharp Return Map = Preserved Conditional Structure", fontsize=14, fontweight='bold')
ax2.set_xlabel("$z_n$ (Current Peak)", fontsize=12)
ax2.set_ylabel("$z_{n+1}$ (Next Peak)", fontsize=12)
ax2.legend(loc="upper left")
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_xlim(zmin, zmax)
ax2.set_ylim(zmin, zmax)

# Add metric box (top-right area)
metric_lines = []
if USE_SINKHORN_2D:
    metric_lines.append(f"2D Sinkhorn↓  base={s_base:.3f}  ours={s_ours:.3f}")
metric_lines.append(f"CondVar(avg)↓ base={wvar_b:.3f}  ours={wvar_o:.3f}" if (wvar_b is not None and wvar_o is not None) else "CondVar(avg)↓  N/A")
metric_lines.append(f"MI (nats)↑     base={mi_base:.3f}  ours={mi_ours:.3f}" if (mi_base is not None and mi_ours is not None) else "MI↑  N/A")
fig.text(0.67, 0.92, "\n".join(metric_lines),
         fontsize=11, va="top",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))

# --- Row 2: Return time distribution
ax3 = fig.add_subplot(gs[1, :])
bins = np.linspace(0.5, 1.5, 100)
ax3.hist(rt_gt, bins=bins, density=True, histtype='stepfilled', color='gray', alpha=0.2, label='Ground Truth Dist.')
ax3.hist(rt_gt, bins=bins, density=True, histtype='step', color='black', linewidth=1)
ax3.hist(rt_base, bins=bins, density=True, histtype='step', color='#d62728', linewidth=2, label='Baseline')
ax3.hist(rt_ours, bins=bins, density=True, histtype='step', color='#1f77b4', linewidth=2, label='Ours')
ax3.set_title("Distribution of Return Times (Time between z-peaks)", fontsize=14, fontweight='bold')
ax3.set_xlabel("Time $\Delta t$", fontsize=12)
ax3.set_ylabel("Probability Density", fontsize=12)
ax3.legend()
ax3.grid(True, linestyle=':', alpha=0.5)

text_str = (
    "Interpretation:\n"
    "Top: Sharp concentration on a 1D manifold in (z_n, z_{n+1}) indicates strong conditional dependence.\n"
    "     Fuzzy clouds imply weakened state-dependent peak-to-peak transitions.\n"
    "Bottom: Return-time distribution captures temporal structure / recurrence statistics of the attractor.\n"
    f"Burn-in used: first {burn_in} steps removed before peak analysis."
)
plt.figtext(0.5, 0.02, text_str, ha="center", fontsize=11,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "gray"})

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()

# ==========================================
# 7) Optional: plot conditional variance profiles (very informative)
# ==========================================
DO_PLOT_CONDVAR = True
if DO_PLOT_CONDVAR and (cent_gt is not None):
    plt.figure(figsize=(10,4))
    # plot only valid points
    mgt = np.isfinite(var_gt)
    mb  = np.isfinite(var_b)
    mo  = np.isfinite(var_o)
    plt.plot(cent_gt[mgt], var_gt[mgt], label="GT Var(z_{n+1}|z_n bin)")
    plt.plot(cent_b[mb],   var_b[mb],   label="Baseline")
    plt.plot(cent_o[mo],   var_o[mo],   label="Ours")
    plt.xlabel("z_n bin center")
    plt.ylabel("Conditional variance of z_{n+1}")
    plt.title("Conditional Variance Profile: sharpness of the return map (lower is sharper)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

print("\nVisualization complete.")
