import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn

# ==========================================
# 0. 全局配置
# ==========================================
EVAL_STEPS = 5000  # 评估长度
DT = 0.01

USE_SINKHORN_3D = True
try:
    from geomloss import SamplesLoss
except Exception:
    print("Warning: geomloss not found. Sinkhorn metric will be skipped.")
    USE_SINKHORN_3D = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# ==========================================
# 1. 动力学核心
# ==========================================
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

# ==========================================
# 2. 智能加载器
# ==========================================
def load_traj_smart(filename, steps=5000, x0_start=None):
    print(f"Loading {filename}...")
    with open(filename, "rb") as f:
        data = pickle.load(f)

    # 旧版 Baseline (List)
    if isinstance(data, (list, tuple)):
        print("  -> Format: Legacy List (Baseline).")
        return np.asarray(data[4], dtype=np.float64)

    # 新版 Ours (Dict)
    elif isinstance(data, dict) and "model" in data:
        print("  -> Format: Model State Dict (Ours). Generating trajectory...")
        transformer = data["model"]["transformer"]
        model = nn.Sequential(
            nn.Linear(3, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 3),
        ).to(device)
        model.load_state_dict(data["model"]["state_dict"])
        model.eval()
        
        if x0_start is None:
            x0_phys = np.array([-5.065457, -7.56735, 19.060379])
        else:
            x0_phys = x0_start
            
        x0_scaled = transformer.transform(x0_phys.reshape(1, -1)).flatten()
        curr = torch.tensor(x0_scaled, dtype=torch.float32, device=device)
        traj_scaled = [curr.cpu().detach().numpy()]
        
        with torch.no_grad():
            for _ in range(steps - 1):
                pred_vel = model(curr)
                curr = curr + DT * pred_vel
                traj_scaled.append(curr.cpu().detach().numpy())
                
        traj_scaled = np.array(traj_scaled)
        return transformer.inverse_transform(traj_scaled)
    else:
        raise ValueError(f"Unknown file format in {filename}")

# ==========================================
# 3. 全套指标函数 (已找回所有丢失项)
# ==========================================
def short_window_mse(traj, gt, win=2000):
    length = min(len(traj), len(gt), win)
    e = np.sum((traj[:length] - gt[:length])**2, axis=1)
    return float(np.mean(e)), e  # Return mean AND curve

def divergence_time(traj, gt, thresholds=(1.0, 5.0, 10.0)):
    length = min(len(traj), len(gt))
    err = np.linalg.norm(traj[:length] - gt[:length], axis=1)
    out = {}
    for th in thresholds:
        if err[0] > th: out[th] = 0
        elif (err > th).any(): out[th] = int(np.argmax(err > th))
        else: out[th] = None
    return out

def w1_marginals(traj_a, traj_b):
    l = min(len(traj_a), len(traj_b))
    xa, ya, za = traj_a[:l,0], traj_a[:l,1], traj_a[:l,2]
    xb, yb, zb = traj_b[:l,0], traj_b[:l,1], traj_b[:l,2]
    ra = np.sqrt(xa**2 + ya**2 + za**2)
    rb = np.sqrt(xb**2 + yb**2 + zb**2)
    return {
        "x": wasserstein_distance(xa, xb),
        "y": wasserstein_distance(ya, yb),
        "z": wasserstein_distance(za, zb),
        "r": wasserstein_distance(ra, rb),
    }

def sinkhorn_3d(a, b, blur=1.0, n=5000, seed=0):
    if not USE_SINKHORN_3D: return None
    rng = np.random.default_rng(seed)
    na, nb = min(n, len(a)), min(n, len(b))
    ia = rng.choice(len(a), size=na, replace=False)
    ib = rng.choice(len(b), size=nb, replace=False)
    A = torch.tensor(a[ia], dtype=torch.float32)
    B = torch.tensor(b[ib], dtype=torch.float32)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur)
    return float(loss(A, B).item())

# --- 找回的动力学指标 ---
def speed(traj, dt=0.01):
    v = np.diff(traj, axis=0) / dt
    return np.linalg.norm(v, axis=1)

def curvature_proxy(traj, dt=0.01):
    v = np.diff(traj, axis=0) / dt
    dv = np.diff(v, axis=0) / dt
    vnorm = np.linalg.norm(v[:-1], axis=1) + 1e-9
    return np.linalg.norm(dv, axis=1) / vnorm

def log_psd_distance(x, y, dt, nfft=4096):
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

# --- 截面分析逻辑 ---
def get_section_hits(traj, axis, value, direction=1):
    dim_data = traj[:, axis]
    hits = []
    times = [] # Keep track of indices for Return Time
    for t in range(1, len(traj)):
        if direction == 1:
            if dim_data[t-1] < value and dim_data[t] >= value:
                hits.append(traj[t])
                times.append(t)
        else:
            if dim_data[t-1] > value and dim_data[t] <= value:
                hits.append(traj[t])
                times.append(t)
    return (np.array(hits) if len(hits) else np.zeros((0, 3))), np.array(times)

def get_return_times(hit_indices, dt=0.01):
    if len(hit_indices) < 2: return np.zeros((0,))
    return np.diff(hit_indices) * dt

def w1_hits_2d(hits_a, hits_b, axis_ignored):
    if len(hits_a) == 0 or len(hits_b) == 0: return (None, None)
    dims = [0, 1, 2]
    dims.remove(axis_ignored)
    w_d1 = wasserstein_distance(hits_a[:, dims[0]], hits_b[:, dims[0]])
    w_d2 = wasserstein_distance(hits_a[:, dims[1]], hits_b[:, dims[1]])
    return (w_d1, w_d2)

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 加载
    try:
        traj_base = load_traj_smart("DIM_recon_short.p")
        traj_ours = load_traj_smart("DualOpt_Best_ep10000_minibatch16.p", steps=EVAL_STEPS, x0_start=traj_base[0])
    except FileNotFoundError as e:
        print(e)
        exit()

    base_len = len(traj_base)
    min_len = min(len(traj_ours), len(traj_base))
    
    # 2. 生成 GT
    print(f"Generating Ground Truth ({EVAL_STEPS} steps)...")
    traj_gt = generate_ground_truth_from_start_point(traj_base[0], EVAL_STEPS, dt=DT)
    
    print("\n" + "="*60)
    print("                FULL METRICS REPORT")
    print("="*60)
    
    # [1] MSE & Divergence
    print("\n[1] Short-window MSE")
    wins = [50, 100, 200, 500, 1000]
    for w in wins:
        m_ours, _ = short_window_mse(traj_ours, traj_gt, win=w)
        m_base, _ = short_window_mse(traj_base, traj_gt, win=w)
        print(f"  First {w:4d} steps | Ours: {m_ours:.6f} | Base: {m_base:.6f}")
        
    div_ours = divergence_time(traj_ours, traj_gt)
    div_base = divergence_time(traj_base, traj_gt[:base_len])
    print("\n[2] Divergence Time (Steps before error > threshold)")
    print(f"  Ours : {div_ours}")
    print(f"  Base : {div_base}")

    # [3] Distribution (W1 & Sinkhorn)
    im_ours = w1_marginals(traj_ours, traj_gt)
    im_base = w1_marginals(traj_base, traj_gt[:base_len])
    print("\n[3] Global Invariant Measure (W1)")
    print(f"  Ours : {im_ours}")
    print(f"  Base : {im_base}")

    if USE_SINKHORN_3D:
        s_ours = sinkhorn_3d(traj_ours, traj_gt, blur=2.0)
        s_base = sinkhorn_3d(traj_base, traj_gt[:base_len], blur=2.0)
        print(f"  Sinkhorn Loss (Blur=2.0) | Ours: {s_ours:.4f} | Base: {s_base:.4f}")

    # [4] Dynamics (Speed, Curvature, PSD) - RECOVERED
    print("\n[4] Local Dynamics & Spectral Analysis")
    
    # Speed
    v_ours, v_base, v_gt = speed(traj_ours), speed(traj_base), speed(traj_gt)
    wd_v_ours = wasserstein_distance(v_ours, v_gt)
    wd_v_base = wasserstein_distance(v_base, v_gt[:len(v_base)])
    print(f"  Speed Dist. W1      | Ours: {wd_v_ours:.6f} | Base: {wd_v_base:.6f}")
    
    # Curvature
    c_ours, c_base, c_gt = curvature_proxy(traj_ours), curvature_proxy(traj_base), curvature_proxy(traj_gt)
    wd_c_ours = wasserstein_distance(c_ours, c_gt)
    wd_c_base = wasserstein_distance(c_base, c_gt[:len(c_base)])
    print(f"  Curvature Dist. W1  | Ours: {wd_c_ours:.6f} | Base: {wd_c_base:.6f}")
    
    # PSD
    psd_ours = log_psd_distance(traj_ours[:,0], traj_gt[:,0], dt=DT)
    psd_base = log_psd_distance(traj_base[:,0], traj_gt[:base_len,0], dt=DT)
    print(f"  Log-PSD Distance    | Ours: {psd_ours:.6f} | Base: {psd_base:.6f}")

    # [5] Advanced Poincare Analysis
    print("\n" + "="*60)
    print("      [5] ADVANCED POINCARE & GENERALIZATION")
    print("="*60)
    
    # 准备绘图 (MSE + Hists + Sections)
    fig = plt.figure(figsize=(18, 10))
    # Layout: Top row for Dynamics, Bottom row for Sections
    ax_mse = plt.subplot2grid((2, 4), (0, 0))
    ax_vel = plt.subplot2grid((2, 4), (0, 1))
    ax_curv = plt.subplot2grid((2, 4), (0, 2))
    ax_sec1 = plt.subplot2grid((2, 4), (1, 0)) # Z=25
    ax_sec2 = plt.subplot2grid((2, 4), (1, 1)) # Z=20
    ax_sec3 = plt.subplot2grid((2, 4), (1, 2)) # X=0
    
    # Plot Dynamics
    # MSE Curve
    _, mse_curve_ours = short_window_mse(traj_ours, traj_gt, win=1000)
    _, mse_curve_base = short_window_mse(traj_base, traj_gt, win=1000)
    ax_mse.plot(mse_curve_ours, label="Ours")
    ax_mse.plot(mse_curve_base, label="Base")
    ax_mse.set_title("MSE Curve (First 1000 steps)")
    ax_mse.legend()
    ax_mse.set_yscale('log')
    
    # Speed Hist
    ax_vel.hist(v_gt, bins=50, density=True, alpha=0.3, color='k', label='GT')
    ax_vel.hist(v_ours, bins=50, density=True, histtype='step', color='r', label='Ours')
    ax_vel.hist(v_base, bins=50, density=True, histtype='step', color='orange', label='Base')
    ax_vel.set_title("Speed Distribution")
    
    # Curvature Hist
    ax_curv.hist(c_gt, bins=50, density=True, alpha=0.3, color='k', label='GT')
    ax_curv.hist(c_ours, bins=50, density=True, histtype='step', color='r', label='Ours')
    ax_curv.hist(c_base, bins=50, density=True, histtype='step', color='orange', label='Base')
    ax_curv.set_title("Curvature Distribution")

    # Analyze Sections
    sections = [
        ("Z=25 (Standard)", 2, 25.0, ax_sec1),
        ("Z=20 (Generaliz)", 2, 20.0, ax_sec2),
        ("X=0  (Symmetry)", 0,  0.0, ax_sec3),
    ]
    
    for name, axis, val, ax in sections:
        print(f"\n--- Section: {name} ---")
        h_gt, t_gt = get_section_hits(traj_gt, axis, val)
        h_ours, t_ours = get_section_hits(traj_ours, axis, val)
        h_base, t_base = get_section_hits(traj_base, axis, val)
        
        # Hits Stats
        print(f"  Hits Count | GT: {len(h_gt)} | Ours: {len(h_ours)} | Base: {len(h_base)}")
        
        # W1 on Slice
        w_ours = w1_hits_2d(h_ours, h_gt, axis)
        w_base = w1_hits_2d(h_base, h_gt, axis)
        if w_ours[0] is not None:
             print(f"  W1 (Slice) | Ours: ({w_ours[0]:.4f}, {w_ours[1]:.4f})")
        if w_base[0] is not None:
             print(f"  W1 (Slice) | Base: ({w_base[0]:.4f}, {w_base[1]:.4f})")
             
        # Return Time (Recovered Metric!)
        rt_gt = get_return_times(t_gt, DT)
        rt_ours = get_return_times(t_ours, DT)
        rt_base = get_return_times(t_base, DT)
        
        if len(rt_gt) > 5 and len(rt_ours) > 5:
            w_rt_ours = wasserstein_distance(rt_gt, rt_ours)
            print(f"  Return Time W1 | Ours: {w_rt_ours:.6f}")
        if len(rt_gt) > 5 and len(rt_base) > 5:
            w_rt_base = wasserstein_distance(rt_gt, rt_base)
            print(f"  Return Time W1 | Base: {w_rt_base:.6f}")

        # Plot Scatter
        dims = [0, 1, 2]
        dims.remove(axis)
        d1, d2 = dims[0], dims[1]
        
        ax.set_title(name)
        if len(h_gt)>0: ax.scatter(h_gt[:,d1], h_gt[:,d2], s=10, c='k', alpha=0.2, label='GT')
        if len(h_ours)>0: ax.scatter(h_ours[:,d1], h_ours[:,d2], s=10, c='r', alpha=0.5, label='Ours')
        if len(h_base)>0: ax.scatter(h_base[:,d1], h_base[:,d2], s=10, c='orange', marker='x', alpha=0.5, label='Base')
        if name == "Z=25 (Standard)": ax.legend()

    plt.tight_layout()
    plt.show()
    print("\nDone. All metrics restored.")