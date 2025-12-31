import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn

# ==========================================
# 0. 配置与依赖检查
# ==========================================
USE_SINKHORN_3D = True
try:
    from geomloss import SamplesLoss
except Exception:
    print("Warning: geomloss not found. Sinkhorn metric will be skipped.")
    USE_SINKHORN_3D = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# ==========================================
# 1. 动力学方程与GT生成
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
# 2. 加载 Baseline (旧格式)
# ==========================================
def load_baseline_traj(filename):
    print(f"Loading Baseline from {filename}...")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    # 你的旧代码保存的是: pickle.dump([y, Ty_true, y_delay_full, y[ixs], xs, ...], f)
    # 所以轨迹 xs 在索引 4
    return np.asarray(data[4], dtype=np.float64)

# ==========================================
# 3. 加载 Ours (新格式) 并生成轨迹
# ==========================================
def load_model_and_generate_traj(filename, x0_start, steps, dt=0.01):
    print(f"Loading Ours model from {filename}...")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    # 1. 恢复 Scaler
    if "model" not in data or "transformer" not in data["model"]:
        raise ValueError("Pickle file structure error.")
    transformer = data["model"]["transformer"]
    
    # 2. 定义网络 (必须与训练代码完全一致)
    # 训练代码用的是直接的 Sequential，这里也照搬
    model = nn.Sequential(
        nn.Linear(3, 100), nn.Tanh(),
        nn.Linear(100, 100), nn.Tanh(),
        nn.Linear(100, 100), nn.Tanh(),
        nn.Linear(100, 3),
    ).to(device)
    
    # 3. 加载权重
    # 因为没有 SimpleNet 包装，现在 key 应该能完美匹配了 (0.weight 等)
    model.load_state_dict(data["model"]["state_dict"])
    model.eval()
    
    # 4. 准备初始条件 (使用 Baseline 的起点)
    # 先归一化
    x0_scaled = transformer.transform(x0_start.reshape(1, -1)).flatten()
    
    # 5. 生成轨迹
    curr = torch.tensor(x0_scaled, dtype=torch.float32, device=device)
    traj_scaled = [curr.cpu().detach().numpy()]
    
    print(f"Generating {steps} steps using Ours model...")
    with torch.no_grad():
        for _ in range(steps - 1):
            pred_vel = model(curr)
            curr = curr + dt * pred_vel
            traj_scaled.append(curr.cpu().detach().numpy())
            
    traj_scaled = np.array(traj_scaled)
    
    # 6. 反归一化
    traj_orig = transformer.inverse_transform(traj_scaled)
    return traj_orig

# ==========================================
# 4. 指标计算函数集
# ==========================================
def short_window_mse(traj, gt, win=2000):
    e = np.sum((traj[:win] - gt[:win])**2, axis=1)
    return float(np.mean(e)), e

def divergence_time(traj, gt, thresholds=(1.0, 5.0, 10.0)):
    err = np.linalg.norm(traj - gt, axis=1)
    out = {}
    for th in thresholds:
        if err[0] > th:
            out[th] = 0
        elif (err > th).any():
            out[th] = int(np.argmax(err > th))
        else:
            out[th] = None
    return out

def w1_marginals(traj_a, traj_b):
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

def speed(traj, dt=0.01):
    v = np.diff(traj, axis=0) / dt
    return np.linalg.norm(v, axis=1)

def curvature_proxy(traj, dt=0.01):
    v = np.diff(traj, axis=0) / dt
    dv = np.diff(v, axis=0) / dt
    vnorm = np.linalg.norm(v[:-1], axis=1) + 1e-9
    return np.linalg.norm(dv, axis=1) / vnorm

def log_psd_distance(x, y, dt, nfft=4096):
    x, y = np.asarray(x), np.asarray(y)
    n = min(len(x), len(y), nfft)
    x, y = x[:n] - np.mean(x[:n]), y[:n] - np.mean(y[:n])
    fx, fy = np.fft.rfft(x), np.fft.rfft(y)
    psdx, psdy = (fx*np.conj(fx)).real + 1e-12, (fy*np.conj(fy)).real + 1e-12
    lx, ly = np.log(psdx), np.log(psdy)
    lx, ly = (lx - lx.mean())/(lx.std()+1e-9), (ly - ly.mean())/(ly.std()+1e-9)
    return float(np.sqrt(np.mean((lx - ly)**2)))

def poincare_section(traj, z0=25.0):
    z = traj[:,2]
    hits, times = [], []
    last_t = None
    for t in range(1, len(traj)):
        if z[t-1] < z0 and z[t] >= z0:
            hits.append(traj[t,:2].copy())
            if last_t is not None: times.append(t - last_t)
            last_t = t
    return (np.array(hits) if len(hits) else np.zeros((0,2))), (np.array(times) if len(times) else np.zeros((0,)))

def w1_hits(hits_a, hits_b):
    if len(hits_a)==0 or len(hits_b)==0: return None
    return (wasserstein_distance(hits_a[:,0], hits_b[:,0]), 
            wasserstein_distance(hits_a[:,1], hits_b[:,1]))

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    dt = 0.01
    
    # 1. 加载 Baseline (为了获取公共起点)
    try:
        # 这里改成了你提供的文件名 "DIM_recon.p" 还是 "DIM_recon_short.p"
        # 请根据实际情况修改文件名，如果你刚才代码生成的是 DIM_recon.p 就改这里
        traj_base = load_baseline_traj("DIM_recon.p") 
    except FileNotFoundError:
        try:
            traj_base = load_baseline_traj("DIM_recon_short.p")
        except FileNotFoundError:
            print("Error: Baseline file not found.")
            exit()
    
    # 2. 提取信息
    x0_common = traj_base[0]  
    total_len = len(traj_base)
    print(f"Baseline length: {total_len}")
    print(f"Start point (Physical): {x0_common}")
    
    # 3. 加载 Ours (DualOpt)
    try:
        traj_ours = load_model_and_generate_traj(
            "DualOpt_MultiShoot_Auto.p", 
            x0_start=x0_common, 
            steps=total_len, 
            dt=dt
        )
    except FileNotFoundError:
        print("Error: 'DualOpt_MultiShoot_Auto.p' not found.")
        exit()
        
    # 4. 生成 Ground Truth
    print("Generating Ground Truth...")
    gt_traj = generate_ground_truth_from_start_point(x0_common, total_len, dt=dt)
    
    # 长度对齐
    min_len = min(len(traj_ours), len(traj_base), len(gt_traj))
    traj_ours = traj_ours[:min_len]
    traj_base = traj_base[:min_len]
    gt_traj   = gt_traj[:min_len]
    
    print(f"Comparing first {min_len} steps.\n")
    
    # 5. 指标计算
    print("="*60)
    print("                METRICS REPORT")
    print("="*60)
    
    # [1] MSE
    wins = [50, 100, 200, 500, 1000]
    print("\n[1] Short-window MSE")
    for w in wins:
        if w > min_len: break
        m_ours, _ = short_window_mse(traj_ours, gt_traj, win=w)
        m_base, _ = short_window_mse(traj_base, gt_traj, win=w)
        print(f"  Steps {w:4d} | Ours: {m_ours:.6f} | Base: {m_base:.6f}")
        
    # [2] Divergence Time
    div_ours = divergence_time(traj_ours, gt_traj)
    div_base = divergence_time(traj_base, gt_traj)
    print("\n[2] Divergence Time (Steps before error > threshold)")
    print(f"  Ours : {div_ours}")
    print(f"  Base : {div_base}")
    
    # [3] Invariant Measure
    im_ours = w1_marginals(traj_ours, gt_traj)
    im_base = w1_marginals(traj_base, gt_traj)
    print("\n[3] Global Distribution W1")
    print(f"  Ours : {im_ours}")
    print(f"  Base : {im_base}")
    
    # [4] Sinkhorn
    if USE_SINKHORN_3D:
        s_ours = sinkhorn_3d(traj_ours, gt_traj, blur=2.0)
        s_base = sinkhorn_3d(traj_base, gt_traj, blur=2.0)
        print("\n[4] 3D Sinkhorn Distance")
        print(f"  Ours : {s_ours:.6f}")
        print(f"  Base : {s_base:.6f}")
        
    # [5] Dynamics
    v_ours, v_base, v_gt = speed(traj_ours), speed(traj_base), speed(gt_traj)
    c_ours, c_base, c_gt = curvature_proxy(traj_ours), curvature_proxy(traj_base), curvature_proxy(gt_traj)
    
    print("\n[5] Dynamics Distribution W1 (Speed / Curvature)")
    print(f"  Speed     | Ours: {wasserstein_distance(v_ours, v_gt):.6f} | Base: {wasserstein_distance(v_base, v_gt):.6f}")
    print(f"  Curvature | Ours: {wasserstein_distance(c_ours, c_gt):.6f} | Base: {wasserstein_distance(c_base, c_gt):.6f}")
    
    # [6] PSD
    psd_ours = log_psd_distance(traj_ours[:,0], gt_traj[:,0], dt=dt)
    psd_base = log_psd_distance(traj_base[:,0], gt_traj[:,0], dt=dt)
    print("\n[6] Spectral Distance (Log-PSD)")
    print(f"  Ours : {psd_ours:.6f}")
    print(f"  Base : {psd_base:.6f}")
    
    print("\nDone.")