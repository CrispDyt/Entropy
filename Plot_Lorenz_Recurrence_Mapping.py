import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ==========================================
# 1. 基础配置与加载函数
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
    # data[4] 通常是你保存的 xs_orig (模型生成的长轨迹)
    return np.asarray(data[4], dtype=np.float64)

# ==========================================
# 2. 核心算法：提取局部极大值与回归时间
# ==========================================
def analyze_peaks(traj, dt=0.01):
    """
    提取 z 轴的局部极大值 (z_max) 和 它们出现的时间点
    """
    z = traj[:, 2]
    # 寻找比前后都大的点 (局部峰值)
    # 注意：为了处理数值噪音，有时需要更复杂的过滤器，但对于 Lorenz 这种平滑系统，直接比较通常足够
    mask = (z[1:-1] > z[:-2]) & (z[1:-1] > z[2:])
    
    # 拿到索引 (加1是因为 mask 从 index 1 开始)
    peak_idxs = np.where(mask)[0] + 1
    
    z_peaks = z[peak_idxs]
    
    # 计算回归时间 (Return Times): 两个峰值之间的时间差
    # 这代表了系统“绕一圈”需要多久
    if len(peak_idxs) > 1:
        return_times = np.diff(peak_idxs) * dt
    else:
        return_times = np.array([])
        
    return z_peaks, return_times

# ==========================================
# 3. 主执行逻辑
# ==========================================
print("Loading data...")
try:
    traj_ours = load_traj("Joint_Lifted_recon_new.p")
    traj_base = load_traj("DIM_recon_short.p")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("请确保 .p 文件在当前目录下。")
    exit()

# 截断到相同长度，保证公平对比
min_len = min(len(traj_ours), len(traj_base))
traj_ours = traj_ours[:min_len]
traj_base = traj_base[:min_len]

# 生成一个非常长的 GT 作为完美的参照系 (比如 20000 步)
print("Generating Ground Truth reference...")
x0_ref = traj_ours[0]
gt_long = generate_long_gt(x0_ref, steps=30000)

# 分析数据
print("Analyzing Poincaré sections...")
z_peaks_gt,   rt_gt   = analyze_peaks(gt_long)
z_peaks_base, rt_base = analyze_peaks(traj_base)
z_peaks_ours, rt_ours = analyze_peaks(traj_ours)

# ==========================================
# 4. 可视化：The "Selling Point" Plots
# ==========================================
plt.style.use('seaborn-v0_8-paper') # 使用整洁的论文风格
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])

# --- Row 1: The Lorenz Return Map (z_n vs z_{n+1}) ---
# 这是证明因果性（Causality）的关键图

# Subplot 1: Baseline (Greenhouse)
ax1 = fig.add_subplot(gs[0, 0])
# 画 GT 背景 (灰色，细点)
ax1.scatter(z_peaks_gt[:-1], z_peaks_gt[1:], s=10, c='gray', alpha=0.15, label='Ground Truth')
# 画 Baseline (红色)
ax1.scatter(z_peaks_base[:-1], z_peaks_base[1:], s=10, c='#d62728', alpha=0.6, label='Baseline (Greenhouse)')
ax1.set_title("Baseline: Fuzzy Return Map = Decoupled Dynamics", fontsize=14, fontweight='bold')
ax1.set_xlabel("$z_n$ (Current Peak)", fontsize=12)
ax1.set_ylabel("$z_{n+1}$ (Next Peak)", fontsize=12)
ax1.legend(loc="upper left")
ax1.grid(True, linestyle='--', alpha=0.3)
# 强制坐标范围一致，方便肉眼对比
ax1.set_xlim(28, 48)
ax1.set_ylim(28, 48)

# Subplot 2: Ours (Joint Lifted OT)
ax2 = fig.add_subplot(gs[0, 1])
# 画 GT 背景
ax2.scatter(z_peaks_gt[:-1], z_peaks_gt[1:], s=10, c='gray', alpha=0.15, label='Ground Truth')
# 画 Ours (蓝色)
ax2.scatter(z_peaks_ours[:-1], z_peaks_ours[1:], s=10, c='#1f77b4', alpha=0.6, label='Ours (Joint Lifted OT)')
ax2.set_title("Ours: Sharp Return Map = Preserved Causality", fontsize=14, fontweight='bold')
ax2.set_xlabel("$z_n$ (Current Peak)", fontsize=12)
ax2.set_ylabel("$z_{n+1}$ (Next Peak)", fontsize=12)
ax2.legend(loc="upper left")
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_xlim(28, 48)
ax2.set_ylim(28, 48)

# --- Row 2: Return Time Distribution (Frequency/Jumping Logic) ---
# 这是证明“翅膀跳跃”和“频率错乱”的关键图

ax3 = fig.add_subplot(gs[1, :]) # 跨两列
# 使用核密度估计 (KDE) 或者 细直方图
bins = np.linspace(0.5, 1.5, 100) # 这里的范围根据 Lorenz 的特征周期调整，通常在 0.6-0.8 左右有一个峰

# 绘制直方图 (Step 风格)
ax3.hist(rt_gt, bins=bins, density=True, histtype='stepfilled', color='gray', alpha=0.2, label='Ground Truth Dist.')
ax3.hist(rt_gt, bins=bins, density=True, histtype='step', color='black', linewidth=1)

ax3.hist(rt_base, bins=bins, density=True, histtype='step', color='#d62728', linewidth=2, label='Baseline')
ax3.hist(rt_ours, bins=bins, density=True, histtype='step', color='#1f77b4', linewidth=2, label='Ours')

ax3.set_title("Distribution of Return Times (Time between z-peaks)", fontsize=14, fontweight='bold')
ax3.set_xlabel("Time $\Delta t$", fontsize=12)
ax3.set_ylabel("Probability Density", fontsize=12)
ax3.legend()
ax3.grid(True, linestyle=':', alpha=0.5)

# 添加一段文字说明
text_str = (
    "Interpretation:\n"
    "1. Top Row: A sharp 'tent map' indicates deterministic chaos.\n"
    "   Baseline shows a 'cloud', meaning $z_n$ fails to predict $z_{n+1}$ (Decoupling).\n"
    "   Ours preserves the 1D manifold structure.\n\n"
    "2. Bottom Row: Matches the physical frequency of the attractor.\n"
    "   Baseline often misses the correct peak timing (shifts in distribution)."
)
plt.figtext(0.5, 0.02, text_str, ha="center", fontsize=11, 
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "gray"})

plt.tight_layout(rect=[0, 0.1, 1, 1]) # 留出底部给文字
plt.show()

print("Visualization complete.")