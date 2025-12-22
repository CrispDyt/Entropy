import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# ==========================================
# 0. Lorenz Solver (Standard)
# ==========================================
def lorenz_dynamics(xyz):
    s, r, b = 10.0, 28.0, 2.667
    x, y, z = xyz
    return np.array([s * (y - x), r * x - y - x * z, x * y - b * z])

def generate_ground_truth_from_start_point(x0, steps, dt=0.01):
    # Strictly Euler integration to match training simulation
    y = np.zeros((steps, 3))
    y[0] = x0
    curr = x0
    for i in range(steps - 1):
        curr = curr + lorenz_dynamics(curr) * dt
        y[i+1] = curr
    return y

# ==========================================
# 1. Load Data
# ==========================================
def load_traj(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    # index 4 is xs_orig (model generated trajectory in original coords)
    return data[4]

print("Loading data...")
try:
    traj_ours = load_traj("Joint_Lifted_recon.p")
    traj_base = load_traj("delay_only.p")
except FileNotFoundError:
    print("Error: .p files not found.")
    exit()

# Trim to same length
min_len = min(len(traj_ours), len(traj_base))
traj_ours = traj_ours[:min_len]
traj_base = traj_base[:min_len]

# ==========================================
# 2. CRITICAL FIX: Generate Matching GT for EACH model
# ==========================================
# 我们不能假设两个模型生成的轨迹起点完全一样（虽然代码里看起来是一样的）
# 即使它们有一点点浮点误差，在混沌系统里也会爆炸。
# 所以：一定要分别为它们生成对应的真值！

print("Generating matched Ground Truth for YOUR model...")
# 使用你模型轨迹的第一个点作为起点
x0_ours = traj_ours[0] 
gt_ours = generate_ground_truth_from_start_point(x0_ours, min_len, dt=0.01)

print("Generating matched Ground Truth for BASELINE model...")
# 使用Baseline轨迹的第一个点作为起点
x0_base = traj_base[0]
gt_base = generate_ground_truth_from_start_point(x0_base, min_len, dt=0.01)

# Consistency check (Uncomment to verify)
print("Baseline First Value:", x0_base[0])
print("Baseline GT First Value:", x0_ours[0])

# ==========================================
# 3. Compute Metrics
# ==========================================

# --- MSE (Forecast Error) ---
# Check t=0 error (Should be 0.0)
mse_ours_0 = np.sum((traj_ours[0] - gt_ours[0])**2)
mse_base_0 = np.sum((traj_base[0] - gt_base[0])**2)
print(f"\nSanity Check (Error at t=0, should be ~0.0):")
print(f"  Ours t=0 Error: {mse_ours_0:.6f}")
print(f"  Base t=0 Error: {mse_base_0:.6f}")

# Calculate MSE over time
mse_ours_t = np.sum((traj_ours - gt_ours)**2, axis=1)
mse_base_t = np.sum((traj_base - gt_base)**2, axis=1)

# Short-term window (e.g., 200 steps = 2 seconds)
win = 20000
avg_mse_ours = np.mean(mse_ours_t[:win])
avg_mse_base = np.mean(mse_base_t[:win])

# --- Velocity Distribution ---
vel_ours = np.linalg.norm(np.diff(traj_ours, axis=0), axis=1) / 0.01
vel_base = np.linalg.norm(np.diff(traj_base, axis=0), axis=1) / 0.01
vel_gt   = np.linalg.norm(np.diff(gt_ours, axis=0),   axis=1) / 0.01 # Use ours GT as reference

wd_ours = wasserstein_distance(vel_ours, vel_gt)
wd_base = wasserstein_distance(vel_base, vel_gt)

# ==========================================
# 4. Results
# ==========================================
print("\n" + "="*60)
print("             TRUE EVALUATION (Aligned Initial Conditions)")
print("="*60)
print(f"1. Short-term Forecast MSE (First {win} steps)")
print(f"   Ours:      {avg_mse_ours:.6f}")
print(f"   Baseline:  {avg_mse_base:.6f}")
print("-" * 60)
if avg_mse_ours < avg_mse_base:
    print(f"   >> WINNER: Ours (Joint Lifted) is better.")
else:
    print(f"   >> WINNER: Baseline is better (Something is wrong with weights).")

print("\n2. Velocity Distribution (Wasserstein)")
print(f"   Ours:      {wd_ours:.6f}")
print(f"   Baseline:  {wd_base:.6f}")
print("="*60)

# ==========================================
# 5. Plot
# ==========================================
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# t = np.arange(min_len) * 0.01
# plt.plot(t[:500], mse_ours_t[:500], 'r-', linewidth=2, label='Ours')
# plt.plot(t[:500], mse_base_t[:500], 'b--', linewidth=1, label='Baseline')
# plt.title("Short-term Forecast Error (Corrected)")
# plt.xlabel("Time (s)")
# plt.ylabel("Squared Error")
# plt.legend()
# plt.grid(True, alpha=0.3)

# plt.subplot(1, 2, 2)
# plt.hist(vel_gt, bins=50, density=True, color='k', alpha=0.2, label='GT')
# plt.hist(vel_base, bins=50, density=True, color='b', histtype='step', label='Base')
# plt.hist(vel_ours, bins=50, density=True, color='r', histtype='step', linewidth=2, label='Ours')
# plt.title("Velocity Distribution")
# plt.legend()

# plt.tight_layout()
# plt.savefig("final_comparison.png")
# plt.show()
