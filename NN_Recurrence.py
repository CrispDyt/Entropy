# ============================================================
# FINAL ROBUST TRAINING SCRIPT
# Config: 10k Steps + Mini-Batching(8) + DualOpt + BestSave
# ============================================================

import numpy as np
import random
import pickle
import torch
import torch.nn as nn
from torch import optim
from geomloss import SamplesLoss
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler

seed = 13531
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

def lorenz(xyz):
    s, r, b = 10.0, 28.0, 2.667
    x, y, z = xyz
    return np.array([s * (y - x), r * x - y - x * z, x * y - b * z], dtype=np.float64)

dt = 0.01
num_steps_data = 5000 

y = np.zeros((num_steps_data + 1, 3), dtype=np.float64)
y[0] = np.array([-5.065457, -7.56735, 19.060379], dtype=np.float64)
for i in range(num_steps_data):
    y[i + 1] = y[i] + lorenz(y[i]) * dt

transformer = MaxAbsScaler().fit(y)
scale_vec = transformer.max_abs_.copy()
y_scaled = transformer.transform(y) 

x_mean_all = y_scaled.mean(axis=0)
x_std_all  = y_scaled.std(axis=0) + 1e-6
x_mean_all_t = torch.tensor(x_mean_all, dtype=torch.float32, device=device)
x_std_all_t  = torch.tensor(x_std_all,  dtype=torch.float32, device=device)

def whiten_x(x: torch.Tensor):
    return (x - x_mean_all_t[None, :]) / x_std_all_t[None, :]

def delay_vectors_at_indices(x_series_1d: np.ndarray, idx: np.ndarray, tau: int, dim: int):
    out = np.zeros((len(idx), dim), dtype=np.float64)
    for a, t in enumerate(idx):
        for i in range(dim):
            out[a, i] = x_series_1d[t - i * tau]
    return out

def compute_section_hits_scaled_numpy(y_scaled: np.ndarray, z0_s: float, delta_z_s: float, dt: float, L_sec: int):
    z = y_scaled[:, 2]
    dz = (z[1:] - z[:-1]) / dt
    hits = []
    last = -10**12
    for t in range(len(z) - 1):
        if abs(z[t] - z0_s) <= delta_z_s and dz[t] > 0:
            if t - last >= L_sec:
                hits.append(t)
                last = t
    return np.array(hits, dtype=np.int64)

z0_orig = 27.0
delta_z_orig = 2.0
z0_s = float(z0_orig / scale_vec[2])
delta_z_s = float(delta_z_orig / scale_vec[2])
L_sec = 10 

USE_DELAY = True
tau_delay = 10
dim_delay = 5
min_t = (dim_delay - 1) * tau_delay if USE_DELAY else 0

hits_data = compute_section_hits_scaled_numpy(y_scaled, z0_s=z0_s, delta_z_s=delta_z_s, dt=dt, L_sec=L_sec)
hits_data = hits_data[hits_data >= min_t]
hits_data = np.sort(hits_data)

print(f"DATA: usable section hits N_U={len(hits_data)}")
if len(hits_data) < 50:
    raise RuntimeError("Too few section hits.")

XU_data = y_scaled[hits_data] 
if USE_DELAY:
    x_series_data = y_scaled[:, 0]
    DU_data = delay_vectors_at_indices(x_series_data, hits_data, tau=tau_delay, dim=dim_delay) 
else:
    DU_data = np.zeros((len(hits_data), 0), dtype=np.float64)

U_hits_data = np.concatenate([XU_data, DU_data], axis=1) 
d_u = U_hits_data.shape[1]

u_mean = U_hits_data.mean(axis=0)
u_std  = U_hits_data.std(axis=0) + 1e-6
u_mean_t = torch.tensor(u_mean, dtype=torch.float32, device=device)
u_std_t  = torch.tensor(u_std,  dtype=torch.float32, device=device)

def whiten_u(u: torch.Tensor):
    return (u - u_mean_t[None, :]) / u_std_t[None, :]

U_hits_data_t  = torch.tensor(U_hits_data, dtype=torch.float32, device=device)
U_hits_data_dw = whiten_u(U_hits_data_t)


def make_sinkhorn(blur):
    return SamplesLoss(loss="sinkhorn", p=2, blur=float(blur))

alpha_pair_state = 1.0
alpha_pair_delay = 0.7
w_marg_u0 = 0.20
w_marg_u1 = 0.20

net = nn.Sequential(
    nn.Linear(3, 100), nn.Tanh(),
    nn.Linear(100, 100), nn.Tanh(),
    nn.Linear(100, 100), nn.Tanh(),
    nn.Linear(100, 3),
).to(device)

optimizer1 = optim.Adam(net.parameters(), lr=1e-3) 
optimizer2 = optim.Adam(net.parameters(), lr=1e-5) 

def rollout_steps(net: nn.Module, x0: torch.Tensor, dt: float, steps: int):
    x = x0
    for _ in range(int(steps)):
        x = x + dt * net(x)
    return x

def stage1_teacher_forced_fulltraj_pairs(batch_size: int, tau_pair: int):
    max_i = len(y_scaled) - tau_pair - 1
    idx = np.random.randint(0, max_i, size=batch_size)
    x0 = torch.tensor(y_scaled[idx], dtype=torch.float32, device=device) 
    x_tau_data = torch.tensor(y_scaled[idx + tau_pair], dtype=torch.float32, device=device)
    x_tau_model = rollout_steps(net, x0, dt=dt, steps=tau_pair)
    return x0, x_tau_data, x_tau_model

@torch.no_grad()
def stage1_diag_loss(tau_pair: int, batch_size: int, blur: float):
    loss_sink = make_sinkhorn(blur)
    x0, x_tau_data, x_tau_model = stage1_teacher_forced_fulltraj_pairs(batch_size=batch_size, tau_pair=tau_pair)
    x0_dw = whiten_x(x0)
    x1_dw = whiten_x(x_tau_data)
    x1_mw = whiten_x(x_tau_model)
    pair_d = torch.cat([x0_dw, x1_dw], dim=1)
    pair_m = torch.cat([x0_dw, x1_mw], dim=1)
    return float(loss_sink(pair_m, pair_d).item())

# Stage-2 multi-shooting induced block
def build_multishoot_induced_block_fullbatch(k0: int, L_hits: int, defect_detach=True):
    assert 0 <= k0 and (k0 + L_hits) <= len(hits_data)
    hit_times = hits_data[k0 : k0 + L_hits].astype(np.int64)
    
    U_blk_d_dw = U_hits_data_dw[k0 : k0 + L_hits] 
    X_blk_d = torch.tensor(XU_data[k0 : k0 + L_hits], dtype=torch.float32, device=device)
    D_blk = torch.tensor(DU_data[k0 : k0 + L_hits], dtype=torch.float32, device=device) if USE_DELAY else \
            torch.zeros((L_hits, 0), dtype=torch.float32, device=device)

    x_hits_m = []
    x_curr = X_blk_d[0] 
    x_hits_m.append(x_curr)

    for j in range(L_hits - 1):
        dt_steps = int(hit_times[j + 1] - hit_times[j])
        if dt_steps < 1: dt_steps = 1
        x_next = rollout_steps(net, x_curr, dt=dt, steps=dt_steps) 
        x_hits_m.append(x_next)
        if defect_detach:
            x_curr = x_next.detach()
        else:
            x_curr = x_next

    X_blk_m = torch.stack(x_hits_m, dim=0) 
    U_blk_m = torch.cat([X_blk_m, D_blk], dim=1) 
    U_blk_m_dw = whiten_u(U_blk_m) 

    def split_weight(Udw):
        if USE_DELAY:
            xw = Udw[:, :3]
            dw = Udw[:, 3:]
            u = torch.cat([alpha_pair_state * xw, alpha_pair_delay * dw], dim=1)
        else:
            u = alpha_pair_state * Udw
        return u

    u_d = split_weight(U_blk_d_dw) 
    u_m = split_weight(U_blk_m_dw)

    u0_d, u1_d = u_d[:-1], u_d[1:]
    u0_m, u1_m = u_m[:-1], u_m[1:]

    pair_d = torch.cat([u0_d, u1_d], dim=1) 
    pair_m = torch.cat([u0_m, u1_m], dim=1)

    state_m = U_blk_m_dw[:, :3]
    state_d = U_blk_d_dw[:, :3]
    defect_loss = torch.mean((state_m - state_d) ** 2)

    return pair_m, pair_d, u0_m, u0_d, u1_m, u1_d, defect_loss

Nsteps = 10000    
warm_steps = 3000 

# pushforward steps for stage 1
tau_pair = 20 
stage1_batch = 512 

# Stage-2 Settings
L_hits_fullbatch = 16  # rollout hits on the induced system
BATCH_SIZE_STAGE2 = 8  # mini-btach size for stage-2

# Hyperparameters for dual-stage training
stage2_steps_per_cycle = 1 
stage1_steps_per_cycle = 5 

blur_start = 0.20 
blur_end   = 0.07 

w_defect = 10.0 
target2 = 0.05 

w2_min = 1e-5
w2_max = 1.0
w2_ramp_steps = 200 
ema_beta = 0.98
ema_L2 = None

print_every = 50

# Best Model Tracking
best_L2_val = float('inf')
best_model_state_dict = None
best_iter = -1

# ============================================================
# Training loop
# ============================================================
net.train()

loss_total_hist = []
loss_stage1_hist = []
loss_stage2_raw_hist = []
w2_hist = []
stage_used_hist = []

print("============================================================")
print(f"Start ROBUST training, Batch({BATCH_SIZE_STAGE2})")
print(f"Nsteps={Nsteps}, L_hits={L_hits_fullbatch}")
print("============================================================")

stage2_update_count = 0

for step in range(Nsteps):
    t = step / max(1, Nsteps - 1)
    blur = blur_start + (blur_end - blur_start) * t
    loss_sink = make_sinkhorn(blur)

    if step < warm_steps:
        stage_used = "stage1"
    else:
        cyc = stage2_steps_per_cycle + stage1_steps_per_cycle
        k = (step - warm_steps) % cyc
        stage_used = "stage2" if k < stage2_steps_per_cycle else "stage1"

    if stage_used == "stage1":
        curr_opt = optimizer1
    else:
        curr_opt = optimizer2
        
    curr_opt.zero_grad()

    if stage_used == "stage1":
        x0, x_tau_data, x_tau_model = stage1_teacher_forced_fulltraj_pairs(
            batch_size=stage1_batch, tau_pair=tau_pair
        )
        x0_dw = whiten_x(x0)
        x1_dw = whiten_x(x_tau_data)
        x1_mw = whiten_x(x_tau_model)

        pair_d = torch.cat([x0_dw, x1_dw], dim=1)
        pair_m = torch.cat([x0_dw, x1_mw], dim=1)

        L1 = loss_sink(pair_m, pair_d) 
        L_total = L1

        # placeholder
        L2_raw_avg = torch.tensor(0.0, device=device) 
        w2_eff = 0.0

    else:
        # Stage 2 training
        stage2_update_count += 1
        
        l2_losses = []
        
        for _ in range(BATCH_SIZE_STAGE2):
            # Random k0
            L_hits = int(min(len(hits_data), L_hits_fullbatch))
            k0_max = len(hits_data) - L_hits
            k0 = int(np.random.randint(0, max(1, k0_max + 1)))

            # Rollout induced block
            pair_m, pair_d, u0_m, u0_d, u1_m, u1_d, L_def = build_multishoot_induced_block_fullbatch(
                k0=k0, L_hits=L_hits, defect_detach=True
            )

            # Loss
            L2_graph = loss_sink(pair_m, pair_d)
            L2_marg0 = loss_sink(u0_m, u0_d)
            L2_marg1 = loss_sink(u1_m, u1_d)
            L2_marg  = w_marg_u0 * L2_marg0 + w_marg_u1 * L2_marg1

            # Normalize
            single_l2 = (L2_graph + L2_marg + w_defect * L_def) / float(L_hits)
            l2_losses.append(single_l2)
        
        L2_raw_avg = torch.stack(l2_losses).mean()

        # Auto-scaling 
        with torch.no_grad():
            val = float(L2_raw_avg.detach().item())
            if ema_L2 is None:
                ema_L2 = val
            else:
                ema_L2 = ema_beta * ema_L2 + (1.0 - ema_beta) * val

            w2_auto = target2 / max(ema_L2, 1e-12)
            w2_auto = float(np.clip(w2_auto, w2_min, w2_max))
            ramp = min(1.0, stage2_update_count / max(1, w2_ramp_steps))
            w2_eff = float(ramp * w2_auto)

        L_total = (w2_eff * L2_raw_avg)

    L_total.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    curr_opt.step()
    
    if stage_used == "stage2":
        current_l2_val = float(L2_raw_avg.detach().item())
        if current_l2_val < best_L2_val:
            best_L2_val = current_l2_val
            best_iter = step
            best_model_state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            print(f"  --> [New Best] Iter {step} | L2_raw_avg={best_L2_val:.4f}")

    loss_total_hist.append(float(L_total.detach().item()))
    w2_hist.append(float(w2_eff))
    loss_stage2_raw_hist.append(float(L2_raw_avg.detach().item()) if torch.is_tensor(L2_raw_avg) else float(L2_raw_avg))
    stage_used_hist.append(stage_used)

    if step % print_every == 0:
        s1_diag = stage1_diag_loss(tau_pair=tau_pair, batch_size=stage1_batch, blur=blur)
        loss_stage1_hist.append(s1_diag)

        if stage_used == "stage1":
            print(
                f"Iter {step:5d} [{stage_used}] | L_tot={loss_total_hist[-1]:.6f} | "
                f"Stage1_diag={s1_diag:.6f} | blur={blur:.3f}"
            )
        else:
            print(
                f"Iter {step:5d} [{stage_used}] | L_tot={loss_total_hist[-1]:.6f} | "
                f"L2_raw(avg)={loss_stage2_raw_hist[-1]:.4f} | w2={w2_eff:.6f} | "
                f"Stage1_diag={s1_diag:.6f} | blur={blur:.3f}"
            )

print("Training complete.")

# ============================================================
# 10) Save
# ============================================================
# Save Final
save_dict_final = {
    "config": {
        "seed": seed, "dt": dt, "Nsteps": Nsteps, "target2": target2,
        "batch_size_stage2": BATCH_SIZE_STAGE2
    },
    "model": {
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "transformer": transformer,
    },
    "train_hist": {
        "loss_total": np.array(loss_total_hist),
        "loss_diag": np.array(loss_stage1_hist),
        "w2": np.array(w2_hist),
    }
}
with open("DualOpt_MultiShoot_Auto.p", "wb") as f:
    pickle.dump(save_dict_final, f)
print("Saved FINAL results.")

# Save Best
if best_model_state_dict is not None:
    save_dict_best = {
        "config": {
            "seed": seed, "dt": dt, "Nsteps": Nsteps, "target2": target2,
            "best_iter": best_iter, "best_L2": best_L2_val
        },
        "model": {
            "state_dict": best_model_state_dict, 
            "transformer": transformer,
        },
        "train_hist": {
            "loss_total": np.array(loss_total_hist),
            "loss_diag": np.array(loss_stage1_hist),
        }
    }
    with open("DualOpt_Best.p", "wb") as f:
        pickle.dump(save_dict_best, f)
    print(f"Saved BEST results (L2={best_L2_val:.4f} at iter {best_iter}).")

# ============================================================
# 11) Plots
# ============================================================
plt.figure()
plt.plot(loss_total_hist, label="L_total")
plt.legend()
plt.title("Total Loss (10k Steps)")
plt.show()

# Visualize BEST
if best_model_state_dict is not None:
    print("Visualizing BEST model...")
    net.load_state_dict(best_model_state_dict) 
    net.eval()
    x = torch.tensor(np.array([0.1, 0.1, 0.5], dtype=np.float32), device=device)
    xs_best = []
    with torch.no_grad():
        for _ in range(num_steps_data):
            xs_best.append(x.cpu().numpy())
            x = x + dt * net(x)
    xs_best = np.array(xs_best)
    xs_best_orig = transformer.inverse_transform(xs_best)

    plt.figure()
    plt.scatter(xs_best_orig[:, 0], xs_best_orig[:, 2], s=0.2, c='r')
    plt.title(f"Model Trajectory (BEST Model, Iter {best_iter})")
    plt.show()