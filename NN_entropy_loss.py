# ============================================================
# Fixed A-style Joint Lifted OT training for Lorenz
# (Greenhouse pushforward + delay + entropy lift in ONE OT)
#
# Key fix:
#   Previously: OT([DIM, gamma*u]) + MSE(Ty)
#   Now:        OT([Ty, DIM, gamma*u])     (optionally include x too)
#
# This prevents degeneracy where DIM+u matches but 3D state collapses.
# ============================================================

import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
from geomloss import SamplesLoss
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MaxAbsScaler

# =============================
# 0) Repro & device
# =============================
seed = 13531
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# =============================
# 1) Lorenz system (original coords)
# =============================
def lorenz(xyz):
    s, r, b = 10.0, 28.0, 2.667
    x, y, z = xyz
    return np.array([s * (y - x), r * x - y - x * z, x * y - b * z], dtype=np.float64)


def jac_lorenz_field_batch(xyz_orig: torch.Tensor):
    s, r, b = 10.0, 28.0, 2.667
    x = xyz_orig[:, 0]
    y = xyz_orig[:, 1]
    z = xyz_orig[:, 2]
    B = xyz_orig.shape[0]

    J = torch.zeros(B, 3, 3, device=xyz_orig.device, dtype=xyz_orig.dtype)
    J[:, 0, 0] = -s
    J[:, 0, 1] = s

    J[:, 1, 0] = (r - z)
    J[:, 1, 1] = -1.0
    J[:, 1, 2] = -x

    J[:, 2, 0] = y
    J[:, 2, 1] = x
    J[:, 2, 2] = -b
    return J


# =============================
# 2) Delay embedding (FORWARD window, reversed order)
# =============================
def delay_1d_forward_reversed(X, tau, dim):
    N = len(X)
    out = np.zeros((N - tau * dim, dim), dtype=np.float64)
    for i in range(dim):
        out[:, i] = X[dim * tau - (i + 1) * tau : -(1 + i) * tau]
    return out


# =============================
# 3) u_data(x): single-step observable
# =============================
@torch.no_grad()
def precompute_u_data_single_step_scaled(
    y_scaled: np.ndarray,
    dt: float,
    scale_vec: np.ndarray,
    beta: float = 5.0,   # <-- softer than 10
    device: str = "cpu",
):
    y_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)
    scales = torch.tensor(scale_vec, dtype=torch.float32, device=device)
    ratio = (scales[None, :] / scales[:, None])

    N = y_t.shape[0]
    I = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(N, 3, 3)

    x_o = y_t * scales[None, :]
    Jf_orig = jac_lorenz_field_batch(x_o)
    Jf_scaled = Jf_orig * ratio[None, :, :]

    DF = I + dt * Jf_scaled
    svals = torch.linalg.svdvals(DF)
    lam = torch.log(svals + 1e-8) / dt
    u = torch.nn.functional.softplus(lam, beta=beta).sum(dim=1, keepdim=True)
    return u.cpu().numpy()


# =============================
# 4) u_theta(x): single-step via JVP
# =============================
def u_theta_single_step_jvp(
    net: nn.Module,
    x: torch.Tensor,
    dt: float,
    beta: float = 5.0,   # <-- softer than 10
    eps: float = 1e-8,
):
    B, d = x.shape
    assert d == 3
    dtype, dev = x.dtype, x.device

    cols = []
    for j in range(3):
        v = torch.zeros(B, 3, device=dev, dtype=dtype)
        v[:, j] = 1.0
        _, jvp = torch.autograd.functional.jvp(lambda inp: net(inp), (x,), (v,), create_graph=True)
        cols.append(v + dt * jvp)

    DF = torch.stack(cols, dim=2)
    svals = torch.linalg.svdvals(DF)
    lam = torch.log(svals + eps) / dt
    u = torch.nn.functional.softplus(lam, beta=beta).sum(dim=1, keepdim=True)
    return u


# =============================
# 5) Experiment parameters
# =============================
dt = 0.01
num_steps = 5000
tau = 10
dim = 5
num_samples = 512
Nsteps = 10000

# ---------------------------------------------------------
# SELECT METHOD HERE:
# 'joint_lifted'      : FIXED joint OT on [Ty, DIM, gamma*u]
# 'measure_theoretical': Greenhouse baseline
# ---------------------------------------------------------
method = 'joint_lifted'
# method = 'measure_theoretical'

# Hyperparameters
# IMPORTANT FIXES:
gamma_u_final = 0.03   # <-- much smaller than 0.5 to avoid leaving correct basin
warmup_frac = 0.3      # <-- longer warmup
u_clip = 3.0           # <-- clip standardized entropy to stabilize

sinkhorn_blur = 0.1    # <-- slightly larger blur helps stability early (0.05 can be too sharp)
use_energy_for_state = False  # keep greenhouse as sinkhorn; optional

# Optional: include x itself in joint feature to lock geometry further (recommended)
include_x_in_joint = True

print(f"Running Method: {method}")
print(f"dt={dt}, tau={tau}, dim={dim}, batch={num_samples}, steps={Nsteps}")
print(f"gamma_u_final={gamma_u_final}, warmup_frac={warmup_frac}, blur={sinkhorn_blur}, include_x={include_x_in_joint}")


# =============================
# 6) Simulate Data
# =============================
y = np.zeros((num_steps + 1, 3), dtype=np.float64)
y[0] = np.array([-5.065457, -7.56735, 19.060379], dtype=np.float64)
for i in range(num_steps):
    y[i + 1] = y[i] + lorenz(y[i]) * dt


# =============================
# 7) Scale & Delay Embedding
# =============================
transformer = MaxAbsScaler().fit(y)
scale_vec = transformer.max_abs_
y_scaled_all = transformer.transform(y)

y_delay_all = delay_1d_forward_reversed(y_scaled_all[:, 0], tau=tau, dim=dim)

# Align lengths
y_scaled_trim = y_scaled_all[: -(dim - 1) * tau]
Ty_true = y_scaled_trim[tau:]      # x_{k+tau}  (scaled)
y_tr = y_scaled_trim[:-tau]        # x_k        (scaled)
y_delay_tr = y_delay_all           # delay vec at x_k

assert len(y_tr) == len(Ty_true) == len(y_delay_tr)
M = len(y_tr)


# =============================
# 8) Precompute u_data
# =============================
u_data_tr = precompute_u_data_single_step_scaled(
    y_scaled=y_tr, dt=dt, scale_vec=scale_vec, beta=5.0, device=device
)
u_mean = float(u_data_tr.mean())
u_std = float(u_data_tr.std() + 1e-6)


# =============================
# 9) Network & Loss
# =============================
net = nn.Sequential(
    nn.Linear(3, 100), nn.Tanh(),
    nn.Linear(100, 100), nn.Tanh(),
    nn.Linear(100, 100), nn.Tanh(),
    nn.Linear(100, 3),
).to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-3)

loss_sink = SamplesLoss(loss="sinkhorn", p=2, blur=sinkhorn_blur)
loss_energy = SamplesLoss(loss="energy")  # optional stable alternative

batch_ixs = np.arange(M)


# =============================
# 10) Training Loop
# =============================
net.train()
losses = []

print(f"Start Training ({method})...")

for step in range(Nsteps):
    ixs = np.random.choice(batch_ixs, size=num_samples, replace=False)

    x0 = torch.tensor(y_tr[ixs], dtype=torch.float32, device=device)            # x_k (scaled)
    y_delay_batch = torch.tensor(y_delay_tr[ixs], dtype=torch.float32, device=device)  # delay(x_k) (scaled)
    Ty_batch = torch.tensor(Ty_true[ixs], dtype=torch.float32, device=device)   # x_{k+tau} (scaled)
    u_data_batch = torch.tensor(u_data_tr[ixs], dtype=torch.float32, device=device)

    optimizer.zero_grad()

    # --- rollout from x0
    sols = []
    yy = x0
    for _ in range(tau * dim):
        sols.append(yy)
        yy = yy + dt * net(yy)

    sol = torch.stack(sols, dim=0)               # [tau*dim, B, 3]
    Ty = sol[tau, :, :]                          # [B,3] scaled
    DIM = torch.flip(sol[::tau, :, 0].T, [1])    # [B,dim] from model rollout (scaled x-component)

    if method == "joint_lifted":
        # ---- entropy on MODEL SAMPLES x0
        u_theta = u_theta_single_step_jvp(net, x0, dt=dt, beta=5.0)

        # standardize + clip
        u_data_z = (u_data_batch - u_mean) / u_std
        u_theta_z = (u_theta - u_mean) / u_std
        u_data_z = torch.clamp(u_data_z, -u_clip, u_clip)
        u_theta_z = torch.clamp(u_theta_z, -u_clip, u_clip)

        # gamma schedule
        warmup_steps = int(warmup_frac * Nsteps)
        if step <= warmup_steps:
            gamma = 0.0
        else:
            prog = (step - warmup_steps) / max(1, Nsteps - warmup_steps)
            gamma = gamma_u_final * min(1.0, prog)

        # ======= CRITICAL FIX =======
        # joint feature includes Ty (3D pushforward) to prevent collapse
        # Optionally also include x0 itself to lock base geometry.
        if include_x_in_joint:
            feat_model = torch.cat([x0, Ty, DIM, gamma * u_theta_z], dim=1)      # [B, 3+3+dim+1]
            feat_data  = torch.cat([x0, Ty_batch, y_delay_batch, gamma * u_data_z], dim=1)
        else:
            feat_model = torch.cat([Ty, DIM, gamma * u_theta_z], dim=1)          # [B, 3+dim+1]
            feat_data  = torch.cat([Ty_batch, y_delay_batch, gamma * u_data_z], dim=1)

        # One OT (no MSE anchor)
        L_total = loss_sink(feat_model, feat_data)

    elif method == "measure_theoretical":
        # Greenhouse baseline: match delay distribution + pushforward distribution
        L_delay_dist = loss_sink(DIM, y_delay_batch)

        # Use sinkhorn or energy here; sinkhorn matches your baseline
        if use_energy_for_state:
            L_state_dist = loss_energy(Ty, Ty_batch)
        else:
            L_state_dist = loss_sink(Ty, Ty_batch)

        L_total = L_delay_dist + L_state_dist
        gamma = 0.0

    else:
        raise ValueError("Unknown method")

    L_total.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()

    losses.append(float(L_total.item()))

    if step % 50 == 0:
        if method == "joint_lifted":
            print(f"Iter {step:5d} | L_tot={L_total.item():.6f} | gamma={gamma:.4f}")
        else:
            print(f"Iter {step:5d} | L_tot={L_total.item():.6f} | L_delay={L_delay_dist.item():.6f} | L_state={L_state_dist.item():.6f}")


# =============================
# 11) Evaluation & Saving
# =============================
print("Training complete. Generating plots and saving...")

# Long simulation
net.eval()
x = torch.tensor(np.array([0.1, 0.1, 0.5], dtype=np.float32), device=device)
xs = []
with torch.no_grad():
    for _ in range(num_steps):
        xs.append(x.cpu().numpy())
        x = x + dt * net(x)
xs = np.array(xs)
xs_orig = transformer.inverse_transform(xs)

plt.figure()
plt.scatter(xs_orig[:, 0], xs_orig[:, 2], s=0.2)
plt.title(f"Model Trajectory ({method})")
plt.show()

plt.figure()
plt.plot(losses)
plt.title(f"Training Loss ({method})")
plt.show()

# Prepare Data for Saving
y_orig = transformer.inverse_transform(y_tr)
Ty_true_orig = transformer.inverse_transform(Ty_true)
y_delay_full_orig = delay_1d_forward_reversed(y[:, 0], tau=tau, dim=dim)

# Pushforward validation samples
with torch.no_grad():
    yy = torch.tensor(y_tr[:2000], dtype=torch.float32, device=device)
    sols = []
    for _ in range(tau * dim):
        sols.append(transformer.inverse_transform(yy.cpu().numpy()))
        yy = yy + dt * net(yy)
sol_np = np.array(sols)
Ty_model = sol_np[tau, :, :]
DIM_model = np.flip(sol_np[::tau, :, 0].T, axis=1)

save_data = [
    y_orig,
    Ty_true_orig,
    y_delay_full_orig,
    y_orig[ixs],     # last batch (not important)
    xs_orig,
    Ty_model,
    DIM_model,
    losses,
    u_data_tr,
]

outfile = "Joint_Lifted_recon.p" if method == "joint_lifted" else "Measure_Theoretical_recon.p"
with open(outfile, "wb") as f:
    pickle.dump(save_data, f)

print(f"Saved results to: {outfile}")
