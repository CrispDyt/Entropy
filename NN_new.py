import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
from geomloss import SamplesLoss
import matplotlib.pyplot as plt
import pickle
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

def delay_1d_forward_reversed(X, tau, dim):
    N = len(X)
    out = np.zeros((N - tau * dim, dim), dtype=np.float64)
    for i in range(dim):
        out[:, i] = X[dim * tau - (i + 1) * tau : -(1 + i) * tau]
    return out

@torch.no_grad()
def precompute_u_data_single_step_scaled(
    y_scaled: np.ndarray,
    dt: float,
    scale_vec: np.ndarray,
    beta: float = 5.0,
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

def u_theta_single_step_jvp(
    net: nn.Module,
    x: torch.Tensor,
    dt: float,
    beta: float = 5.0,
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

    DF = torch.stack(cols, dim=2)  # [B,3,3]
    svals = torch.linalg.svdvals(DF)
    lam = torch.log(svals + eps) / dt
    u = torch.nn.functional.softplus(lam, beta=beta).sum(dim=1, keepdim=True)
    return u

dt = 0.01
num_steps = 5000
tau = 10
dim = 5
num_samples = 512
Nsteps = 12000  

stageA_frac = 0.35 
stageB_frac = 0.40  

blur_start = 0.20
blur_end   = 0.07

gamma_u_final = 0.03
u_softclip = 500.0   

alpha_x = 2.0 
alpha_T = 4.0  
alpha_D = 1.0   
alpha_u = 1.0  

w_geom_T = 1.0     
w_geom_D = 1.0     
w_joint_max = 1.0   

include_x_in_joint = False  

print("CONFIG:")
print(f"  Nsteps={Nsteps}, stageA={stageA_frac}, stageB={stageB_frac}")
print(f"  blur: {blur_start} -> {blur_end}")
print(f"  alpha_x={alpha_x}, alpha_T={alpha_T}, alpha_D={alpha_D}, gamma_final={gamma_u_final}")
print(f"  include_x_in_joint={include_x_in_joint}")


y = np.zeros((num_steps + 1, 3), dtype=np.float64)
y[0] = np.array([-5.065457, -7.56735, 19.060379], dtype=np.float64)
for i in range(num_steps):
    y[i + 1] = y[i] + lorenz(y[i]) * dt

transformer = MaxAbsScaler().fit(y)
scale_vec = transformer.max_abs_
y_scaled_all = transformer.transform(y)

y_delay_all = delay_1d_forward_reversed(y_scaled_all[:, 0], tau=tau, dim=dim)

y_scaled_trim = y_scaled_all[: -(dim - 1) * tau]
Ty_true = y_scaled_trim[tau:]     
y_tr = y_scaled_trim[:-tau]        
y_delay_tr = y_delay_all          

assert len(y_tr) == len(Ty_true) == len(y_delay_tr)
M = len(y_tr)

u_data_tr = precompute_u_data_single_step_scaled(
    y_scaled=y_tr, dt=dt, scale_vec=scale_vec, beta=5.0, device=device
)
u_mean = float(u_data_tr.mean())
u_std = float(u_data_tr.std() + 1e-6)

x_mean = y_tr.mean(axis=0)
x_std  = y_tr.std(axis=0) + 1e-6
Ty_mean = Ty_true.mean(axis=0)
Ty_std  = Ty_true.std(axis=0) + 1e-6
D_mean = y_delay_tr.mean(axis=0)
D_std  = y_delay_tr.std(axis=0) + 1e-6

x_mean_t  = torch.tensor(x_mean, dtype=torch.float32, device=device)
x_std_t   = torch.tensor(x_std,  dtype=torch.float32, device=device)
Ty_mean_t = torch.tensor(Ty_mean, dtype=torch.float32, device=device)
Ty_std_t  = torch.tensor(Ty_std,  dtype=torch.float32, device=device)
D_mean_t  = torch.tensor(D_mean, dtype=torch.float32, device=device)
D_std_t   = torch.tensor(D_std,  dtype=torch.float32, device=device)


def whiten(a, mean_t, std_t):
    return (a - mean_t) / std_t

def softclip_tanh(z, c):
    return c * torch.tanh(z / c)

net = nn.Sequential(
    nn.Linear(3, 100), nn.Tanh(),
    nn.Linear(100, 100), nn.Tanh(),
    nn.Linear(100, 100), nn.Tanh(),
    nn.Linear(100, 3),
).to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-3)
batch_ixs = np.arange(M)

def make_sinkhorn(blur):
    return SamplesLoss(loss="sinkhorn", p=2, blur=float(blur))

net.train()
losses = []

print("Start Training (improved curriculum + whitening + weights)...")

for step in range(Nsteps):
    ixs = np.random.choice(batch_ixs, size=num_samples, replace=False)

    x0 = torch.tensor(y_tr[ixs], dtype=torch.float32, device=device)                   # [B,3]
    Ty_batch = torch.tensor(Ty_true[ixs], dtype=torch.float32, device=device)          # [B,3]
    y_delay_batch = torch.tensor(y_delay_tr[ixs], dtype=torch.float32, device=device)  # [B,dim]
    u_data_batch = torch.tensor(u_data_tr[ixs], dtype=torch.float32, device=device)    # [B,1]

    optimizer.zero_grad()

    sols = []
    yy = x0
    for _ in range(tau * dim):
        sols.append(yy)
        yy = yy + dt * net(yy)

    sol = torch.stack(sols, dim=0)               # [tau*dim, B, 3]
    Ty = sol[tau, :, :]                          # [B,3]
    DIM = torch.flip(sol[::tau, :, 0].T, [1])    # [B,dim]

    t = step / max(1, Nsteps - 1)
    blur = blur_start + (blur_end - blur_start) * t
    loss_sink = make_sinkhorn(blur)

    sA = int(stageA_frac * Nsteps)
    sB = int((stageA_frac + stageB_frac) * Nsteps)

    if step <= sA:
        w_joint = 0.0
    else:
        prog = (step - sA) / max(1, (Nsteps - sA))
        w_joint = w_joint_max * min(1.0, prog)

    if step <= sB:
        gamma = 0.0
    else:
        prog = (step - sB) / max(1, (Nsteps - sB))
        gamma = gamma_u_final * min(1.0, prog)

    x0w  = whiten(x0, x_mean_t, x_std_t)
    Tyw  = whiten(Ty, Ty_mean_t, Ty_std_t)
    Tydw = whiten(Ty_batch, Ty_mean_t, Ty_std_t)
    DIMw = whiten(DIM, D_mean_t, D_std_t)
    DIMdw = whiten(y_delay_batch, D_mean_t, D_std_t)

    L_geom_T = loss_sink(alpha_T * Tyw, alpha_T * Tydw)
    L_geom_D = loss_sink(alpha_D * DIMw, alpha_D * DIMdw)
    L_geom = w_geom_T * L_geom_T + w_geom_D * L_geom_D

    if w_joint > 0:
        # entropy on model samples x0
        u_theta = u_theta_single_step_jvp(net, x0, dt=dt, beta=5.0)

        u_data_z  = (u_data_batch - u_mean) / u_std
        u_theta_z = (u_theta     - u_mean) / u_std

        u_data_z  = softclip_tanh(u_data_z,  u_softclip)
        u_theta_z = softclip_tanh(u_theta_z, u_softclip)

        if include_x_in_joint:
            feat_model = torch.cat([
                alpha_x * x0w,
                alpha_T * Tyw,
                alpha_D * DIMw,
                (gamma * alpha_u) * u_theta_z
            ], dim=1)
            feat_data = torch.cat([
                alpha_x * x0w,          
                alpha_T * Tydw,
                alpha_D * DIMdw,
                (gamma * alpha_u) * u_data_z
            ], dim=1)
        else:
            feat_model = torch.cat([
                alpha_T * Tyw,
                alpha_D * DIMw,
                (gamma * alpha_u) * u_theta_z
            ], dim=1)
            feat_data = torch.cat([
                alpha_T * Tydw,
                alpha_D * DIMdw,
                (gamma * alpha_u) * u_data_z
            ], dim=1)

        L_joint = loss_sink(feat_model, feat_data)
    else:
        L_joint = torch.tensor(0.0, device=device)

    L_total = L_geom + w_joint * L_joint

    L_total.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()

    losses.append(float(L_total.item()))

    if step % 50 == 0:
        print(
            f"Iter {step:5d} | L_tot={L_total.item():.6f} | "
            f"L_geom={L_geom.item():.6f} (T={L_geom_T.item():.4f},D={L_geom_D.item():.4f}) | "
            f"L_joint={L_joint.item():.6f} | w_joint={w_joint:.3f} | gamma={gamma:.4f} | blur={blur:.3f}"
        )

print("Training complete. Generating plots and saving...")

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
plt.title("Model Trajectory (improved)")
plt.show()

plt.figure()
plt.plot(losses)
plt.title("Training Loss (improved)")
plt.show()
y_orig = transformer.inverse_transform(y_tr)
Ty_true_orig = transformer.inverse_transform(Ty_true)
y_delay_full_orig = delay_1d_forward_reversed(y[:, 0], tau=tau, dim=dim)

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
    y_orig[ixs],
    xs_orig,
    Ty_model,
    DIM_model,
    losses,
    u_data_tr,
]
outfile = "Joint_Lifted_recon_new.p"
with open(outfile, "wb") as f:
    pickle.dump(save_data, f)

print(f"Saved results to: {outfile}")
