import numpy as np
import torch
import torch.nn as nn
from torch import optim
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from geomloss import SamplesLoss

# =============================
# Lorenz-96 system (ODE)
# =============================
def l96(x, F=8.0):
    # x: shape (D,)
    xp1 = np.roll(x, -1)
    xm1 = np.roll(x,  1)
    xm2 = np.roll(x,  2)
    return (xp1 - xm2) * xm1 - x + F

def delay_1d(X, tau, dim):
    X = np.asarray(X).reshape(-1)
    new = np.zeros((len(X) - tau*dim, dim))
    for i in range(dim):
        new[:, i] = X[dim*tau-(i+1)*tau:-(1+i)*tau]
    return new

# =============================
# Hyperparameters
# =============================
D = 20          # <-- L96 dimension: try 10 or 20 first
F = 8.0
dt = 0.01
num_steps = int(2e5)
burn_in = 2000  # discard transient

tau = 10
dim = 5

method = "IM"   # IM-only
num_samples = 512
Nsteps = 10000

device = "cpu"  # or "cuda"
torch.manual_seed(1252)
np.random.seed(13531)

# =============================
# Simulate L96 trajectory (Euler) in ORIGINAL coords
# =============================
y = np.zeros((num_steps + 1, D), dtype=np.float64)
y[0] = F * np.ones(D)
y[0][0] += 0.01  # small perturbation to break symmetry

for i in range(num_steps):
    y[i + 1] = y[i] + dt * l96(y[i], F=F)

# discard transient
y = y[burn_in:]  # shape (N, D)

# delay on x0 in ORIGINAL coords (for compatibility / later plotting)
y_delay_full = delay_1d(y[:, 0], tau=tau, dim=dim)

# =============================
# Scale for NN learning
# =============================
transformer = MaxAbsScaler().fit(y)
y_scaled = transformer.transform(y)  # [N, D]

# align lengths like your Lorenz script
y_scaled = y_scaled[:-(dim - 1) * tau]
Ty_true = y_scaled[tau:]     # shifted by tau
y_scaled = y_scaled[:-tau]   # align
N0 = len(y_scaled)

y_tr = y_scaled
Ty_true_tr = Ty_true
batch_ixs = list(range(N0))

# =============================
# Build network: R^D -> R^D
# =============================
net = nn.Sequential(
    nn.Linear(D, 256),
    nn.Tanh(),
    nn.Linear(256, 256),
    nn.Tanh(),
    nn.Linear(256, 256),
    nn.Tanh(),
    nn.Linear(256, D),
).to(device)

loss = SamplesLoss(loss="energy")
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# =============================
# Training loop (IM-only)
# =============================
net.train()
losses = []

for step in range(Nsteps):
    ixs = random.sample(batch_ixs, num_samples)

    y_batch  = torch.tensor(y_tr[ixs], dtype=torch.float32, device=device, requires_grad=True)
    Ty_batch = torch.tensor(Ty_true_tr[ixs], dtype=torch.float32, device=device)

    optimizer.zero_grad()

    # tau-step rollout
    yy = y_batch
    for _ in range(tau):
        yy = yy + dt * net(yy)
    Ty = yy

    L_state = loss(Ty, Ty_batch)
    L = L_state

    L.backward()
    optimizer.step()

    losses.append(float(L.detach().cpu().numpy()))
    if step % 50 == 0:
        print(f"iter {step:5d} | L={losses[-1]:.6f} | L_state={float(L_state):.6f}")

# =============================
# Long simulation from learned model (for your evaluation)
# =============================
net.eval()
x0 = y_tr[0]  # start from a data point in scaled space
x = torch.tensor(x0, dtype=torch.float32, device=device)
xs = []
with torch.no_grad():
    for _ in range(len(y_tr)):
        xs.append(x.detach().cpu().numpy())
        x = x + dt * net(x)

xs = np.array(xs)              # scaled
xs_orig = transformer.inverse_transform(xs)

# quick sanity plot: x0(t)
plt.plot(xs_orig[:5000, 0], linewidth=1)
plt.title("L96 IM_only: x0(t) (first 5000 steps)")
plt.show()

# =============================
# Compute tau-step pushforward samples for plotting/saving
# =============================
yy = torch.tensor(y_tr, dtype=torch.float32, device=device)
with torch.no_grad():
    for _ in range(tau):
        yy = yy + dt * net(yy)
Ty_model = transformer.inverse_transform(yy.detach().cpu().numpy())

DIM_model = None  # IM-only, keep compatibility

# =============================
# Save (same list format your parser expects)
# [y_orig, Ty_true_orig, y_delay_full, y_orig, xs_orig, Ty_model, DIM_model, losses]
# =============================
y_orig = transformer.inverse_transform(y_tr)
Ty_true_orig = transformer.inverse_transform(Ty_true_tr)

outfile = "L96_IM_only.p"
with open(outfile, "wb") as f:
    pickle.dump([y_orig, Ty_true_orig, y_delay_full, y_orig, xs_orig, Ty_model, DIM_model, losses], f)

print("Saved to:", outfile)
