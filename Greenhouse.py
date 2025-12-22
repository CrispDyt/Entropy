import matplotlib.pyplot as plt
import numpy as np
from geomloss import SamplesLoss
import torch.nn as nn
import torch
from torch import optim
from sklearn.preprocessing import MaxAbsScaler
import pickle
import random

############################# lorenz system
def lorenz(xyz):
    s, r, b = 10, 28, 2.667
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

################################ experiment parameters
dt = 0.01
num_steps = int(5e3)
y = np.zeros((num_steps + 1, 3))
y[0] = np.array([-5.065457, -7.56735 , 19.060379])
tau = 10
method = 'delay_IM'   # 'delay_IM' or 'IM'
num_samples = 500
Nsteps = 10000
plot_every = 200
dim = 5

# Sinkhorn params (for DIM loss)
sinkhorn_blur = 0.05   # 0.05~0.2 都常见；更大更平滑更稳定
sinkhorn_p = 2

# simulate trajectory (Euler)
for i in range(num_steps):
    y[i + 1] = y[i] + lorenz(y[i]) * dt

# time-delay map
def delay(X):
    new = np.zeros((len(X)-tau*dim, dim))
    for i in range(dim):
        # print(dim*tau-(i+1)*tau)  # 如果你不想刷屏可以注释掉
        new[:, i] = X[dim*tau-(i+1)*tau:-(1+i)*tau]
    return new

y_delay_full = delay(y[:, 0])

# rescale for NN learning
transformer = MaxAbsScaler().fit(y)
y = transformer.transform(y)

# delayed trajectory (scaled)
y_delay = delay(y[:, 0])

# align lengths
y = y[:-(dim-1)*tau]
Ty_true = y[tau:]
y = y[:-tau]

batch_ixs = list(range(0, len(y)))

############################## Build network
torch.manual_seed(1252)
np.random.seed(13531)

net = nn.Sequential(
    nn.Linear(3, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, 3),
)

############################## Define Loss
loss_energy = SamplesLoss(loss="energy")  # for Ty
loss_sinkhorn_dim = SamplesLoss(loss="sinkhorn", p=2, blur=0.1)  # for DIM

optimizer = optim.Adam(net.parameters(), lr=1e-3)

############################## Training loop
net.train()
losses = []

for step in range(Nsteps):
    ixs = random.sample(batch_ixs, num_samples)

    y_batch = torch.tensor(y[ixs], dtype=torch.float32)          # [B,3]
    Ty_batch = torch.tensor(Ty_true[ixs], dtype=torch.float32)   # [B,3]
    y_delay_batch = torch.tensor(y_delay[ixs], dtype=torch.float32)  # [B,dim]

    optimizer.zero_grad()

    sols = []
    yy = y_batch
    for _ in range(tau * dim):
        sols.append(yy)
        yy = net(yy) * dt + yy

    sol = torch.stack(sols, dim=0)
    Ty = sol[tau, :, :]                      # [B,3]
    DIM = torch.flip(sol[::tau, :, 0].T, [1])  # [B,dim]

    if method == 'delay_IM':
        L_Ty = loss_energy(Ty, Ty_batch)
        L_DIM = loss_energy(DIM, y_delay_batch)  
        L = L_Ty + L_DIM
    elif method == 'IM':
        L_Ty = loss_energy(Ty, Ty_batch)
        L_DIM = torch.tensor(0.0)
        L = L_Ty
    else:
        raise ValueError("method must be 'delay_IM' or 'IM'")

    L.backward()
    optimizer.step()

    # 记录 float（方便后处理）
    losses.append(float(L.item()))

    # 每 50 轮输出一次
    if step % 1000 == 0:
        if method == 'delay_IM':
            print(f"iter {step:5d} | L={L.item():.6f} | L_Ty(energy)={L_Ty.item():.6f} | L_DIM(energy)={L_DIM.item():.6f}")
        else:
            print(f"iter {step:5d} | L={L.item():.6f} | L_Ty(energy)={L_Ty.item():.6f}")

############################## Long simulation
net.eval()
x = torch.tensor(np.array([0.1, 0.1, 0.5]), dtype=torch.float32)
xs = []
with torch.no_grad():
    for _ in range(num_steps):
        xs.append(x.numpy())
        x = net(x) * dt + x

xs = transformer.inverse_transform(np.array(xs))
plt.scatter(xs[:, 0], xs[:, 2], s=0.1)
plt.show()

plt.plot(xs[:, 0][:5000][5:], xs[:, 2][:5000][5:], linewidth=0.5, linestyle='--', marker='o', markersize=3)

############################## compute pushforwards
sols = []
yy = torch.tensor(y, dtype=torch.float32)
with torch.no_grad():
    for _ in range(tau * dim):
        sols.append(transformer.inverse_transform(yy.numpy()))
        yy = net(yy) * dt + yy

sol = np.array(sols)
Ty = sol[tau, :, :]
DIM = np.flip(sol[::tau, :, 0].T, axis=1)

plt.scatter(Ty[:, 0], Ty[:, 1], s=0.01)
plt.show()

# back to original coords for saving
y_orig = transformer.inverse_transform(y)
Ty_true_orig = transformer.inverse_transform(Ty_true)

if method == 'delay_IM':
    with open("DIM_recon_short_change.p", "wb") as f:
        pickle.dump([y_orig, Ty_true_orig, y_delay_full, y_orig[ixs], xs, Ty, DIM, losses], f)
if method == 'IM':
    with open("IM_recon_short.p", "wb") as f:
        pickle.dump([y_orig, Ty_true_orig, y_delay_full, y_orig[ixs], xs, Ty, DIM, losses], f)
