# Used to check whether the learned model is degenerate/collapsed.
import pickle
import numpy as np

def cov_logdet(X, eps=1e-9):
    X = np.asarray(X)
    X = X - X.mean(axis=0, keepdims=True)
    C = (X.T @ X) / max(1, (len(X)-1))
    C = C + eps*np.eye(C.shape[0])
    sign, logdet = np.linalg.slogdet(C)
    return float(logdet), float(sign)

def basic_stats(name, X):
    X = np.asarray(X)
    print(f"\n== {name} ==")
    print("shape:", X.shape)
    print("min:", np.min(X, axis=0))
    print("max:", np.max(X, axis=0))
    print("mean:", np.mean(X, axis=0))
    print("std:", np.std(X, axis=0))
    print("nan/inf:", np.isnan(X).any(), np.isinf(X).any())

def w1_1d(a, b):
    a = np.sort(np.asarray(a).ravel())
    b = np.sort(np.asarray(b).ravel())
    n = min(len(a), len(b))
    if len(a) != n:
        idx = np.linspace(0, len(a)-1, n).astype(int)
        a = a[idx]
    if len(b) != n:
        idx = np.linspace(0, len(b)-1, n).astype(int)
        b = b[idx]
    return float(np.mean(np.abs(a - b)))

def ks_1d(a, b):
    a = np.sort(np.asarray(a).ravel())
    b = np.sort(np.asarray(b).ravel())
    n = min(len(a), len(b))
    if len(a) != n:
        idx = np.linspace(0, len(a)-1, n).astype(int)
        a = a[idx]
    if len(b) != n:
        idx = np.linspace(0, len(b)-1, n).astype(int)
        b = b[idx]
    grid = np.sort(np.concatenate([a, b]))
    Fa = np.searchsorted(a, grid, side="right") / len(a)
    Fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(Fa - Fb)))

def main(pfile):
    with open(pfile, "rb") as f:
        y_orig, Ty_true, y_delay_full, y_lastbatch, xs_orig, Ty_model, DIM_model, losses, u_data_tr = pickle.load(f)

    basic_stats("xs_orig (long traj)", xs_orig)
    basic_stats("Ty_true (GT pushforward)", Ty_true)
    basic_stats("Ty_model (model pushforward)", Ty_model)
    basic_stats("DIM_model (delay embedding from model)", DIM_model)

    logdet_gt, sign_gt = cov_logdet(Ty_true)
    logdet_md, sign_md = cov_logdet(Ty_model)
    print("\n== Covariance volume check (Ty) ==")
    print("logdet cov(Ty_true):", logdet_gt, "sign:", sign_gt)
    print("logdet cov(Ty_model):", logdet_md, "sign:", sign_md)
    var_ratio = (np.var(Ty_model, axis=0) + 1e-12) / (np.var(Ty_true, axis=0) + 1e-12)
    print("var_ratio model/gt per-dim:", var_ratio)

    print("\n== 1D distribution distances (Ty marginals) ==")
    for j, nm in enumerate(["x","y","z"]):
        w1 = w1_1d(Ty_model[:,j], Ty_true[:,j])
        ks = ks_1d(Ty_model[:,j], Ty_true[:,j])
        print(f"{nm}: W1_1d={w1:.4f} | KS={ks:.4f}")

    v = np.linalg.norm(xs_orig[1:] - xs_orig[:-1], axis=1)
    print("\n== Long-trajectory speed stats ==")
    print("speed mean/std:", float(v.mean()), float(v.std()))
    print("speed min/max:", float(v.min()), float(v.max()))

    losses = np.asarray(losses)
    print("\n== Training losses summary ==")
    print("loss min:", float(losses.min()), "final:", float(losses[-1]))
    tail = losses[int(0.9*len(losses)):]
    print("tail mean:", float(tail.mean()), "tail std:", float(tail.std()))

if __name__ == "__main__":
    main("Joint_Lifted_recon_new.p")
