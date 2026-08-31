# Duffing oscillator - hidden-state reconstruction from input/output histories
#
# Research question
# -----------------
# The measured output in the article is y = x2.  Can the hidden physical state
# x1 be reconstructed using only current/past output samples and past inputs?
#
# This file isolates that question from Koopman rollout error.
# It uses the true I/O history to build delay coordinates and then identifies a
# static decoder from those coordinates to x1.
#
# Two experiments are performed:
#   1) nD = 1: does adding an intercept b remove the apparent x1 offset?
#   2) nD in [1, 2, 5, 10, 20]: how does I/O memory affect x1 reconstruction?
#
# For a fair delay sweep, ALL nD values use the SAME training trajectories,
# the SAME target time, and the SAME test interval.  Thus the only deliberately
# varied quantity is the amount of past I/O information supplied to the decoder.

from pathlib import Path

import matplotlib.pyplot as plt
import torch


# =============================================================================
# 1. Parameters
# =============================================================================

torch.set_default_dtype(torch.float64)
torch.manual_seed(156)

dt = 0.01
delta = 2.0
N_trajectories = 20_000
Nrbf = 10
nD_values = [1, 2, 5, 10, 20]
max_nD = max(nD_values)

# Output used in the authors' Duffing example: y = [0 1] x = x2.
Cy = torch.tensor([0.0, 1.0])


# =============================================================================
# 2. True nonlinear plant
# =============================================================================


def dynamics(x, u):
    x1 = x[..., 0]
    x2 = x[..., 1]

    dx1 = x2
    dx2 = -delta * x2 - x1 * torch.cos(x1 + x2) + u

    return torch.stack([dx1, dx2], dim=-1)


def simulate_step(x, u):
    """One RK4 step with constant u over the sampling interval."""
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def output(x):
    return x @ Cy


# =============================================================================
# 3. Common training trajectories
# =============================================================================
#
# We generate enough history for the largest delay count.  Every decoder is
# trained at the same anchor time k=max_nD, so their target x1 distribution is
# identical.  This prevents trajectory evolution from confounding the nD sweep.


torch.manual_seed(1560)
x = -1.0 + 2.0 * torch.rand((N_trajectories, 2))

X_train_history = [x]
Y_train_history = [output(x)]
U_train_history = []

for _ in range(max_nD + 1):
    u = -1.0 + 2.0 * torch.rand(N_trajectories)
    U_train_history.append(u)

    x = simulate_step(x, u)
    X_train_history.append(x)
    Y_train_history.append(output(x))

anchor_k = max_nD
X1_train_target = X_train_history[anchor_k][:, 0]


# =============================================================================
# 4. Fixed validation trajectory
# =============================================================================

T_test = 10.0
N_test = int(T_test / dt)
t_test = torch.arange(N_test + 1) * dt

x0_test = torch.tensor([-0.6, 1.4])
U_test = 0.8 * torch.sin(torch.arange(N_test) * 0.2)

x = x0_test.clone()
X_true = [x.clone()]
Y_true = [output(x)]

for k in range(N_test):
    x = simulate_step(x, U_test[k])
    X_true.append(x.clone())
    Y_true.append(output(x))

X_true = torch.stack(X_true)
Y_true = torch.stack(Y_true)


# =============================================================================
# 5. Delay-coordinate helpers
# =============================================================================
#
# For ny = 1 and m = 1:
#
#   zeta_k = [y_k, y_(k-1), ..., y_(k-nD),
#             u_(k-1), ..., u_(k-nD)]
#
# so
#
#   n_zeta = (nD + 1)*ny + nD*m = 2*nD + 1.


def build_training_zeta(nD):
    y_terms = [Y_train_history[anchor_k - j] for j in range(nD + 1)]
    u_terms = [U_train_history[anchor_k - 1 - j] for j in range(nD)]

    return torch.stack(y_terms + u_terms, dim=1)


def build_test_zeta(k, nD):
    y_terms = [Y_true[k - j] for j in range(nD + 1)]
    u_terms = [U_test[k - 1 - j] for j in range(nD)]

    return torch.stack(y_terms + u_terms)


# =============================================================================
# 6. Least-squares decoder helpers
# =============================================================================


def fit_decoder(features, target, use_bias):
    """Fit x1 ~= w^T feature (+ b when use_bias=True)."""
    if use_bias:
        features = torch.cat(
            [features, torch.ones((features.shape[0], 1))],
            dim=1,
        )

    return torch.linalg.lstsq(features, target).solution


def apply_decoder(feature, weights, use_bias):
    if use_bias:
        feature = torch.cat([feature, torch.ones(1)])

    return feature @ weights


def rmse_and_bias(reference, estimate):
    error = estimate - reference
    rmse = torch.sqrt(torch.mean(error**2)).item()
    bias = torch.mean(error).item()

    return rmse, bias


# =============================================================================
# 7. Experiment 1 - nD=1: intercept or missing information?
# =============================================================================

nD = 1
ZETA_1 = build_training_zeta(nD)

# Thin-plate RBF lifting, matching the I/O Koopman experiment.
torch.manual_seed(5001)
centers_1 = -1.0 + 2.0 * torch.rand((Nrbf, ZETA_1.shape[1]))


def lift_1(zeta):
    diff = zeta.unsqueeze(-2) - centers_1
    r = torch.linalg.vector_norm(diff, dim=-1)
    phi = r**2 * torch.log(r + 1e-6)

    return torch.cat([zeta, phi], dim=-1)


Z_1 = lift_1(ZETA_1)

w_delay_no_bias = fit_decoder(ZETA_1, X1_train_target, use_bias=False)
w_delay_bias = fit_decoder(ZETA_1, X1_train_target, use_bias=True)
w_koop_no_bias = fit_decoder(Z_1, X1_train_target, use_bias=False)
w_koop_bias = fit_decoder(Z_1, X1_train_target, use_bias=True)

# Use the same validation interval for all comparisons below.
test_start_k = max_nD
x1_true_common = X_true[test_start_k:, 0]

x1_delay_no_bias = []
x1_delay_bias = []
x1_koop_no_bias = []
x1_koop_bias = []

for k in range(test_start_k, N_test + 1):
    zeta = build_test_zeta(k, 1)
    z = lift_1(zeta)

    x1_delay_no_bias.append(apply_decoder(zeta, w_delay_no_bias, False))
    x1_delay_bias.append(apply_decoder(zeta, w_delay_bias, True))
    x1_koop_no_bias.append(apply_decoder(z, w_koop_no_bias, False))
    x1_koop_bias.append(apply_decoder(z, w_koop_bias, True))

x1_delay_no_bias = torch.stack(x1_delay_no_bias)
x1_delay_bias = torch.stack(x1_delay_bias)
x1_koop_no_bias = torch.stack(x1_koop_no_bias)
x1_koop_bias = torch.stack(x1_koop_bias)

intercept_results = {
    "linear_no_bias": rmse_and_bias(x1_true_common, x1_delay_no_bias),
    "linear_bias": rmse_and_bias(x1_true_common, x1_delay_bias),
    "koopman_no_bias": rmse_and_bias(x1_true_common, x1_koop_no_bias),
    "koopman_bias": rmse_and_bias(x1_true_common, x1_koop_bias),
}

print("\n" + "=" * 86)
print("EXPERIMENT 1 - DOES AN INTERCEPT REMOVE THE x1 OFFSET? (nD = 1)")
print("=" * 86)
for name, (rmse, bias) in intercept_results.items():
    print(f"{name:>20s}: RMSE={rmse:.8f} | mean error={bias:+.8f}")


# =============================================================================
# 8. Experiment 2 - delay sweep
# =============================================================================

sweep_results = []
sweep_curves = {}

for nD in nD_values:
    zeta_train = build_training_zeta(nD)
    n_zeta = zeta_train.shape[1]

    # Linear delay decoder with intercept.
    w_delay = fit_decoder(zeta_train, X1_train_target, use_bias=True)

    zeta_augmented = torch.cat(
        [zeta_train, torch.ones((zeta_train.shape[0], 1))],
        dim=1,
    )
    cond_delay = torch.linalg.cond(zeta_augmented).item()

    # Koopman decoder with the same fixed number of RBFs for every nD.
    torch.manual_seed(5000 + nD)
    centers = -1.0 + 2.0 * torch.rand((Nrbf, n_zeta))

    def lift(zeta):
        diff = zeta.unsqueeze(-2) - centers
        r = torch.linalg.vector_norm(diff, dim=-1)
        phi = r**2 * torch.log(r + 1e-6)

        return torch.cat([zeta, phi], dim=-1)

    z_train = lift(zeta_train)
    w_koop = fit_decoder(z_train, X1_train_target, use_bias=True)

    z_augmented = torch.cat(
        [z_train, torch.ones((z_train.shape[0], 1))],
        dim=1,
    )
    cond_koop = torch.linalg.cond(z_augmented).item()

    x1_hat_delay = []
    x1_hat_koop = []

    for k in range(test_start_k, N_test + 1):
        zeta = build_test_zeta(k, nD)
        z = lift(zeta)

        x1_hat_delay.append(apply_decoder(zeta, w_delay, True))
        x1_hat_koop.append(apply_decoder(z, w_koop, True))

    x1_hat_delay = torch.stack(x1_hat_delay)
    x1_hat_koop = torch.stack(x1_hat_koop)

    rmse_delay, bias_delay = rmse_and_bias(x1_true_common, x1_hat_delay)
    rmse_koop, bias_koop = rmse_and_bias(x1_true_common, x1_hat_koop)

    sweep_results.append({
        "nD": nD,
        "n_zeta": n_zeta,
        "rmse_delay": rmse_delay,
        "bias_delay": bias_delay,
        "rmse_koopman": rmse_koop,
        "bias_koopman": bias_koop,
        "cond_delay": cond_delay,
        "cond_koopman": cond_koop,
    })

    sweep_curves[nD] = {
        "time": t_test[test_start_k:].clone(),
        "true": x1_true_common.clone(),
        "delay": x1_hat_delay.clone(),
        "koopman": x1_hat_koop.clone(),
    }

print("\n" + "=" * 108)
print("EXPERIMENT 2 - DELAY SWEEP")
print(f"Common training anchor: k={anchor_k} ({anchor_k*dt:.2f} s)")
print(f"Common test interval  : k={test_start_k} ... {N_test}")
print("=" * 108)
print(" nD | n_zeta | linear RMSE | linear bias | Koopman RMSE | Koopman bias | cond(linear)")
print("-" * 108)
for result in sweep_results:
    print(
        f"{result['nD']:>3d} | {result['n_zeta']:>6d} | "
        f"{result['rmse_delay']:.8f} | {result['bias_delay']:+.8f} | "
        f"{result['rmse_koopman']:.8f} | {result['bias_koopman']:+.8f} | "
        f"{result['cond_delay']:.3e}"
    )


# =============================================================================
# 9. Save numeric results
# =============================================================================

base_dir = Path(__file__).parent
csv_path = base_dir / "duffing_input_output_delay_sweep.csv"

with csv_path.open("w", encoding="utf-8") as file:
    file.write(
        "nD,n_zeta,rmse_delay,bias_delay,rmse_koopman,bias_koopman,"
        "cond_delay,cond_koopman\n"
    )

    for result in sweep_results:
        file.write(
            f"{result['nD']},{result['n_zeta']},"
            f"{result['rmse_delay']:.12g},{result['bias_delay']:.12g},"
            f"{result['rmse_koopman']:.12g},{result['bias_koopman']:.12g},"
            f"{result['cond_delay']:.12g},{result['cond_koopman']:.12g}\n"
        )


# =============================================================================
# 10. Plots
# =============================================================================

# Intercept comparison.
plt.figure(figsize=(12, 5))
plt.plot(t_test[test_start_k:].numpy(), x1_true_common.numpy(), label="True x1")
plt.plot(
    t_test[test_start_k:].numpy(),
    x1_koop_no_bias.numpy(),
    "--",
    label="Koopman decoder - no intercept",
)
plt.plot(
    t_test[test_start_k:].numpy(),
    x1_koop_bias.numpy(),
    "--",
    label="Koopman decoder - with intercept",
)
plt.title("Hidden-state x1 reconstruction: effect of intercept (nD = 1)")
plt.xlabel("Time [s]")
plt.ylabel("x1")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(base_dir / "duffing_x1_decoder_bias.png", dpi=160)
plt.close()

# RMSE versus nD.
plt.figure(figsize=(8, 5))
plt.plot(
    [r["nD"] for r in sweep_results],
    [r["rmse_delay"] for r in sweep_results],
    marker="o",
    label="Linear delay decoder",
)
plt.plot(
    [r["nD"] for r in sweep_results],
    [r["rmse_koopman"] for r in sweep_results],
    marker="o",
    label="Koopman decoder",
)
plt.title("Hidden-state reconstruction error versus I/O memory")
plt.xlabel("Number of delays nD")
plt.ylabel("RMSE of x1")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(base_dir / "duffing_x1_delay_sweep_rmse.png", dpi=160)
plt.close()

# Signed mean error versus nD.
plt.figure(figsize=(8, 5))
plt.plot(
    [r["nD"] for r in sweep_results],
    [r["bias_delay"] for r in sweep_results],
    marker="o",
    label="Linear delay decoder",
)
plt.plot(
    [r["nD"] for r in sweep_results],
    [r["bias_koopman"] for r in sweep_results],
    marker="o",
    label="Koopman decoder",
)
plt.axhline(0.0, linewidth=1)
plt.title("Signed x1 reconstruction bias versus I/O memory")
plt.xlabel("Number of delays nD")
plt.ylabel("Mean error of x1 estimate")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(base_dir / "duffing_x1_delay_sweep_bias.png", dpi=160)
plt.close()

# Representative curves.
for nD in [1, 2, 5, 20]:
    curve = sweep_curves[nD]

    plt.figure(figsize=(12, 5))
    plt.plot(curve["time"].numpy(), curve["true"].numpy(), label="True x1")
    plt.plot(
        curve["time"].numpy(),
        curve["delay"].numpy(),
        "--",
        label="Linear delay decoder",
    )
    plt.plot(
        curve["time"].numpy(),
        curve["koopman"].numpy(),
        "--",
        label="Koopman decoder",
    )
    plt.title(f"Hidden-state reconstruction from I/O history: nD = {nD}")
    plt.xlabel("Time [s]")
    plt.ylabel("x1")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / f"duffing_x1_reconstruction_nD_{nD}.png", dpi=160)
    plt.close()

print(f"\nCSV saved to: {csv_path}")
print("Plots saved next to the script.")
