# Duffing oscillator - input/output Koopman identification
#
# Goal of this experiment

# Identify a predictive model using ONLY measured input u and output y.
# The internal state x = [x1, x2] is used only inside the simulated plant
# to generate data. It is never given to the identification algorithm.

# Plant (same as in SiShiAta 2026):
#   dx1/dt = x2
#   dx2/dt = -delta*x2 - x1*cos(x1 + x2) + u

# Output used in the article:
#   y = [0 1] x = x2

# With one delay (nD = 1), we define an input/output delay state:
#   zeta_k = [y_k, y_{k-1}, u_{k-1}]^T
#
# and identify
#   zeta_{k+1} ~= A_delay zeta_k + B_delay u_k            
#
# followed by the Koopman-lifted model
#   z_k = psi(zeta_k) = [zeta_k, RBF_1(zeta_k), ..., RBF_N(zeta_k)]^T
#   z_{k+1} ~= A z_k + B u_k                               
#
# No stability constraint is imposed here on purpose. This file isolates the
# input/output question. LMI/Lyapunov constraints can be added afterwards.

from pathlib import Path

import matplotlib.pyplot as plt
import torch
# Simulation parameters
torch.set_default_dtype(torch.float64)
torch.manual_seed(156)

dt = 0.01
delta = 2.0

N_trajectories = 20000
N_steps = 2              # enough for y0 -> y1 -> y2 when using one delay
Nrbf = 10

# Output matrix used in the authors Duffing example: y = x2
Cy = torch.tensor([0.0, 1.0])


# Dynamics of the Duffing oscillator, used for simulation only. The identifier never receives x, only y and u.

def dynamics(x, u):
    x1 = x[..., 0]
    x2 = x[..., 1]

    dx1 = x2
    dx2 = -delta * x2 - x1 * torch.cos(x1 + x2) + u

    return torch.stack([dx1, dx2], dim=-1)

# Using RK4 integration

def simulate_step(x, u):
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def output(x):
    return x @ Cy


# i/o training dataset

# For every independent trajectory:
#
#   x0 --u0--> x1 --u1--> x2
#    |          |          |
#   y0         y1         y2
#
# From only y and u we construct:
#
#   zeta_1 = [y1, y0, u0]
#   zeta_2 = [y2, y1, u1]
#
# and therefore one identification sample:
#
#   (zeta_1, u1) -> zeta_2
#
# x0, x1 and x2 exist only to simulate the hidden plant. They are NOT passed
# to the least-squares identification. The operations below are vectorized only
# for speed: each row is still one independent two-step trajectory.

x0 = -1.0 + 2.0 * torch.rand((N_trajectories, 2))
u0 = -1.0 + 2.0 * torch.rand(N_trajectories)
u1 = -1.0 + 2.0 * torch.rand(N_trajectories)

y0 = output(x0)

x1 = simulate_step(x0, u0)
y1 = output(x1)

x2 = simulate_step(x1, u1)
y2 = output(x2)

# One-delay I/O state at k=1 and k=2.
ZETA = torch.stack([y1, y0, u0], dim=1)
ZETA_next = torch.stack([y2, y1, u1], dim=1)

# Current input for the transition zeta_1 -> zeta_2 is u1.
U = u1.unsqueeze(1)


# =============================================================================
# 4. Linear input/output delay model (baseline, BEFORE Koopman lifting)
# =============================================================================
#
# This baseline answers an important research question:
# does the nonlinear lifting actually improve on a simple delay model?
#
# Regression:
#   ZETA_next ~= ZETA A_delay^T + U B_delay^T

Theta_delay = torch.cat([ZETA, U], dim=1)

K_delay = torch.linalg.lstsq(
    Theta_delay,
    ZETA_next,
).solution

n_zeta = ZETA.shape[1]

A_delay = K_delay[:n_zeta, :].T
B_delay = K_delay[n_zeta:, :].T


# =============================================================================
# 5. Koopman lifting of the INPUT/OUTPUT delay state
# =============================================================================
#
# Important conceptual change relative to the original state-based code:
# the RBFs now live in zeta-space = [y_k, y_{k-1}, u_{k-1}], not x-space.

# Centers in the same nominal [-1, 1] box used for training signals.
rbf_centers = -1.0 + 2.0 * torch.rand((Nrbf, n_zeta))


def create_koopman_state(zeta):
    """Lift one zeta or a batch of zetas with thin-plate RBFs."""
    # zeta shape can be [3] or [N, 3]. The new axis aligns every zeta with
    # all RBF centers, giving distances with shape [Nrbf] or [N, Nrbf].
    diff = zeta.unsqueeze(-2) - rbf_centers
    r = torch.linalg.vector_norm(diff, dim=-1)
    phi = r**2 * torch.log(r + 1e-6)

    # Keeping zeta itself in the lifted vector makes the measured output
    # directly readable: the first component of z is y_k.
    return torch.cat([zeta, phi], dim=-1)


Z = create_koopman_state(ZETA)
Z_next = create_koopman_state(ZETA_next)


# =============================================================================
# 6. Unconstrained Koopman least-squares identification
# =============================================================================
#
#   z_{k+1} ~= A z_k + B u_k
#
# Same least-squares idea as in duffing_oscillator.py; only the definition of
# the state being lifted has changed.

Theta_koopman = torch.cat([Z, U], dim=1)

K = torch.linalg.lstsq(
    Theta_koopman,
    Z_next,
).solution

Nz = Z.shape[1]

A = K[:Nz, :].T
B = K[Nz:, :].T


# =============================================================================
# 7. State reconstruction from the I/O representation
# =============================================================================
#
# The identified dynamics above never use the hidden state x.  Here we learn a
# SEPARATE decoder whose only purpose is to reconstruct the unmeasured state x1
# from the delay coordinates.  During training the simulator provides x1 as a
# label; during prediction the decoder receives only the I/O representation.
#
# Since the measured output is y = x2, the second physical state is already
# available directly from the output.  Therefore only x1 needs to be learned.
#
# Linear delay decoder:
#   x1_k ~= w_delay^T zeta_k
#
# Koopman decoder:
#   x1_k ~= w_koopman^T z_k
#
# ZETA and Z correspond to time k=1 in every training trajectory, so the target
# state is x1 (the variable named x1 below is the physical vector [x1, x2]).

X_state_train = x1
X1_state_train = X_state_train[:, 0]

w_x1_delay = torch.linalg.lstsq(
    ZETA,
    X1_state_train,
).solution

w_x1_koopman = torch.linalg.lstsq(
    Z,
    X1_state_train,
).solution

X1_train_hat_delay = ZETA @ w_x1_delay
X1_train_hat_koopman = Z @ w_x1_koopman

state_train_rmse_delay = torch.sqrt(
    torch.mean((X1_train_hat_delay - X1_state_train) ** 2)
)
state_train_rmse_koopman = torch.sqrt(
    torch.mean((X1_train_hat_koopman - X1_state_train) ** 2)
)


# =============================================================================
# 8. Training diagnostics
# =============================================================================

Z_next_hat_train = Z @ A.T + U @ B.T
Y_next_hat_train = Z_next_hat_train[:, 0]
Y_next_train = ZETA_next[:, 0]

train_error = Y_next_hat_train - Y_next_train
train_mae = torch.mean(torch.abs(train_error))
train_rmse = torch.sqrt(torch.mean(train_error**2))

rho_A = torch.max(torch.abs(torch.linalg.eigvals(A)))
rho_A_delay = torch.max(torch.abs(torch.linalg.eigvals(A_delay)))


# =============================================================================
# 9. Test trajectory - same initial condition and sinusoidal input pattern
# =============================================================================

T_test = 10.0
N_test = int(T_test / dt)
t_test = torch.arange(N_test + 1) * dt

x0_test = torch.tensor([-0.6, 1.4])

U_test = 0.8 * torch.sin(
    torch.arange(N_test) * 0.2
)

# True nonlinear state/output trajectory.  X_true is kept ONLY for
# validation of the reconstruction; the I/O identifier never receives it.
x_true = x0_test.clone()
X_true = [x_true.clone()]
Y_true = [output(x_true)]

for k in range(N_test):
    x_true = simulate_step(x_true, U_test[k])
    X_true.append(x_true.clone())
    Y_true.append(output(x_true))

X_true = torch.stack(X_true)
Y_true = torch.stack(Y_true)


# =============================================================================
# 10. One-step-ahead test (teacher forcing)
# =============================================================================
#
# At every step we rebuild zeta_k from TRUE measured past signals and predict
# only y_{k+1}. This measures local identification quality without accumulation.

Y_one_step_delay = [Y_true[0], Y_true[1]]
Y_one_step_koopman = [Y_true[0], Y_true[1]]

for k in range(1, N_test):
    zeta_true = torch.stack([
        Y_true[k],
        Y_true[k - 1],
        U_test[k - 1],
    ])

    # Linear delay baseline.
    zeta_next_delay = A_delay @ zeta_true + B_delay[:, 0] * U_test[k]
    Y_one_step_delay.append(zeta_next_delay[0])

    # Koopman model.
    z_true = create_koopman_state(zeta_true)
    z_next_koopman = A @ z_true + B[:, 0] * U_test[k]
    Y_one_step_koopman.append(z_next_koopman[0])

Y_one_step_delay = torch.stack(Y_one_step_delay)
Y_one_step_koopman = torch.stack(Y_one_step_koopman)

# State reconstruction using TRUE I/O history at each time.  This isolates the
# quality of the decoder from rollout error in A and B.  The first sample with a
# complete one-delay history is k=1.
X1_reconstructed_delay = [X_true[0, 0]]
X1_reconstructed_koopman = [X_true[0, 0]]

for k in range(1, N_test + 1):
    zeta_true = torch.stack([
        Y_true[k],
        Y_true[k - 1],
        U_test[k - 1],
    ])
    z_true = create_koopman_state(zeta_true)

    X1_reconstructed_delay.append(zeta_true @ w_x1_delay)
    X1_reconstructed_koopman.append(z_true @ w_x1_koopman)

X1_reconstructed_delay = torch.stack(X1_reconstructed_delay)
X1_reconstructed_koopman = torch.stack(X1_reconstructed_koopman)


# =============================================================================
# 11. Free rollout test
# =============================================================================
#
# We need ONE measured transition to initialize the delayed state:
#   y0 --u0--> y1
#
# After that the model runs freely; future true outputs are not fed back.

zeta_initial = torch.stack([
    Y_true[1],
    Y_true[0],
    U_test[0],
])

# ----- Linear delay baseline rollout -----
zeta_delay = zeta_initial.clone()
Y_rollout_delay = [Y_true[0], Y_true[1]]
X1_rollout_delay = [X_true[0, 0], zeta_delay @ w_x1_delay]

for k in range(1, N_test):
    zeta_delay = A_delay @ zeta_delay + B_delay[:, 0] * U_test[k]
    Y_rollout_delay.append(zeta_delay[0])
    X1_rollout_delay.append(zeta_delay @ w_x1_delay)

Y_rollout_delay = torch.stack(Y_rollout_delay)
X1_rollout_delay = torch.stack(X1_rollout_delay)

# ----- Koopman rollout -----
z_koopman = create_koopman_state(zeta_initial)
Y_rollout_koopman = [Y_true[0], Y_true[1]]
X1_rollout_koopman = [X_true[0, 0], z_koopman @ w_x1_koopman]

for k in range(1, N_test):
    z_koopman = A @ z_koopman + B[:, 0] * U_test[k]
    Y_rollout_koopman.append(z_koopman[0])
    X1_rollout_koopman.append(z_koopman @ w_x1_koopman)

Y_rollout_koopman = torch.stack(Y_rollout_koopman)
X1_rollout_koopman = torch.stack(X1_rollout_koopman)

# The second physical state is the measured output itself: x2 = y.
X_rollout_delay = torch.stack([X1_rollout_delay, Y_rollout_delay], dim=1)
X_rollout_koopman = torch.stack([X1_rollout_koopman, Y_rollout_koopman], dim=1)


# =============================================================================
# 12. Metrics
# =============================================================================


def metrics(y_true, y_hat):
    error = y_hat - y_true

    mae = torch.mean(torch.abs(error))
    rmse = torch.sqrt(torch.mean(error**2))
    iae = dt * torch.sum(torch.abs(error))

    return mae.item(), rmse.item(), iae.item()


one_delay_mae, one_delay_rmse, one_delay_iae = metrics(
    Y_true,
    Y_one_step_delay,
)
one_koop_mae, one_koop_rmse, one_koop_iae = metrics(
    Y_true,
    Y_one_step_koopman,
)

roll_delay_mae, roll_delay_rmse, roll_delay_iae = metrics(
    Y_true,
    Y_rollout_delay,
)
roll_koop_mae, roll_koop_rmse, roll_koop_iae = metrics(
    Y_true,
    Y_rollout_koopman,
)

# Hidden-state x1 reconstruction metrics.
_, state_decode_delay_rmse, _ = metrics(
    X_true[:, 0],
    X1_reconstructed_delay,
)
_, state_decode_koop_rmse, _ = metrics(
    X_true[:, 0],
    X1_reconstructed_koopman,
)
_, state_roll_delay_rmse, _ = metrics(
    X_true[:, 0],
    X1_rollout_delay,
)
_, state_roll_koop_rmse, _ = metrics(
    X_true[:, 0],
    X1_rollout_koopman,
)


# =============================================================================
# 13. Print results
# =============================================================================

print("\n" + "=" * 78)
print("DUFFING - INPUT/OUTPUT IDENTIFICATION")
print("=" * 78)
print(f"Measured output: y = x2")
print(f"Delay state: zeta_k = [y_k, y_(k-1), u_(k-1)]")
print(f"Training trajectories: {N_trajectories}")
print(f"Delay-state dimension: {n_zeta}")
print(f"Lifted dimension: {Nz} = {n_zeta} raw I/O coordinates + {Nrbf} RBFs")

print("\nTraining - Koopman one-step output fit")
print(f"  MAE : {train_mae.item():.8f}")
print(f"  RMSE: {train_rmse.item():.8f}")

print("\nSpectral radius")
print(f"  Linear delay A : {rho_A_delay.item():.8f}")
print(f"  Koopman A      : {rho_A.item():.8f}")

print("\nTest - one-step ahead (true history supplied at every step)")
print(f"  Linear delay -> MAE={one_delay_mae:.8f} | RMSE={one_delay_rmse:.8f} | IAE={one_delay_iae:.8f}")
print(f"  Koopman      -> MAE={one_koop_mae:.8f} | RMSE={one_koop_rmse:.8f} | IAE={one_koop_iae:.8f}")

print("\nTest - free rollout (only y0, y1 and u0 used for initialization)")
print(f"  Linear delay -> MAE={roll_delay_mae:.8f} | RMSE={roll_delay_rmse:.8f} | IAE={roll_delay_iae:.8f}")
print(f"  Koopman      -> MAE={roll_koop_mae:.8f} | RMSE={roll_koop_rmse:.8f} | IAE={roll_koop_iae:.8f}")

print("\nHidden-state reconstruction: x1 from I/O history")
print(f"  Training decoder RMSE - linear delay : {state_train_rmse_delay.item():.8f}")
print(f"  Training decoder RMSE - Koopman      : {state_train_rmse_koopman.item():.8f}")
print(f"  Test decoder RMSE (true I/O history) - linear delay : {state_decode_delay_rmse:.8f}")
print(f"  Test decoder RMSE (true I/O history) - Koopman      : {state_decode_koop_rmse:.8f}")
print(f"  Test x1 RMSE (free rollout) - linear delay : {state_roll_delay_rmse:.8f}")
print(f"  Test x1 RMSE (free rollout) - Koopman      : {state_roll_koop_rmse:.8f}")
print("=" * 78)


# =============================================================================
# 14. Plot: output and reconstructed physical states
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Measured/predicted output.
axes[0].plot(
    t_test.numpy(),
    Y_true.numpy(),
    label="Real output",
)
axes[0].plot(
    t_test.numpy(),
    Y_rollout_delay.numpy(),
    "--",
    label="Linear I/O delay",
)
axes[0].plot(
    t_test.numpy(),
    Y_rollout_koopman.numpy(),
    "--",
    label="Koopman I/O + RBF",
)
axes[0].set_title("Output prediction: y = x2")
axes[0].set_ylabel("y")
axes[0].grid()
axes[0].legend()

# Hidden physical state reconstructed from the I/O representation.
axes[1].plot(
    t_test.numpy(),
    X_true[:, 0].numpy(),
    label="True x1 (validation only)",
)
axes[1].plot(
    t_test.numpy(),
    X1_rollout_delay.numpy(),
    "--",
    label="Reconstructed x1 - linear delay",
)
axes[1].plot(
    t_test.numpy(),
    X1_rollout_koopman.numpy(),
    "--",
    label="Reconstructed x1 - Koopman",
)
axes[1].set_title("Hidden-state reconstruction from input/output history")
axes[1].set_ylabel("x1")
axes[1].grid()
axes[1].legend()

# x2 is directly measured because y = x2.  Plotting it as a state makes the
# distinction explicit when comparing the complete physical state vector.
axes[2].plot(
    t_test.numpy(),
    X_true[:, 1].numpy(),
    label="True x2",
)
axes[2].plot(
    t_test.numpy(),
    X_rollout_koopman[:, 1].numpy(),
    "--",
    label="Reconstructed x2 = predicted y",
)
axes[2].set_title("Second physical state (directly measured output)")
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("x2")
axes[2].grid()
axes[2].legend()

plt.tight_layout()

plot_path = Path(__file__).with_name("duffing_input_output_states.png")
plt.savefig(plot_path, dpi=160)

print(f"\nPlot saved to: {plot_path}")

# Uncomment the next line if you want the interactive window as well.
# plt.show()
