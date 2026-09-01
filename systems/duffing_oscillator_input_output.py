# Duffing Oscillator - Input/Output Koopman identification
# dot{x1} = x2
# dot{x2} = -delta*x2 - x1*cos(x1 + x2) + u
# y = x2

import matplotlib.pyplot as plt
import torch

# For reproducibility purposes
torch.set_default_dtype(torch.float64)
torch.manual_seed(156)

# Parameters of simulation
dt = 0.01
delta = 2.0

# Parameters of the training dataset
N_trajectories = 20000
N_steps = 2
Nrbf = 10

# Output matrix: y = x2
Cy = torch.tensor([0.0, 1.0])


# Duffing oscillator dynamics, used only to simulate the plant
def dynamics(x, u):
    x1 = x[..., 0]
    x2 = x[..., 1]

    dx1 = x2
    dx2 = -delta * x2 - x1 * torch.cos(x1 + x2) + u

    return torch.stack([dx1, dx2], dim=-1)


# Simulation of one time step using RK4
def simulate_step(x, u):
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# Output of the Duffing oscillator
def output(x):
    return x @ Cy


# Input/output training dataset
# zeta_k = [y_k, y_(k-1), u_(k-1)]
x0 = -1.0 + 2.0 * torch.rand((N_trajectories, 2))
u0 = -1.0 + 2.0 * torch.rand(N_trajectories)
u1 = -1.0 + 2.0 * torch.rand(N_trajectories)

y0 = output(x0)

x1 = simulate_step(x0, u0)
y1 = output(x1)

x2 = simulate_step(x1, u1)
y2 = output(x2)

# Delay states at k = 1 and k = 2
ZETA = torch.stack([y1, y0, u0], dim=1)
ZETA_next = torch.stack([y2, y1, u1], dim=1)

# Input associated with the transition ZETA -> ZETA_next
U = u1.unsqueeze(1)

n_zeta = ZETA.shape[1]


# Koopman lifting of the input/output delay state
rbf_centers = -1.0 + 2.0 * torch.rand((Nrbf, n_zeta))


def create_koopman_state(zeta):
    diff = zeta.unsqueeze(-2) - rbf_centers
    r = torch.linalg.vector_norm(diff, dim=-1)
    phi = r**2 * torch.log(r + 1e-6)

    return torch.cat([zeta, phi], dim=-1)


Z = create_koopman_state(ZETA)
Z_next = create_koopman_state(ZETA_next)


# Koopman with least squares regression
Theta = torch.cat([Z, U], dim=1)

K = torch.linalg.lstsq(
    Theta,
    Z_next,
).solution

Nz = Z.shape[1]

A = K[:Nz, :].T
B = K[Nz:, :].T


# Training diagnostics
Z_next_hat_train = Z @ A.T + U @ B.T
Y_next_hat_train = Z_next_hat_train[:, 0]
Y_next_train = ZETA_next[:, 0]

train_error = Y_next_hat_train - Y_next_train
train_mae = torch.mean(torch.abs(train_error))
train_rmse = torch.sqrt(torch.mean(train_error**2))

rho_A = torch.max(torch.abs(torch.linalg.eigvals(A)))


# Test against the real system
T_test = 10.0
N_test = int(T_test / dt)
t_test = torch.arange(N_test + 1) * dt

x0_test = torch.tensor([-0.6, 1.4])

U_test = 0.8 * torch.sin(
    torch.arange(N_test) * 0.2
)

x_true = x0_test.clone()
Y_true = [output(x_true)]

for k in range(N_test):
    x_true = simulate_step(x_true, U_test[k])
    Y_true.append(output(x_true))

Y_true = torch.stack(Y_true)


# One-step-ahead prediction using the real input/output history
Y_one_step = [Y_true[0], Y_true[1]]

for k in range(1, N_test):
    zeta_true = torch.stack([
        Y_true[k],
        Y_true[k - 1],
        U_test[k - 1],
    ])

    z_true = create_koopman_state(zeta_true)
    z_next = A @ z_true + B[:, 0] * U_test[k]
    Y_one_step.append(z_next[0])

Y_one_step = torch.stack(Y_one_step)


# Free rollout using one measured transition for initialization
zeta_initial = torch.stack([
    Y_true[1],
    Y_true[0],
    U_test[0],
])

z_koopman = create_koopman_state(zeta_initial)
Y_rollout = [Y_true[0], Y_true[1]]

for k in range(1, N_test):
    z_koopman = A @ z_koopman + B[:, 0] * U_test[k]
    Y_rollout.append(z_koopman[0])

Y_rollout = torch.stack(Y_rollout)


# Error metrics
def metrics(y_true, y_hat):
    error = y_hat - y_true

    mae = torch.mean(torch.abs(error))
    rmse = torch.sqrt(torch.mean(error**2))
    iae = dt * torch.sum(torch.abs(error))

    return mae.item(), rmse.item(), iae.item()


one_mae, one_rmse, one_iae = metrics(
    Y_true,
    Y_one_step,
)

roll_mae, roll_rmse, roll_iae = metrics(
    Y_true,
    Y_rollout,
)


# Results
print("\n" + "=" * 78)
print("DUFFING - INPUT/OUTPUT KOOPMAN IDENTIFICATION")
print("=" * 78)
print("Measured output: y = x2")
print("Delay state: zeta_k = [y_k, y_(k-1), u_(k-1)]")
print(f"Training trajectories: {N_trajectories}")
print(f"Delay-state dimension: {n_zeta}")
print(f"Lifted dimension: {Nz} = {n_zeta} raw I/O coordinates + {Nrbf} RBFs")

print("\nTraining - one-step output fit")
print(f"  MAE : {train_mae.item():.8f}")
print(f"  RMSE: {train_rmse.item():.8f}")

print("\nSpectral radius")
print(f"  rho(A): {rho_A.item():.8f}")

print("\nTest - one-step ahead")
print(f"  MAE={one_mae:.8f} | RMSE={one_rmse:.8f} | IAE={one_iae:.8f}")

print("\nTest - free rollout")
print(f"  MAE={roll_mae:.8f} | RMSE={roll_rmse:.8f} | IAE={roll_iae:.8f}")
print("=" * 78)


# Comparison of the output trajectories
plt.figure(figsize=(12, 5))
plt.plot(
    t_test.numpy(),
    Y_true.numpy(),
    label="Real system",
)
plt.plot(
    t_test.numpy(),
    Y_rollout.numpy(),
    "--",
    label="Koopman I/O",
)
plt.title("Output y = x2")
plt.xlabel("Time [s]")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()