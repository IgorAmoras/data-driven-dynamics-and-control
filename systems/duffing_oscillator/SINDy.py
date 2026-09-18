# Duffing Oscillator - SINDy with hidden-state reconstruction
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
T = 10.0
N = int(T / dt)
delta = 2.0
threshold = 0.1

# Duffing oscillator dynamics, used only to simulate the plant
def dynamics(x, u):
    x1, x2 = x
    return torch.stack([
        x2,
        -delta * x2 - x1 * torch.cos(x1 + x2) + u,
    ])

# Simulation of one time step using RK4
def simulate_step(x, u):
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# Reconstruction from the known physics: dot{x1} = y
def reconstruct_x1(y, x10):
    x1 = torch.empty_like(y)
    x1[0] = x10
    x1[1:] = x10 + dt * torch.cumsum(
        0.5 * (y[:-1] + y[1:]),
        dim=0,
    )
    return x1

# SINDy candidate library
def library(x1, y, u):
    s = x1 + y
    return torch.column_stack([
        torch.ones_like(y),
        x1,
        y,
        u,
        torch.cos(s),
        x1 * torch.cos(s),
    ])

# Sequential thresholded least squares
def sindy(theta, target):
    xi = torch.linalg.lstsq(theta, target).solution
    for _ in range(10):
        active = torch.abs(xi) >= threshold
        xi.zero_()
        xi[active] = torch.linalg.lstsq(theta[:, active], target).solution
    return xi

# Real trajectory
x0 = torch.tensor([-0.6, 1.4])
U = 0.8 * torch.sin(torch.arange(N) * 0.2)

X_true = [x0]
for u in U:
    X_true.append(simulate_step(X_true[-1], u))
X_true = torch.stack(X_true)

# Measurements available to the identifier
Y = X_true[:, 1]
dY = (Y[1:] - Y[:-1]) / dt
Y_mid = 0.5 * (Y[:-1] + Y[1:])

# Search for the hidden initial state
best = None
for x10 in torch.linspace(-1.0, 1.0, 101):
    x1 = reconstruct_x1(Y, x10)
    x1_mid = 0.5 * (x1[:-1] + x1[1:])
    theta = library(x1_mid, Y_mid, U)
    xi = sindy(theta, dY)
    error = torch.mean((theta @ xi - dY) ** 2)

    if best is None or error < best[0]:
        best = (error, x10, xi)

_, x10_hat, xi = best
x1_hat = reconstruct_x1(Y, x10_hat)
X_hat = torch.column_stack([x1_hat, Y])

# Results
names = [
    "1",
    "x1",
    "y",
    "u",
    "cos(x1+y)",
    "x1*cos(x1+y)",
]

print("\n" + "=" * 70)
print("DUFFING - SINDY WITH HIDDEN-STATE RECONSTRUCTION")
print("=" * 70)
print(f"Estimated x1(0): {x10_hat.item():.4f}")
print(f"Real x1(0):      {X_true[0, 0].item():.4f}")
print("\nIdentified equation: dot{y} =")
for name, coefficient in zip(names, xi):
    if abs(coefficient) >= threshold:
        print(f"  {coefficient.item():+.6f} * {name}")
print("=" * 70)

# Comparison of the real and reconstructed states
t = torch.arange(N + 1) * dt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t.numpy(), X_true[:, 0].numpy(), label="Real x1")
plt.plot(t.numpy(), X_hat[:, 0].numpy(), "--", label="Estimated x1")
plt.title("Hidden state x1")
plt.xlabel("Time [s]")
plt.ylabel("x1")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t.numpy(), X_true[:, 1].numpy(), label="Real x2")
plt.plot(t.numpy(), X_hat[:, 1].numpy(), "--", label="Measured y = x2")
plt.title("Measured state x2")
plt.xlabel("Time [s]")
plt.ylabel("x2")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
