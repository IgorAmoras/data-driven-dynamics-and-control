# Duffing Oscillator - SINDy with measurement and process disturbances
# dot{x1} = x2
# dot{x2} = -delta*x2 - x1*cos(x1 + x2) + u + w
# y = x2 + v

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
process_noise = 0.05
noise_levels = [0.01, 0.03, 0.05, 0.10]

# Duffing oscillator dynamics, used only to simulate the plant
def dynamics(x, u, w):
    x1, x2 = x
    return torch.stack([
        x2,
        -delta * x2 - x1 * torch.cos(x1 + x2) + u + w,
    ])

# Simulation of one time step using RK4
def simulate_step(x, u, w):
    k1 = dynamics(x, u, w)
    k2 = dynamics(x + 0.5 * dt * k1, u, w)
    k3 = dynamics(x + 0.5 * dt * k2, u, w)
    k4 = dynamics(x + dt * k3, u, w)
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

# SINDy library containing the true nonlinear structure
def structured_library(x1, y, u):
    s = x1 + y
    return torch.column_stack([
        torch.ones_like(y),
        x1,
        y,
        u,
        torch.cos(s),
        x1 * torch.cos(s),
    ])

# SINDy library without the true nonlinear term
def blind_library(x1, y, u):
    s = x1 + y
    return torch.column_stack([
        torch.ones_like(y),
        y,
        u,
        x1**2,
        y**2,
        x1 * y,
        torch.sin(s),
        torch.cos(s),
        x1 * torch.sin(s),
    ])

# Sequential thresholded least squares
def sindy(theta, target):
    xi = torch.linalg.lstsq(theta, target).solution
    for _ in range(10):
        active = torch.abs(xi) >= threshold
        if not torch.any(active):
            break
        new_xi = torch.zeros_like(xi)
        new_xi[active] = torch.linalg.lstsq(theta[:, active], target).solution
        xi = new_xi
    return xi

# SINDy identification and hidden-state reconstruction
def identify(y, u, library):
    dy = (y[1:] - y[:-1]) / dt
    y_mid = 0.5 * (y[:-1] + y[1:])
    best = None

    for x10 in torch.linspace(-1.0, 1.0, 101):
        x1 = reconstruct_x1(y, x10)
        x1_mid = 0.5 * (x1[:-1] + x1[1:])
        theta = library(x1_mid, y_mid, u)
        xi = sindy(theta, dy)
        error = torch.mean((theta @ xi - dy) ** 2)

        if best is None or error < best[0]:
            best = (error, x10, xi)

    return best[1], reconstruct_x1(y, best[1]), best[2]

# Real trajectory with 5% process disturbance
x0 = torch.tensor([-0.6, 1.4])
U = 0.8 * torch.sin(torch.arange(N) * 0.2)
W = process_noise * torch.randn(N)

X_true = [x0]
for u, w in zip(U, W):
    X_true.append(simulate_step(X_true[-1], u, w))
X_true = torch.stack(X_true)
Y_true = X_true[:, 1]

# Noisy measurements
measurements = {
    noise: Y_true + noise * torch.std(Y_true) * torch.randn_like(Y_true)
    for noise in noise_levels
}

structured_results = {}
blind_results = {}

for noise, Y in measurements.items():
    x10_hat, x1_hat, xi = identify(Y, U, structured_library)
    structured_results[noise] = (x10_hat, x1_hat, xi)

    x10_hat, x1_hat, xi = identify(Y, U, blind_library)
    blind_results[noise] = (x10_hat, x1_hat, xi)

# State reconstruction errors
print("\n" + "=" * 72)
print("DUFFING - SINDY WITH DISTURBANCES")
print("=" * 72)
print("Noise    Structured RMSE x1    Blind RMSE x1")
for noise in noise_levels:
    structured_rmse = torch.sqrt(torch.mean(
        (structured_results[noise][1] - X_true[:, 0])**2
    ))
    blind_rmse = torch.sqrt(torch.mean(
        (blind_results[noise][1] - X_true[:, 0])**2
    ))
    print(f"{100 * noise:>4.0f}%      {structured_rmse.item():>12.6f}       {blind_rmse.item():>12.6f}")
print("=" * 72)

# Identified equations for the 5% measurement-noise case
structured_names = [
    "1", "x1", "y", "u", "cos(x1+y)", "x1*cos(x1+y)"
]
blind_names = [
    "1", "y", "u", "x1^2", "y^2", "x1*y",
    "sin(x1+y)", "cos(x1+y)", "x1*sin(x1+y)"
]

for title, result, names in [
    ("Structured library", structured_results[0.05], structured_names),
    ("Blind library", blind_results[0.05], blind_names),
]:
    print(f"\n{title} - 5% measurement noise")
    print(f"Estimated x1(0): {result[0].item():.4f}")
    print("dot{y} =")
    for name, coefficient in zip(names, result[2]):
        if abs(coefficient) >= threshold:
            print(f"  {coefficient.item():+.6f} * {name}")

# Comparison of the real and reconstructed states
t = torch.arange(N + 1) * dt

def plot_result(noise, results, title):
    Y = measurements[noise]
    x1_hat = results[noise][1]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t.numpy(), X_true[:, 0].numpy(), label="Real x1")
    plt.plot(t.numpy(), x1_hat.numpy(), "--", label="Estimated x1")
    plt.title(f"{title} - hidden state x1 - {100 * noise:.0f}% noise")
    plt.xlabel("Time [s]")
    plt.ylabel("x1")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t.numpy(), X_true[:, 1].numpy(), label="Real x2")
    plt.plot(t.numpy(), Y.numpy(), "--", label="Measured y")
    plt.title(f"{title} - measured state x2 - {100 * noise:.0f}% noise")
    plt.xlabel("Time [s]")
    plt.ylabel("x2")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# Structured library
# plot_result(0.01, structured_results, "Structured SINDy")
# plot_result(0.03, structured_results, "Structured SINDy")
plot_result(0.05, structured_results, "Structured SINDy")
# plot_result(0.10, structured_results, "Structured SINDy")

# Blind library
# plot_result(0.01, blind_results, "Blind SINDy")
# plot_result(0.03, blind_results, "Blind SINDy")
plot_result(0.05, blind_results, "Blind SINDy")
# plot_result(0.10, blind_results, "Blind SINDy")
