"""Didactic Port-Hamiltonian identification of the modified Duffing system.

The PH structure J, G, H and R is known up to three coefficients (a, r0, r1).
We estimate these coefficients directly from data by least squares, without
neural networks, Adam or gradient-based training.
"""

from pathlib import Path

import torch
import matplotlib.pyplot as plt

# 1) Same benchmark as classic_koopman.py
# dq/dt = p
# dp/dt = -2*p - q*cos(q+p) + u

torch.set_default_dtype(torch.float64)
torch.manual_seed(156)
DT = 0.01


# Known plant dynamics
def true_dynamics(x, u):
    q, p = x.unbind(-1)
    return torch.stack((p, -2.0 * p - q * torch.cos(q + p) + u), dim=-1)


# 2) Exact local PH representation of the SAME dynamics
# x=[q,p]^T, dx/dt=(J-R) grad_x H + G*u
# partial H/partial q = a*q*cos(q), partial H/partial p = p.
# H = p^2 / 2 + q*sin(q) + cos(q) - 1
# J = [[0,1],[-1,0]], G = [0,1]^T
# R = diag(0, r(q,p)), with r >= 2 - |q|.
# The bound guarantees R >= 0 on |q| <= 2.

# Creted for normalization purposes
def sinc_half(p):
    """sin(p/2)/(p/2), including the continuous value 1 at p=0."""
    return torch.sinc(p / (2.0 * torch.pi))


def reference_energy(x):
    q, p = x.unbind(-1)
    return 0.5 * p.square() + q * torch.sin(q) + torch.cos(q) - 1.0


def reference_dissipation(x):
    q, p = x.unbind(-1)
    return 2.0 - q * torch.sin(q + p / 2.0) * sinc_half(p)


def reference_ph_dynamics(x, u):
    q, p = x.unbind(-1)
    h_q = q * torch.cos(q)
    h_p = p
    r = reference_dissipation(x)
    return torch.stack((h_p, -h_q - r * h_p + u), dim=-1)


# 3) Regression from the PH equation:
# dp/dt - u = -a*q*cos(q) - r0*p - r1*(1-D(q,p))*p.
# For known (q,p,u,dp/dt), this is LINEAR in theta = [a, r0, r1].
# Enforce theta >= 0 to keep dissipation nonnegative on |q| <= 1.
def ph_features(x):
    q, p = x.unbind(-1)
    D = q * torch.sin(q + p / 2.0) * sinc_half(p)
    return torch.stack((-q * torch.cos(q), -p, -(1.0 - D) * p), dim=-1)


def fit_nonnegative_lstsq(features, target):
    """Minimize ||features @ theta - target||^2 with theta >= 0.

    Only three parameters: test the 8 possible active subsets (including zero).
    Each candidate is an ordinary least-squares solve with torch.linalg.lstsq.
    """
    best_coefficients = torch.zeros(3, dtype=features.dtype, device=features.device)
    best_loss = torch.mean(target.square()).item()

    for mask in range(1, 8):
        active = [j for j in range(3) if mask & (1 << j)]
        solution = torch.linalg.lstsq(features[:, active], target).solution
        if torch.any(solution < -1e-10):
            continue  # Physical constraint: coefficients cannot be negative.

        coefficients = torch.zeros_like(best_coefficients)
        coefficients[active] = solution.clamp(min=0.0)
        loss = torch.mean((features @ coefficients - target).square()).item()
        if loss < best_loss:
            best_loss, best_coefficients = loss, coefficients

    return best_coefficients, best_loss


def ph_energy(x, coefficients):
    q, p = x.unbind(-1)
    a, _, _ = coefficients
    return 0.5 * p.square() + a * (q * torch.sin(q) + torch.cos(q) - 1.0)


def ph_dissipation(x, coefficients):
    q, p = x.unbind(-1)
    _, r0, r1 = coefficients
    D = q * torch.sin(q + p / 2.0) * sinc_half(p)
    return r0 + r1 * (1.0 - D)


def ph_dynamics(x, u, coefficients):
    # grad_x H = [partial H/partial q, partial H/partial p] = [a*q*cos(q), p].
    # J is canonical and R = diag(0, r).
    q, p = x.unbind(-1)
    a, _, _ = coefficients
    r = ph_dissipation(x, coefficients)
    return torch.stack((p, -a * q * torch.cos(q) - r * p + u), dim=-1)


def rk4_step(field, x, u, dt=DT):
    """One step with piecewise-constant input, matching the existing benchmark."""
    k1 = field(x, u)
    k2 = field(x + 0.5 * dt * k1, u)
    k3 = field(x + 0.5 * dt * k2, u)
    k4 = field(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def main():
    # A) Mathematical sanity check before any fitting.
    x_check = -1.0 + 2.0 * torch.rand(2000, 2)
    u_check = -1.0 + 2.0 * torch.rand(2000)
    mismatch = (true_dynamics(x_check, u_check) - reference_ph_dynamics(x_check, u_check)).abs().max()
    print(f"Exact local PH decomposition: max vector-field error = {mismatch.item():.3e}")
    print(f"Reference R minimum on samples = {reference_dissipation(x_check).min().item():.6f}")

    # B) Identify theta from simulated data. The exact time derivatives
    # dq/dt, dp/dt are evaluated using the known plant field.
    x_train = -1.0 + 2.0 * torch.rand(3000, 2)
    u_train = -1.0 + 2.0 * torch.rand(3000)
    xdot_train = true_dynamics(x_train, u_train)

    features = ph_features(x_train)
    target = xdot_train[:, 1] - u_train  # Y = dp/dt - u
    coefficients, train_mse = fit_nonnegative_lstsq(features, target)
    a, r0, r1 = coefficients

    print(f"Least-squares derivative MSE = {train_mse:.4e}")
    print(f"Identified: a={a.item():.6f}, r0={r0.item():.6f}, r1={r1.item():.6f}")
    print("Ground truth: a=1, r0=1, r1=1")

    # C) Shared hold-out test: one RK4 step and free trajectory rollout.
    x_test = -1.0 + 2.0 * torch.rand(500, 2)
    u_test = -1.0 + 2.0 * torch.rand(500)
    true_next = rk4_step(true_dynamics, x_test, u_test)
    ph_next = rk4_step(lambda x, u: ph_dynamics(x, u, coefficients), x_test, u_test)
    print(f"Held-out one-step RMSE = {torch.mean((true_next - ph_next).square()).sqrt().item():.3e}")

    x0 = torch.tensor([-0.6, 1.4])  # same test initial state as existing Koopman code
    steps = 1000
    inputs = 0.8 * torch.sin(0.2 * torch.arange(steps, dtype=torch.float64))
    true_states = [x0]
    ph_states = [x0]
    for u in inputs:
        true_states.append(rk4_step(true_dynamics, true_states[-1], u).detach())
        ph_states.append(rk4_step(lambda x, u: ph_dynamics(x, u, coefficients), ph_states[-1], u).detach())
    true_states = torch.stack(true_states)
    ph_states = torch.stack(ph_states)
    rollout_rmse = torch.mean((true_states - ph_states).square()).sqrt()
    print(f"10-second free-rollout RMSE = {rollout_rmse.item():.3e}")
    print(f"Observed |q| maximum during PH rollout = {ph_states[:,0].abs().max().item():.3f}")

    times = DT * torch.arange(steps + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for state_index, name in enumerate(("q", "p")):
        ax = axes[state_index]
        ax.plot(times, true_states[:, state_index], label=f"True {name}")
        ax.plot(times, ph_states[:, state_index], "--", label=f"Learned PH {name}")
        ax.set_title(f"Duffing — {name}")
        ax.set_xlabel("Time [s]")
        ax.legend()
        ax.grid()
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parent / "duffing_ph_intro_rollout.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
