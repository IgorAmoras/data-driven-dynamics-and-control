"""Port-Hamiltonian identification of the repository's modified Duffing system.

Self-contained educational benchmark matching the repository's Duffing dynamics.
Stage PH-1: canonical J, known G, learn a scalar Hamiltonian coefficient
and two coefficients of positive dissipation on |q| <= 1.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1) Same benchmark as classic_koopman.py
# q_dot = p
# p_dot = -2*p - q*cos(q+p) + u

torch.set_default_dtype(torch.float64)
torch.manual_seed(156)
DT = 0.01


# Known plant dynamics
def true_dynamics(x, u):
    q, p = x.unbind(-1)
    return torch.stack((p, -2.0 * p - q * torch.cos(q + p) + u), dim=-1)


# 2) Exact local PH representation of the SAME dynamics
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


# 3) Identification: learn unknown H and R parameters, but preserve structure.
# H_theta = p^2/2 + a * V0(q)
# R_theta = diag(0, r0 + r1*[1 - q*sin(q+p/2)*sinc(p/2)])
# a, r0, r1 >= 0 are imposed by softplus.
# When |q| <= 1, the bracket is nonnegative, so R_theta >= 0.
# The true parameters are a = r0 = r1 = 1.

class ParametricPH(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_a = nn.Parameter(torch.tensor(0.0))
        self.raw_r0 = nn.Parameter(torch.tensor(0.0))
        self.raw_r1 = nn.Parameter(torch.tensor(0.0))

    def coefficients(self):
        return F.softplus(self.raw_a), F.softplus(self.raw_r0), F.softplus(self.raw_r1)

    def energy(self, x):
        q, p = x.unbind(-1)
        a, _, _ = self.coefficients()
        potential_shape = q * torch.sin(q) + torch.cos(q) - 1.0
        return 0.5 * p.square() + a * potential_shape

    def dissipation(self, x):
        q, p = x.unbind(-1)
        _, r0, r1 = self.coefficients()
        return r0 + r1 * (1.0 - q * torch.sin(q + p / 2.0) * sinc_half(p))

    def forward(self, x, u):
        # Autograd differentiates the *learned Hamiltonian* w.r.t. its states.
        x = x.detach().requires_grad_(True)
        grad_H = torch.autograd.grad(
            self.energy(x).sum(), x, create_graph=self.training
        )[0]
        H_q, H_p = grad_H.unbind(-1)
        r = self.dissipation(x)
        # (J - R) @ grad(H) + G*u, written component-by-component
        return torch.stack((H_p, -H_q - r * H_p + u), dim=-1)


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

    # B) Learn from state, input and known derivatives (an intentionally simple first test).
    x_train = -1.0 + 2.0 * torch.rand(3000, 2)
    u_train = -1.0 + 2.0 * torch.rand(3000)
    xdot_train = true_dynamics(x_train, u_train)

    model = ParametricPH()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.06)
    for epoch in range(401):
        model.train()
        optimizer.zero_grad()
        prediction = model(x_train, u_train)
        loss = torch.mean((prediction - xdot_train).square())
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch {epoch:3d} | derivative MSE: {loss.item():.4e}")

    a, r0, r1 = model.coefficients()
    print(f"Identified: a={a.item():.6f}, r0={r0.item():.6f}, r1={r1.item():.6f}")
    print("Ground truth: a=1, r0=1, r1=1")

    # C) Shared hold-out test: one RK4 step and free trajectory rollout.
    model.eval()
    x_test = -1.0 + 2.0 * torch.rand(500, 2)
    u_test = -1.0 + 2.0 * torch.rand(500)
    true_next = rk4_step(true_dynamics, x_test, u_test)
    ph_next = rk4_step(model, x_test, u_test)
    print(f"Held-out one-step RMSE = {torch.mean((true_next - ph_next).square()).sqrt().item():.3e}")

    x0 = torch.tensor([-0.6, 1.4])  # same test initial state as existing Koopman code
    steps = 1000
    inputs = 0.8 * torch.sin(0.2 * torch.arange(steps, dtype=torch.float64))
    true_states = [x0]
    ph_states = [x0]
    for u in inputs:
        true_states.append(rk4_step(true_dynamics, true_states[-1], u).detach())
        ph_states.append(rk4_step(model, ph_states[-1], u).detach())
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
