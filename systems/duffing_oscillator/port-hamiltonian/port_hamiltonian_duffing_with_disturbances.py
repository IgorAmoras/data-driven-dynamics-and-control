"""Port-Hamiltonian identification with measurement and process noise.

Companion to port_hamiltonian_duffing.py and SINDy_with_disturbances.py.

Same modified Duffing plant, input, RK4 step, process noise and measurement
noise levels as the repository's noisy SINDy experiment:

    q_dot = p
    p_dot = -2*p - q*cos(q+p) + u + w
    y = p + v

PH-1 is deliberately STRUCTURED: we know the functional form of H and R but
not their three positive coefficients a, r0 and r1. We identify them by
nonnegative least squares, not gradient descent. This is not a neural PH model
and should not be compared as if it had equal prior knowledge to SINDy.

Measurements: only noisy y=p. We reconstruct q by integrating smoothed y,
using the TRUE q(0) supplied to this test. SINDy searches for the hidden q(0),
so its task is harder. This assumption is made explicit to avoid a misleading
head-to-head score.

Process disturbance w is unmeasured, zero-mean, piecewise constant per RK4 step.
Measurement noise is scaled by std(y_true), just as in SINDy_with_disturbances.
We estimate y_dot from noisy measurements, not from the plant's true dynamics.
Validation is on a clean holdout with DIFFERENT initial condition and input.
The table also includes a 0%-sensor-noise diagnostic to separate measurement
noise from process disturbance and smoothing bias.

The PH dissipativity statement holds on |q| <= 1 for positive parameters.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from port_hamiltonian_duffing import (
    ph_features, fit_nonnegative_lstsq, ph_dynamics, ph_dissipation,
    true_dynamics, rk4_step, DT,
)

# 1) Experimental configuration (same reference as noisy SINDy).
torch.set_default_dtype(torch.float64)
torch.manual_seed(156)
T = 10.0
N = int(T / DT)
PROCESS_NOISE_STD = 0.5
NOISE_LEVELS = [0.01, 0.03, 0.05, 0.10]
DIAGNOSTIC_NOISE_LEVEL = 0.0
SAVGOL_WINDOW = 41   # odd; smoothing helps avoid amplifying sensor noise by /DT
SAVGOL_DEGREE = 3


def disturbed_step(x, u, w):
    """Integrate the disturbed *true plant* with constant u,w for one step."""
    return rk4_step(lambda state, uk: true_dynamics(state, uk) +
                    torch.stack((torch.zeros_like(w), w)), x, u)


def measurement_and_derivative(y, dt=DT, window=SAVGOL_WINDOW,
                               degree=SAVGOL_DEGREE):
    """Local polynomial fit: smoothed y and dy/dt without dividing raw noise by dt.

    We build the Savitzky-Golay coefficients ourselves to avoid new dependencies.
    Truncate half a window at both ends when fitting (edge estimates are biased).
    """
    half = window // 2
    offsets = dt * torch.arange(-half, half + 1)
    vandermonde = torch.stack([offsets**k for k in range(degree + 1)], dim=1)
    pseudo_inverse = torch.linalg.pinv(vandermonde)
    padded = F.pad(y.reshape(1, 1, -1), (half, half), mode="replicate")
    y_smooth = F.conv1d(padded, pseudo_inverse[0].reshape(1, 1, -1)).flatten()
    dy_smooth = F.conv1d(padded, pseudo_inverse[1].reshape(1, 1, -1)).flatten()
    return y_smooth, dy_smooth, slice(half, -half)


def reconstruct_position(p_smooth, q0):
    """Use q_dot=p and the KNOWN q0: trapezoidal quadrature of observed velocity."""
    dq = 0.5 * DT * (p_smooth[1:] + p_smooth[:-1])
    return torch.cat((q0.reshape(1), q0 + torch.cumsum(dq, dim=0)))


def fit_ph(x_measured, u, p_dot_estimate, train_range):
    """Estimate a, r0, r1 from reconstructed states and measured acceleration.

    With noisy data, Y = estimated(p_dot) - u and Phi uses estimated(q,p).
    """
    # Approximate input at the instants used by the smoothed derivative.
    u_at_samples = torch.cat((u[:1], 0.5 * (u[:-1] + u[1:]), u[-1:]))
    features = ph_features(x_measured[train_range])
    target = p_dot_estimate[train_range] - u_at_samples[train_range]
    return fit_nonnegative_lstsq(features, target)


def clean_holdout_metrics(coefficients, inputs):
    """Noiseless out-of-sample test with a different input and initial state."""
    x0 = torch.tensor([0.4, -1.0])
    x_true = [x0]
    x_ph = [x0]
    for uk in inputs:
        x_true.append(rk4_step(true_dynamics, x_true[-1], uk).detach())
        x_ph.append(rk4_step(lambda x, u: ph_dynamics(x, u, coefficients), x_ph[-1], uk).detach())
    true_states = torch.stack(x_true)
    predicted_states = torch.stack(x_ph)
    rollout_rmse = (true_states - predicted_states).square().mean().sqrt().item()
    next_true = torch.stack([
        rk4_step(true_dynamics, true_states[k], inputs[k]) for k in range(N)
    ])
    next_predicted = rk4_step(lambda x, u: ph_dynamics(x, u, coefficients), true_states[:-1], inputs)
    one_step_rmse = (next_true - next_predicted).square().mean().sqrt().item()
    return one_step_rmse, rollout_rmse, true_states, predicted_states


def main():
    # Exact field match is established and tested in the companion intro script.
    # 2) Generate one shared disturbed realization, exactly as noisy SINDy does.
    x0 = torch.tensor([-0.6, 1.4])
    u = 0.8 * torch.sin(0.2 * torch.arange(N))
    w = PROCESS_NOISE_STD * torch.randn(N)
    states = [x0]
    for uk, wk in zip(u, w):
        states.append(disturbed_step(states[-1], uk, wk).detach())
    states = torch.stack(states)
    p_true = states[:, 1]

    # 3) Noise is relative to the standard deviation of the measured output.
    # These four trials share the same true plant, process noise and control.
    u_test = 0.65 * torch.sin(0.17 * torch.arange(N) + 0.4)
    results = []
    for noise in [DIAGNOSTIC_NOISE_LEVEL] + NOISE_LEVELS:
        y = p_true + noise * p_true.std() * torch.randn_like(p_true)
        p_smooth, p_dot_est, valid_range = measurement_and_derivative(y)
        q_est = reconstruct_position(p_smooth, x0[0])
        x_est = torch.stack((q_est, p_smooth), dim=-1)
        coefficients, train_mse = fit_ph(x_est, u, p_dot_est, valid_range)
        a, r0, r1 = [v.item() for v in coefficients]
        one_step, rollout, true_clean, ph_clean = clean_holdout_metrics(coefficients, u_test)
        q_rmse = (q_est - states[:, 0]).square().mean().sqrt().item()
        r_min = ph_dissipation(x_est, coefficients).min().item()
        results.append({
            "noise": noise, "a": a, "r0": r0, "r1": r1,
            "train_mse": train_mse, "q_rmse": q_rmse,
            "one_step": one_step, "rollout": rollout, "min_r": r_min,
            "states": ph_clean,
        })

    # 4) Interpretability is part of the experiment: print parameters and errors.
    print("\nDUFFING - PORT-HAMILTONIAN WITH DISTURBANCES")
    print("True drift: q_dot=p, p_dot=-2p-q*cos(q+p)+u")
    print("Sensor: y=p+v; process w ~ N(0, 0.5^2) each time step")
    print("True parameters: a=r0=r1=1; true q(0) provided to identifier")
    print("Measurements smoothed by local polynomial window=41, degree=3")
    print("PH coefficients estimated by nonnegative least squares")
    print("q RMSE and train MSE use the noisy TRAINING trajectory.")
    print("One-step/rollout RMSE use a clean HOLDOUT: x0=(0.4,-1.0), different input.")
    print("=" * 109)
    print("Noise       a       r0       r1   q RMSE    train MSE   one-step RMSE   rollout RMSE    min R")
    for r in results:
        print(f"{100*r['noise']:>4.0f}%   {r['a']:>7.4f} {r['r0']:>8.4f} {r['r1']:>8.4f}"
              f" {r['q_rmse']:>9.5f} {r['train_mse']:>12.5e}"
              f" {r['one_step']:>15.6e} {r['rollout']:>14.6e} {r['min_r']:>8.4f}")
    print("=" * 109)
    print("0% row = diagnostic with process noise but no sensor noise (not a requested noise level).")
    print(f"Maximum |q| in clean holdout: {true_clean[:, 0].abs().max().item():.4f}")
    print(f"Maximum |q| in disturbed training trajectory: {states[:, 0].abs().max().item():.4f}")
    print("PH condition R>=0 is only claimed on |q|<=1. Measurements must be in that domain.")

    # 5) Overlay trajectories for all noise levels, using the SAME holdout case.
    times = DT * torch.arange(N + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for j, label in enumerate(("q", "p")):
        axes[j].plot(times, true_clean[:, j], color="black", label="True (clean)", lw=2)
        for r in results:
            if r["noise"] == DIAGNOSTIC_NOISE_LEVEL:
                continue
            axes[j].plot(times, r["states"][:, j], ls="--",
                         label=f"PH learned at {100*r['noise']:.0f}% noise")
        axes[j].set(xlabel="Time [s]", ylabel=label, title=f"Duffing - {label}")
        axes[j].grid()
        axes[j].legend(fontsize=8)
    fig.tight_layout()
    from pathlib import Path
    output = Path(__file__).resolve().parent / "duffing_ph_noise_rollouts.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {output}")


if __name__ == "__main__":
    main()
