"""Duffing — Port-Hamiltonian identification with UNKNOWN initial position.

Plan A (grid search), analogous to SINDy_with_disturbances.py:
1) Measure only y = p + sensor noise.
2) For each candidate c = q(0), reconstruct q(t) = c + integral(p dt).
3) Identify positive PH coefficients (a, r0, r1) for this candidate.
4) Pick c and coefficients with the smallest acceleration fitting error.

The TRUE q(0) is used only to GENERATE the plant and CHECK the result.
The PH model preserves J canonical and R >= 0 on |q| <= 1.
"""

import matplotlib.pyplot as plt
import torch

from port_hamiltonian_duffing import true_dynamics, rk4_step, sinc_half, DT
from port_hamiltonian_duffing_with_disturbances import (
    N, NOISE_LEVELS, PROCESS_NOISE_STD,
    disturbed_step, measurement_and_derivative, reconstruct_position,
)

torch.set_default_dtype(torch.float64)
torch.manual_seed(156)

# Same process/measurement noise as SINDy_with_disturbances.py.
# Search a priori in [-1, 1], our chosen PH dissipativity domain.
CANDIDATES = torch.linspace(-1.0, 1.0, 101)


def fit_positive_coefficients(features, target):
    """Solve min ||features @ [a,r0,r1] - target||^2, with a,r0,r1 >= 0.

    Only 3 coefficients: test all 8 active subsets (including all zeros).
    This is nonnegative least squares without adding a dependency.
    """
    best_loss = torch.mean(target.square()).item()
    best_coefficients = torch.zeros(3)

    for mask in range(1, 8):
        active = [j for j in range(3) if mask & (1 << j)]
        solution = torch.linalg.lstsq(features[:, active], target).solution
        if torch.any(solution < -1e-10):
            continue  # Negative coefficients are not allowed.

        coefficients = torch.zeros(3)
        coefficients[active] = solution.clamp(min=0.0)
        loss = torch.mean((features @ coefficients - target).square()).item()
        if loss < best_loss:
            best_loss, best_coefficients = loss, coefficients

    return best_coefficients, best_loss


def identify_hidden_position(y, u):
    """Estimate q(0), a, r0 and r1 using ONLY measured y and known input u."""
    p, p_dot, valid = measurement_and_derivative(y)
    # Input is constant on each RK4 interval. Average adjacent intervals to
    # approximate the input at the instants of smoothed derivative estimates.
    u_samples = torch.cat((u[:1], (u[:-1] + u[1:]) / 2, u[-1:]))

    best = None
    curve = []  # Store the loss versus each candidate q(0).

    for c in CANDIDATES:
        q = reconstruct_position(p, c)

        # The learned PH dissipation is guaranteed nonnegative on |q| <= 1.
        # Do not accept a candidate outside the assumed admissible domain.
        if q.abs().max().item() > 1.0:
            continue

        D = q * torch.sin(q + p / 2) * sinc_half(p)

        # PH equation: p_dot = -a*q*cos(q) - r0*p - r1*(1-D)*p + u.
        # Thus p_dot - u = FEATURES @ [a, r0, r1].
        features = torch.stack((-q * torch.cos(q), -p, -(1 - D) * p), dim=1)
        coefficients, loss = fit_positive_coefficients(
            features[valid], (p_dot - u_samples)[valid]
        )
        curve.append((c.item(), loss))

        if best is None or loss < best["loss"]:
            best = {"q0": c.item(), "coefficients": coefficients, "loss": loss,
                    "q": q, "p": p}

    if best is None:
        raise RuntimeError("No candidates remain inside |q| <= 1; expand the PH model/domain.")
    best["curve"] = curve
    return best


def ph_dynamics(x, u, coefficients):
    """The SAME parametrized PH equation, evaluated with identified coefficients."""
    a, r0, r1 = coefficients
    q, p = x.unbind(-1)
    D = q * torch.sin(q + p / 2) * sinc_half(p)
    r = r0 + r1 * (1 - D)
    return torch.stack((p, -a * q * torch.cos(q) - r * p + u), dim=-1)


def clean_rollout_rmse(coefficients, inputs):
    """Compare identified PH with the plant on a NEW, unperturbed trajectory."""
    initial_state = torch.tensor([0.4, -1.0])  # Validation only, not used to fit.
    truth = initial_state.clone()
    estimate = initial_state.clone()
    true_states, estimated_states = [truth], [estimate]
    for uk in inputs:
        truth = rk4_step(true_dynamics, truth, uk)
        estimate = rk4_step(lambda x, u: ph_dynamics(x, u, coefficients), estimate, uk)
        true_states.append(truth)
        estimated_states.append(estimate)
    true_states = torch.stack(true_states)
    estimated_states = torch.stack(estimated_states)
    rmse = torch.mean((true_states - estimated_states).square()).sqrt().item()
    return rmse, true_states, estimated_states


def main():
    # 1) Generate the disturbed plant, using the hidden q(0) ONLY here.
    true_x0 = torch.tensor([-0.6, 1.4])
    u = 0.8 * torch.sin(0.2 * torch.arange(N))
    w = PROCESS_NOISE_STD * torch.randn(N)
    states = [true_x0]
    for uk, wk in zip(u, w):
        states.append(disturbed_step(states[-1], uk, wk))
    states = torch.stack(states)
    p_true = states[:, 1]

    # 2) Same four noise levels as the previous PH and SINDy experiments.
    test_inputs = 0.65 * torch.sin(0.17 * torch.arange(N) + 0.4)
    results = []
    for noise in [0.0] + NOISE_LEVELS:
        y = p_true + noise * p_true.std() * torch.randn_like(p_true)
        result = identify_hidden_position(y, u)  # No true q0 passed!
        result["noise"] = noise
        result["q_rmse"] = (result["q"] - states[:, 0]).square().mean().sqrt().item()
        result["rollout_rmse"], result["true_test"], result["ph_test"] = (
            clean_rollout_rmse(result["coefficients"], test_inputs)
        )
        results.append(result)

    # 3) Print both hidden-state error and model-prediction error.
    print("\nDUFFING — PH WITH UNKNOWN INITIAL POSITION (PLAN A)")
    print("Only y=p+v and u are passed to the identifier.")
    print("True q(0)=-0.6 is used ONLY for evaluation; reference a=r0=r1=1.")
    print("0% sensor noise is a diagnostic; process noise remains active.")
    print("PH positivity is enforced on |q|<=1; q0 candidate spacing is 0.02.")
    print("=" * 102)
    print("Noise    q0_hat    |q0 error|     a      r0      r1     q RMSE    loss      rollout RMSE")
    for r in results:
        a, r0, r1 = [v.item() for v in r["coefficients"]]
        print(f"{100*r['noise']:>4.0f}%   {r['q0']:>8.3f}   {abs(r['q0']-true_x0[0].item()):>9.4f}"
              f"   {a:>6.3f}  {r0:>6.3f}  {r1:>6.3f}"
              f"   {r['q_rmse']:>8.4f}  {r['loss']:>8.5f}  {r['rollout_rmse']:>12.5f}")
    print("=" * 102)
    print("NOTE: Good q(0) estimation does not imply exact recovery of a,r0,r1.")
    print("NOTE: This is OFFLINE initial-state identification, not a dynamic observer.")

    # 4) Plot reconstructed hidden state and the objective profile versus q(0).
    times = DT * torch.arange(N + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(times, states[:, 0], "k", linewidth=2, label="True hidden q")
    for r in results[1:]:  # omit 0% diagnostic from graph
        axes[0].plot(times, r["q"], "--", label=f"PH {100*r['noise']:.0f}%")
        candidates, losses = zip(*r["curve"])
        axes[1].plot(candidates, losses, label=f"{100*r['noise']:.0f}%")
    axes[0].set(title="Hidden position reconstruction", xlabel="Time [s]", ylabel="q")
    axes[1].axvline(true_x0[0].item(), color="black", linestyle=":", label="True q(0) — evaluation only")
    axes[1].set(title="PH fit error versus candidate q(0)", xlabel="Candidate q(0)", ylabel="Acceleration MSE")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid()
    fig.tight_layout()
    from pathlib import Path
    output = Path(__file__).resolve().parent / "duffing_ph_hidden_state.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {output}")


if __name__ == "__main__":
    main()
