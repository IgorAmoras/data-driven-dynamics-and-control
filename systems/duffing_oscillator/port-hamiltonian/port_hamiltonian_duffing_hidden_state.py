"""Duffing — Port-Hamiltonian identification with UNKNOWN initial position.

Plan A (grid search), analogous to SINDy_with_disturbances.py:
1) Measure only y = p + sensor noise.
2) For each candidate c = q(0), reconstruct q(t) = c + integral(p dt).
3) Identify positive PH coefficients (a, r0, r1) by least squares.
4) Pick c and coefficients with the smallest acceleration fitting error.

The TRUE q(0) is used only to GENERATE the plant and CHECK the result.
The PH model preserves J canonical and R >= 0 on |q| <= 1.
"""

import matplotlib.pyplot as plt
import torch

from port_hamiltonian_duffing import (
    true_dynamics, rk4_step, DT,
    ph_features, fit_nonnegative_lstsq, ph_dynamics,
)
from port_hamiltonian_duffing_with_disturbances import (
    N, NOISE_LEVELS, PROCESS_NOISE_STD,
    disturbed_step, measurement_and_derivative, reconstruct_position,
)

torch.set_default_dtype(torch.float64)
torch.manual_seed(156)

# Same process/measurement noise as SINDy_with_disturbances.py.
# Search a priori in [-1, 1], our chosen PH dissipativity domain.
CANDIDATES = torch.linspace(-1.0, 1.0, 101)


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

        # The same PH regression as the other experiments:
        # Y = estimated(dp/dt) - u = Phi(q,p) @ [a, r0, r1].
        features = ph_features(torch.stack((q, p), dim=-1))
        target = p_dot - u_samples
        coefficients, loss = fit_nonnegative_lstsq(features[valid], target[valid])
        curve.append((c.item(), loss))

        if best is None or loss < best["loss"]:
            best = {"q0": c.item(), "coefficients": coefficients, "loss": loss,
                    "q": q, "p": p}

    if best is None:
        raise RuntimeError("No candidates remain inside |q| <= 1; expand the PH model/domain.")
    best["curve"] = curve
    return best


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

    # 4) Plot both states, including the identified PH parameters in the legend.
    # Important: q is reconstructed by integration and p is a smoothed measurement.
    # These are NOT free trajectories generated by the identified PH dynamics.
    from pathlib import Path
    times = DT * torch.arange(N + 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].plot(times, states[:, 0], "k", linewidth=2, label="True q")
    axes[1].plot(times, states[:, 1], "k", linewidth=2, label="True p")

    legend_handles = [axes[0].lines[0]]
    legend_labels = ["True state (evaluation only)"]
    for r in results[1:]:  # Omit the 0% diagnostic from the figures.
        a, r0, r1 = [value.item() for value in r["coefficients"]]
        label = (f"PH {100*r['noise']:.0f}% | q0={r['q0']:.2f}, "
                 f"a={a:.3f}, r0={r0:.3f}, r1={r1:.3f}")
        line_q, = axes[0].plot(times, r["q"], "--")
        axes[1].plot(times, r["p"], "--", color=line_q.get_color())
        legend_handles.append(line_q)
        legend_labels.append(label)

    axes[0].set(title="Hidden position reconstructed from p", xlabel="Time [s]", ylabel="q")
    axes[1].set(title="Measured velocity after smoothing", xlabel="Time [s]", ylabel="p")
    for ax in axes:
        ax.grid()

    # Shared legend stays below the graphs instead of obscuring the trajectories.
    fig.legend(legend_handles, legend_labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.025), ncol=2, fontsize=9)
    fig.subplots_adjust(bottom=0.27, wspace=0.23)
    output_qp = Path(__file__).resolve().parent / "duffing_ph_hidden_state_qp.png"
    fig.savefig(output_qp, dpi=150)
    plt.close(fig)
    print(f"State reconstruction figure: {output_qp}")

    # 5) Keep the objective profile as a separate figure: why was q(0) selected?
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results[1:]:
        candidates, losses = zip(*r["curve"])
        ax.plot(candidates, losses, label=f"PH {100*r['noise']:.0f}%")
    ax.axvline(true_x0[0].item(), color="black", linestyle=":",
               label="True q(0) (evaluation only)")
    ax.set(title="PH fit error versus candidate q(0)",
           xlabel="Candidate q(0)", ylabel="Acceleration MSE")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    output_objective = Path(__file__).resolve().parent / "duffing_ph_hidden_state_objective.png"
    fig.savefig(output_objective, dpi=150)
    plt.close(fig)
    print(f"Objective figure: {output_objective}")
    print("The q/p curves are reconstructed states, not PH free-rollouts.")


if __name__ == "__main__":
    main()
