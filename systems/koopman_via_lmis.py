# Duffing Oscillator System (as in SiShiAta 2026)
# dot{x1} = x2
# dot{x2} = -delta*x2 - x1*cos(x1 + x2) + u

import time

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdiffeq import odeint

# For reproducibility purposes
torch.manual_seed(156)

# Parameters of simulation
dt = 0.01
T = 20.0
N = int(T / dt)
t = torch.linspace(0, T, N + 1, dtype=torch.float64)
method = "rk4"
options = {"step_size": dt}

# Parameters of the Duffing oscillator
delta = 2.0
x0 = -1.0 + 2.0 * torch.rand(2, dtype=torch.float64)
u0 = -1.0 + 2.0 * torch.rand((), dtype=torch.float64)

# Parameters of the thin plate RBFs
Nrbf = 10
c = -1.0 + 2.0 * torch.rand((Nrbf, 2), dtype=torch.float64)

# Parameters of the training dataset
N_trajectories = 20000
N_steps = 2

# Weight used in the original soft constraint
lambda_soft = 500.0

# Numerical margins for the hard LMIs
LMI_EPS = 1e-6
P_MIN_EIG = 1e-3

PARITY_CHECK_TRAJECTORIES = 5
PARITY_TOL = 1e-12


# Duffing oscillator dynamics
def dynamics(t, x, u):
    x1 = x[..., 0]
    x2 = x[..., 1]

    dx1 = x2
    dx2 = -delta * x2 - x1 * torch.cos(x1 + x2) + u

    return torch.stack((dx1, dx2), dim=-1)


# Simulation of one time step with constant input u
def simulate_step(x, u):
    t_step = torch.tensor([0.0, dt], dtype=torch.float64)
    solution = odeint(
        lambda current_t, current_x: dynamics(current_t, current_x, u),
        x,
        t_step,
        method=method,
        options=options,
    )
    return solution[-1]


# Creation of the Koopman lifted state using thin plate radial basis functions
def create_koopman_state(x, centers, n_rbf):
    phi = []
    for i in range(n_rbf):
        r = torch.norm(x - centers[i])
        phi_i = r**2 * torch.log(r + 1e-6)
        phi.append(phi_i)

    phi = torch.stack(phi)
    return torch.cat((x, phi))


def lift_batch(X_batch):
    diff = X_batch[:, None, :] - c[None, :, :]
    r = torch.norm(diff, dim=2)
    phi = r**2 * torch.log(r + 1e-6)
    return torch.cat((X_batch, phi), dim=1)


# Generation of the training random variables
def draw_gold_training_random_variables():
    initial_states = torch.empty((N_trajectories, 2), dtype=torch.float64)
    inputs = torch.empty((N_trajectories, N_steps), dtype=torch.float64)

    for trajectory in range(N_trajectories):
        initial_states[trajectory] = -1.0 + 2.0 * torch.rand(
            2,
            dtype=torch.float64,
        )
        for step in range(N_steps):
            inputs[trajectory, step] = -1.0 + 2.0 * torch.rand(
                (),
                dtype=torch.float64,
            )

    return initial_states, inputs


# Check that batched integration reproduces the scalar implementation
def verify_batched_odeint_parity(initial_states, inputs):
    n_check = min(PARITY_CHECK_TRAJECTORIES, initial_states.shape[0])

    x_batch = initial_states[:n_check].clone()
    x_scalar = initial_states[:n_check].clone()
    max_error = 0.0

    for step in range(N_steps):
        x_batch = simulate_step(x_batch, inputs[:n_check, step])

        scalar_next = []
        for trajectory in range(n_check):
            scalar_next.append(
                simulate_step(
                    x_scalar[trajectory],
                    inputs[trajectory, step],
                )
            )
        x_scalar = torch.stack(scalar_next)

        max_error = max(
            max_error,
            torch.max(torch.abs(x_batch - x_scalar)).item(),
        )

    print(f"torchdiffeq batch/scalar parity max error: {max_error:.3e}")

    if max_error > PARITY_TOL:
        raise RuntimeError(
            "Batched torchdiffeq does not reproduce the scalar GOLD baseline: "
            f"{max_error:.3e} > {PARITY_TOL:.1e}."
        )


# Training dataset
def generate_training_dataset():
    initial_states, inputs = draw_gold_training_random_variables()
    verify_batched_odeint_parity(initial_states, inputs)

    x = initial_states.clone()

    X = torch.empty((N_trajectories, N_steps, 2), dtype=torch.float64)
    U = torch.empty((N_trajectories, N_steps, 1), dtype=torch.float64)
    X_next = torch.empty_like(X)

    for step in range(N_steps):
        u = inputs[:, step]
        x_next = simulate_step(x, u)

        X[:, step, :] = x
        U[:, step, 0] = u
        X_next[:, step, :] = x_next

        x = x_next

    X = X.reshape(-1, 2)
    U = U.reshape(-1, 1)
    X_next = X_next.reshape(-1, 2)

    Z = lift_batch(X)
    Z_next = lift_batch(X_next)

    return X, U, X_next, Z, Z_next


# Gram matrix used to reduce the CVXPY problem size
def psd_gram(data, normalize=True):
    gram = data.T @ data
    if normalize:
        gram = gram / data.shape[0]
    gram = 0.5 * (gram + gram.T)
    return cp.psd_wrap(gram)


def ls_quadratic_expression(A_var, B_var, Z_np, U_np, Z_next_np, normalize=True):
    n = Z_np.shape[0]
    theta = np.hstack((Z_np, U_np))

    scale = n if normalize else 1.0
    gram_theta = psd_gram(theta, normalize=normalize)
    cross = (theta.T @ Z_next_np) / scale
    constant = np.sum(Z_next_np**2) / scale

    M = cp.hstack((A_var, B_var))

    terms = []
    for output_index in range(Z_next_np.shape[1]):
        row = M[output_index, :]
        terms.append(
            cp.quad_form(row, gram_theta)
            - 2.0 * cross[:, output_index] @ row
        )

    return cp.sum(terms) + constant


# Koopman with least squares regression
def identify_koopman_least_squares(Z, U, Z_next):
    Theta = torch.cat((Z, U), dim=1)
    K = torch.linalg.lstsq(
        Theta,
        Z_next,
    ).solution

    Nz = Z.shape[1]
    A = K[:Nz, :].T
    B = K[Nz:, :].T

    return A, B


# Koopman with the original soft stability constraint
def identify_koopman_soft_gold(Z, U, Z_next):
    Z_np = Z.numpy()
    U_np = U.numpy()
    Z_next_np = Z_next.numpy()

    nz = Z.shape[1]
    I = np.eye(nz)

    A_var = cp.Variable((nz, nz))
    B_var = cp.Variable((nz, 1))
    gamma = cp.Variable()

    residual_sum_squares = ls_quadratic_expression(
        A_var,
        B_var,
        Z_np,
        U_np,
        Z_next_np,
        normalize=False,
    )

    # LMI used to impose the spectral norm bound ||A||_2 <= gamma
    soft_lmi = cp.bmat(
        [
            [gamma * I, A_var],
            [A_var.T, gamma * I],
        ]
    )

    problem = cp.Problem(
        cp.Minimize(
            residual_sum_squares
            + lambda_soft * cp.square(gamma)
        ),
        [soft_lmi >> 0],
    )

    problem.solve(
        solver=cp.SCS,
        verbose=False,
    )

    if A_var.value is None or B_var.value is None or gamma.value is None:
        raise RuntimeError(
            f"Soft GOLD identification failed. Solver status: {problem.status}"
        )

    A = torch.tensor(A_var.value, dtype=torch.float64)
    B = torch.tensor(B_var.value, dtype=torch.float64)

    return A, B, float(gamma.value), problem.status, problem.value


# Koopman with hard stability constraint using P = I
def identify_koopman_p_identity(Z, U, Z_next):
    Z_np = Z.numpy()
    U_np = U.numpy()
    Z_next_np = Z_next.numpy()

    nz = Z.shape[1]
    I = np.eye(nz)

    A_var = cp.Variable((nz, nz))
    B_var = cp.Variable((nz, 1))

    mse_objective = ls_quadratic_expression(
        A_var,
        B_var,
        Z_np,
        U_np,
        Z_next_np,
        normalize=True,
    )

    # Equivalent to A.T @ A - I < 0
    lyapunov_lmi = cp.bmat(
        [
            [I, A_var],
            [A_var.T, I],
        ]
    )

    problem = cp.Problem(
        cp.Minimize(mse_objective),
        [lyapunov_lmi >> LMI_EPS * np.eye(2 * nz)],
    )

    problem.solve(
        solver=cp.SCS,
        verbose=False,
        eps=1e-5,
        max_iters=15000,
        warm_start=True,
    )

    if A_var.value is None or B_var.value is None:
        raise RuntimeError(
            f"P=I identification failed. Solver status: {problem.status}"
        )

    A = torch.tensor(A_var.value, dtype=torch.float64)
    B = torch.tensor(B_var.value, dtype=torch.float64)

    return A, B, problem.status, problem.value


# Koopman with variable Lyapunov matrix P
def identify_koopman_variable_p(Z, U, Z_next):
    Z_np = Z.numpy()
    U_np = U.numpy()
    Z_next_np = Z_next.numpy()

    nz = Z.shape[1]
    I = np.eye(nz)

    # Variable substitution: Y = P A and G = P B
    D = np.hstack((Z_next_np, Z_np, U_np))
    gram_D = psd_gram(D, normalize=True)

    P = cp.Variable((nz, nz), symmetric=True)
    Y = cp.Variable((nz, nz))
    G = cp.Variable((nz, 1))

    C = cp.vstack((P, -Y.T, -G.T))

    transformed_mse = cp.sum(
        [
            cp.quad_form(C[:, output_index], gram_D)
            for output_index in range(nz)
        ]
    )

    # Schur-complement form of A.T @ P @ A - P < 0
    schur_lmi = cp.bmat(
        [
            [P, Y.T],
            [Y, P],
        ]
    )

    constraints = [
        P >> P_MIN_EIG * I,
        cp.trace(P) == nz,
        schur_lmi >> LMI_EPS * np.eye(2 * nz),
    ]

    problem = cp.Problem(
        cp.Minimize(transformed_mse),
        constraints,
    )

    problem.solve(
        solver=cp.SCS,
        verbose=False,
        eps=1e-5,
        max_iters=15000,
        warm_start=True,
    )

    if P.value is None or Y.value is None or G.value is None:
        raise RuntimeError(
            f"Variable-P identification failed. Solver status: {problem.status}"
        )

    A_np = np.linalg.solve(P.value, Y.value)
    B_np = np.linalg.solve(P.value, G.value)

    A = torch.tensor(A_np, dtype=torch.float64)
    B = torch.tensor(B_np, dtype=torch.float64)
    P_torch = torch.tensor(P.value, dtype=torch.float64)

    return A, B, P_torch, problem.status, problem.value


# Analysis functions
def spectral_radius(A):
    return torch.max(torch.abs(torch.linalg.eigvals(A))).item()


def spectral_norm(A):
    return torch.linalg.matrix_norm(A, ord=2).item()


def lyapunov_max_eigenvalue(A, P):
    M = A.T @ P @ A - P
    M = 0.5 * (M + M.T)
    return torch.max(torch.linalg.eigvalsh(M)).item()


def one_step_metrics(A, B, Z, U, Z_next, X_next):
    Z_pred = Z @ A.T + U @ B.T

    lifted_rmse = torch.sqrt(
        torch.mean((Z_pred - Z_next) ** 2)
    ).item()

    state_rmse = torch.sqrt(
        torch.mean((Z_pred[:, :2] - X_next) ** 2)
    ).item()

    return lifted_rmse, state_rmse


def simulate_identified_model(A, B, x0_test, U_test):
    z = create_koopman_state(x0_test, c, Nrbf)
    trajectory = [z]

    for u in U_test:
        z = A @ z + B[:, 0] * u
        trajectory.append(z)

    return torch.stack(trajectory)


def rollout_rmse(Z_model, X_real):
    X_model = Z_model[:, :2]
    return torch.sqrt(
        torch.mean((X_model - X_real) ** 2)
    ).item()


# Test against the real system
def build_test_trajectory():
    T_test = 10.0
    N_test = int(T_test / dt)
    t_test = torch.arange(
        N_test + 1,
        dtype=torch.float64,
    ) * dt

    x0_test = torch.tensor(
        [-0.6, 1.4],
        dtype=torch.float64,
    )

    U_test = []
    for k in range(N_test):
        u = 0.8 * torch.sin(
            torch.tensor(
                0.2 * k,
                dtype=torch.float64,
            )
        )
        U_test.append(u)
    U_test = torch.stack(U_test)

    x_real = x0_test.clone()
    X_real = [x_real]

    for u in U_test:
        x_real = simulate_step(x_real, u)
        X_real.append(x_real)

    return t_test, x0_test, U_test, torch.stack(X_real)


# Comparison table
def print_comparison_table(results):
    print("\n" + "=" * 114)
    print("KOOPMAN IDENTIFICATION - SAME DATA / SAME TEST / FOUR IDENTIFICATION METHODS")
    print("=" * 114)
    print(
        f"{'Model':<31}"
        f"{'rho(A)':>12}"
        f"{'||A||2':>12}"
        f"{'1-step z RMSE':>18}"
        f"{'1-step x RMSE':>18}"
        f"{'rollout x RMSE':>18}"
    )
    print("-" * 114)

    for result in results:
        print(
            f"{result['name']:<31}"
            f"{result['rho']:>12.6f}"
            f"{result['norm2']:>12.6f}"
            f"{result['one_step_z']:>18.6e}"
            f"{result['one_step_x']:>18.6e}"
            f"{result['rollout_x']:>18.6e}"
        )

    print("=" * 114)


# Comparison of the trajectories
def plot_comparison(t_test, X_real, model_trajectories):
    styles = {
        "Koopman LS (GOLD)": {
            "linestyle": "--",
            "linewidth": 2.0,
        },
        "Koopman soft (GOLD)": {
            "linestyle": (0, (6, 2, 1, 2)),
            "linewidth": 2.0,
            "marker": "o",
            "markevery": 100,
            "markersize": 3.5,
        },
        "Koopman P=I": {
            "linestyle": "-.",
            "linewidth": 1.8,
            "marker": "s",
            "markevery": 100,
            "markersize": 3.5,
        },
        "Koopman variable P": {
            "linestyle": ":",
            "linewidth": 2.2,
            "marker": "x",
            "markevery": 100,
            "markersize": 4.0,
        },
    }

    plt.figure(figsize=(13, 5))

    for subplot_index, state_index in enumerate((0, 1), start=1):
        plt.subplot(1, 2, subplot_index)
        plt.plot(
            t_test.numpy(),
            X_real[:, state_index].numpy(),
            linestyle="-",
            linewidth=2.6,
            label="Real system",
        )

        for label, Z_model in model_trajectories:
            plt.plot(
                t_test.numpy(),
                Z_model[:, state_index].numpy(),
                label=label,
                **styles[label],
            )

        plt.title(f"State x{state_index + 1}")
        plt.xlabel("Time [s]")
        plt.ylabel(f"x{state_index + 1}")
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.show()


# Main experiment
def main():
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    print("Generating GOLD Duffing dataset with torchdiffeq RK4...")
    X, U, X_next, Z, Z_next = generate_training_dataset()
    print(f"  done in {time.perf_counter() - stage_start:.3f} s")

    nz = Z.shape[1]
    print(f"Samples: {Z.shape[0]}")
    print(f"Lifted-state dimension: {nz}")

    stage_start = time.perf_counter()
    print("\n[1/4] GOLD Koopman least squares...")
    A_ls, B_ls = identify_koopman_least_squares(Z, U, Z_next)
    print(f"  done in {time.perf_counter() - stage_start:.3f} s")

    stage_start = time.perf_counter()
    print("[2/4] GOLD Koopman soft constraint (lambda=500)...")
    A_soft, B_soft, gamma_soft, status_soft, objective_soft = (
        identify_koopman_soft_gold(Z, U, Z_next)
    )
    print(f"  done in {time.perf_counter() - stage_start:.3f} s")

    stage_start = time.perf_counter()
    print("[3/4] Koopman with hard Lyapunov condition, P = I...")
    A_identity, B_identity, status_identity, objective_identity = (
        identify_koopman_p_identity(Z, U, Z_next)
    )
    print(f"  done in {time.perf_counter() - stage_start:.3f} s")

    stage_start = time.perf_counter()
    print("[4/4] Koopman with variable P + Schur/Lyapunov LMI...")
    A_p, B_p, P_p, status_p, objective_p = identify_koopman_variable_p(
        Z,
        U,
        Z_next,
    )
    print(f"  done in {time.perf_counter() - stage_start:.3f} s")

    t_test, x0_test, U_test, X_real = build_test_trajectory()

    Z_ls = simulate_identified_model(A_ls, B_ls, x0_test, U_test)
    Z_soft = simulate_identified_model(A_soft, B_soft, x0_test, U_test)
    Z_identity = simulate_identified_model(
        A_identity,
        B_identity,
        x0_test,
        U_test,
    )
    Z_p = simulate_identified_model(A_p, B_p, x0_test, U_test)

    models = [
        ("Koopman LS (GOLD)", A_ls, B_ls, Z_ls),
        ("Koopman soft (GOLD)", A_soft, B_soft, Z_soft),
        ("Koopman P=I", A_identity, B_identity, Z_identity),
        ("Koopman variable P", A_p, B_p, Z_p),
    ]

    results = []
    for name, A_model, B_model, Z_rollout in models:
        one_step_z, one_step_x = one_step_metrics(
            A_model,
            B_model,
            Z,
            U,
            Z_next,
            X_next,
        )

        results.append(
            {
                "name": name,
                "rho": spectral_radius(A_model),
                "norm2": spectral_norm(A_model),
                "one_step_z": one_step_z,
                "one_step_x": one_step_x,
                "rollout_x": rollout_rmse(Z_rollout, X_real),
            }
        )

    print_comparison_table(results)

    I_torch = torch.eye(nz, dtype=torch.float64)
    identity_lyapunov_eig = lyapunov_max_eigenvalue(
        A_identity,
        I_torch,
    )
    variable_p_lyapunov_eig = lyapunov_max_eigenvalue(
        A_p,
        P_p,
    )

    p_eigs = torch.linalg.eigvalsh(P_p)
    p_condition_number = (
        torch.max(p_eigs) / torch.min(p_eigs)
    ).item()

    print("\nConstraint / solver verification")
    print("--------------------------------")
    print(f"Soft GOLD solver status:           {status_soft}")
    print(f"Soft GOLD objective:               {objective_soft:.6e}")
    print(f"Soft GOLD gamma:                   {gamma_soft:.6e}")
    print(f"Soft GOLD ||A||2:                  {spectral_norm(A_soft):.6e}")
    print(f"Soft GOLD gamma-||A||2:            {gamma_soft - spectral_norm(A_soft):.6e}")
    print()
    print(f"P=I solver status:                 {status_identity}")
    print(f"P=I optimization objective:        {objective_identity:.6e}")
    print(
        "max eig(A^T A - I):            "
        f"{identity_lyapunov_eig:.6e}"
    )
    print()
    print(f"Variable-P solver status:          {status_p}")
    print(f"Variable-P optimization objective: {objective_p:.6e}")
    print(
        "max eig(A^T P A - P):          "
        f"{variable_p_lyapunov_eig:.6e}"
    )
    print(
        "eig(P) range:                   "
        f"[{torch.min(p_eigs).item():.6e}, "
        f"{torch.max(p_eigs).item():.6e}]"
    )
    print(f"cond(P) from eigenvalues:          {p_condition_number:.6e}")

    print(f"\nTotal runtime: {time.perf_counter() - total_start:.3f} s")

    plot_comparison(
        t_test,
        X_real,
        [
            ("Koopman LS (GOLD)", Z_ls),
            ("Koopman soft (GOLD)", Z_soft),
            ("Koopman P=I", Z_identity),
            ("Koopman variable P", Z_p),
        ],
    )


if __name__ == "__main__":
    main()
