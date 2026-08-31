# Duffing oscillator - Koopman identification with hard stability constraints via LMIs
#
# Comparison implemented in this experimental script:
#   1) Standard Koopman identification by least squares (unconstrained baseline)
#   2) Koopman identification with the classical discrete-time Lyapunov condition
#      using P = I
#   3) Koopman identification with variable P, using the change of variables
#         Y = P A,  G = P B
#      and the Schur-complement LMI
#
# PERFORMANCE NOTE
# ----------------
# The original version sent all 40,000 sample residuals directly to CVXPY and
# generated the training trajectories with 40,000 individual ODE solver calls.
# That is unnecessary here. This version:
#   - uses the same fixed-step RK4 rule in vectorized form;
#   - vectorizes the RBF lifting;
#   - compresses the least-squares objectives into Gram matrices before CVXPY.
#
# Therefore the SDP size depends essentially on the lifted-state dimension, not
# on the number of training samples. No training samples are discarded.

import time

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Reproducibility and system parameters
# -----------------------------------------------------------------------------

torch.manual_seed(156)

dt = 0.01
delta = 2.0

# Keep these draws to preserve the random-number sequence of the original
# duffing_oscillator.py before the RBF centers are generated.
x0 = -1.0 + 2.0 * torch.rand(2, dtype=torch.float64)
u0 = -1.0 + 2.0 * torch.rand((), dtype=torch.float64)

Nrbf = 10
c = -1.0 + 2.0 * torch.rand((Nrbf, 2), dtype=torch.float64)

N_trajectories = 20000
N_steps = 2

# Numerical margins for strict LMIs.
LMI_EPS = 1e-6
P_MIN_EIG = 1e-3


# -----------------------------------------------------------------------------
# Duffing dynamics and fast fixed-step RK4
# -----------------------------------------------------------------------------

def dynamics(x, u):
    """Duffing dynamics for either one state (..., 2) or a batch of states."""
    x1 = x[..., 0]
    x2 = x[..., 1]

    dx1 = x2
    dx2 = -delta * x2 - x1 * torch.cos(x1 + x2) + u

    return torch.stack((dx1, dx2), dim=-1)


def simulate_step(x, u):
    """
    One fixed RK4 step of length dt.

    This is the same numerical integration rule used in the original script
    with torchdiffeq's method='rk4' and step_size=dt, but without the overhead
    of constructing an ODE solver for every single training sample.
    """
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# -----------------------------------------------------------------------------
# Koopman lifting
# -----------------------------------------------------------------------------

def create_koopman_state(x, centers, n_rbf):
    """Lift a single state."""
    diff = x.unsqueeze(0) - centers[:n_rbf]
    r = torch.linalg.vector_norm(diff, dim=1)
    phi = r**2 * torch.log(r + 1e-6)
    return torch.cat((x, phi))


def lift_batch(X_batch):
    """Vectorized thin-plate RBF lifting for all samples at once."""
    diff = X_batch[:, None, :] - c[None, :, :]
    r = torch.linalg.vector_norm(diff, dim=2)
    phi = r**2 * torch.log(r + 1e-6)
    return torch.cat((X_batch, phi), dim=1)


# -----------------------------------------------------------------------------
# Dataset generation
# -----------------------------------------------------------------------------

def generate_training_dataset():
    """
    Generate all Duffing trajectories in batch.

    A (N_trajectories, 2 + N_steps) random tensor is used so the flat random
    draw order remains x1, x2, u0, u1 for each trajectory, matching the logical
    order of the original nested loop.
    """
    random_draws = -1.0 + 2.0 * torch.rand(
        (N_trajectories, 2 + N_steps),
        dtype=torch.float64,
    )

    x = random_draws[:, :2].clone()
    u_schedule = random_draws[:, 2:]

    X = torch.empty(
        (N_trajectories, N_steps, 2),
        dtype=torch.float64,
    )
    U = torch.empty(
        (N_trajectories, N_steps, 1),
        dtype=torch.float64,
    )
    X_next = torch.empty_like(X)

    for step in range(N_steps):
        u = u_schedule[:, step]
        x_next = simulate_step(x, u)

        X[:, step, :] = x
        U[:, step, 0] = u
        X_next[:, step, :] = x_next

        x = x_next

    # Preserve the original ordering: trajectory 0 step 0, trajectory 0 step 1,
    # trajectory 1 step 0, trajectory 1 step 1, ...
    X = X.reshape(-1, 2)
    U = U.reshape(-1, 1)
    X_next = X_next.reshape(-1, 2)

    Z = lift_batch(X)
    Z_next = lift_batch(X_next)

    return X, U, X_next, Z, Z_next


# -----------------------------------------------------------------------------
# Small quadratic objectives from sufficient statistics
# -----------------------------------------------------------------------------

def gram_matrix(data):
    """Return E[data^T data] and mark it PSD for CVXPY."""
    n = data.shape[0]
    gram = (data.T @ data) / n
    # Numerical symmetrization avoids tiny floating-point asymmetries.
    gram = 0.5 * (gram + gram.T)
    return cp.psd_wrap(gram)


# -----------------------------------------------------------------------------
# Identification methods
# -----------------------------------------------------------------------------

def identify_koopman_least_squares(Z, U, Z_next):
    """Unconstrained Koopman baseline."""
    theta = torch.cat((Z, U), dim=1)
    K = torch.linalg.lstsq(theta, Z_next).solution

    nz = Z.shape[1]
    A = K[:nz, :].T
    B = K[nz:, :].T
    return A, B


def identify_koopman_p_identity(Z, U, Z_next):
    """
    Stable identification with the classical Lyapunov matrix fixed to P = I.

    A^T A - I < 0

    is equivalent, by the Schur complement, to

        [ I   A^T ] > 0.
        [ A    I  ]

    The least-squares objective is written from Gram matrices, so CVXPY never
    sees the 40,000-row residual matrix.
    """
    Z_np = Z.numpy()
    U_np = U.numpy()
    Z_next_np = Z_next.numpy()

    n = Z_np.shape[0]
    nz = Z.shape[1]
    I = np.eye(nz)

    theta = np.hstack((Z_np, U_np))
    gram_theta = gram_matrix(theta)
    cross = (theta.T @ Z_next_np) / n
    constant = np.sum(Z_next_np**2) / n

    A_var = cp.Variable((nz, nz))
    B_var = cp.Variable((nz, 1))
    M = cp.hstack((A_var, B_var))

    quadratic_terms = []
    for output_index in range(nz):
        row = M[output_index, :]
        quadratic_terms.append(
            cp.quad_form(row, gram_theta)
            - 2.0 * cross[:, output_index] @ row
        )

    mse_objective = cp.sum(quadratic_terms) + constant

    lyapunov_lmi = cp.bmat(
        [
            [I, A_var.T],
            [A_var, I],
        ]
    )

    constraints = [
        lyapunov_lmi >> LMI_EPS * np.eye(2 * nz),
    ]

    problem = cp.Problem(
        cp.Minimize(mse_objective),
        constraints,
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


def identify_koopman_variable_p(Z, U, Z_next):
    r"""
    Stable Koopman identification with a variable Lyapunov matrix P.

    Start from

        A^T P A - P < 0,    P > 0.

    and define

        Y = P A,
        G = P B.

    The stability condition becomes the LMI

        [ P   Y^T ] > 0.
        [ Y    P  ]

    while the identified dynamics are left-multiplied by P:

        P z(k+1) ~= Y z(k) + G u(k).

    Thus the transformed residual is

        Z_next P - Z Y^T - U G^T.

    Its full 40,000-sample Frobenius norm is represented exactly by a small Gram
    matrix. No subsampling is performed.

    trace(P) = nz removes the arbitrary scaling of (P, Y, G).
    """
    Z_np = Z.numpy()
    U_np = U.numpy()
    Z_next_np = Z_next.numpy()

    nz = Z.shape[1]
    I = np.eye(nz)

    # If
    #   D = [Z_next, Z, U]
    # and
    #   C = [P; -Y^T; -G^T],
    # then the transformed residual is simply D C.
    D = np.hstack((Z_next_np, Z_np, U_np))
    gram_D = gram_matrix(D)

    P = cp.Variable((nz, nz), symmetric=True)
    Y = cp.Variable((nz, nz))
    G = cp.Variable((nz, 1))

    C = cp.vstack((P, -Y.T, -G.T))

    quadratic_terms = [
        cp.quad_form(C[:, output_index], gram_D)
        for output_index in range(nz)
    ]
    transformed_mse = cp.sum(quadratic_terms)

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

    # Better conditioned than explicitly forming inv(P).
    A_np = np.linalg.solve(P.value, Y.value)
    B_np = np.linalg.solve(P.value, G.value)

    A = torch.tensor(A_np, dtype=torch.float64)
    B = torch.tensor(B_np, dtype=torch.float64)
    P_torch = torch.tensor(P.value, dtype=torch.float64)

    return A, B, P_torch, problem.status, problem.value


# -----------------------------------------------------------------------------
# Metrics and verification
# -----------------------------------------------------------------------------

def spectral_radius(A):
    return torch.max(torch.abs(torch.linalg.eigvals(A))).item()


def spectral_norm(A):
    return torch.linalg.matrix_norm(A, ord=2).item()


def lyapunov_max_eigenvalue(A, P):
    """Maximum eigenvalue of A^T P A - P; negative means strict stability."""
    M = A.T @ P @ A - P
    M = 0.5 * (M + M.T)
    return torch.max(torch.linalg.eigvalsh(M)).item()


def one_step_metrics(A, B, Z, U, Z_next, X_next):
    Z_pred = Z @ A.T + U @ B.T
    lifted_rmse = torch.sqrt(torch.mean((Z_pred - Z_next) ** 2)).item()
    state_rmse = torch.sqrt(torch.mean((Z_pred[:, :2] - X_next) ** 2)).item()
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
    return torch.sqrt(torch.mean((X_model - X_real) ** 2)).item()


# -----------------------------------------------------------------------------
# Test trajectory and reporting
# -----------------------------------------------------------------------------

def build_test_trajectory():
    T_test = 10.0
    N_test = int(T_test / dt)
    t_test = torch.arange(N_test + 1, dtype=torch.float64) * dt

    x0_test = torch.tensor([-0.6, 1.4], dtype=torch.float64)

    k = torch.arange(N_test, dtype=torch.float64)
    U_test = 0.8 * torch.sin(0.2 * k)

    x_real = x0_test.clone()
    X_real = [x_real]

    for u in U_test:
        x_real = simulate_step(x_real, u)
        X_real.append(x_real)

    return t_test, x0_test, U_test, torch.stack(X_real)


def print_comparison_table(results):
    print("\n" + "=" * 108)
    print("KOOPMAN IDENTIFICATION - STABILITY COMPARISON")
    print("=" * 108)
    print(
        f"{'Model':<28}"
        f"{'rho(A)':>12}"
        f"{'||A||2':>12}"
        f"{'1-step z RMSE':>18}"
        f"{'1-step x RMSE':>18}"
        f"{'rollout x RMSE':>18}"
    )
    print("-" * 108)

    for result in results:
        print(
            f"{result['name']:<28}"
            f"{result['rho']:>12.6f}"
            f"{result['norm2']:>12.6f}"
            f"{result['one_step_z']:>18.6e}"
            f"{result['one_step_x']:>18.6e}"
            f"{result['rollout_x']:>18.6e}"
        )

    print("=" * 108)


def plot_comparison(t_test, X_real, model_trajectories):
    plt.figure(figsize=(13, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_test.numpy(), X_real[:, 0].numpy(), label="Real system")
    for label, Z_model in model_trajectories:
        plt.plot(t_test.numpy(), Z_model[:, 0].numpy(), "--", label=label)
    plt.title("State x1")
    plt.xlabel("Time [s]")
    plt.ylabel("x1")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_test.numpy(), X_real[:, 1].numpy(), label="Real system")
    for label, Z_model in model_trajectories:
        plt.plot(t_test.numpy(), Z_model[:, 1].numpy(), "--", label=label)
    plt.title("State x2")
    plt.xlabel("Time [s]")
    plt.ylabel("x2")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def main():
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    print("Generating Duffing dataset + vectorized lifting...")
    X, U, X_next, Z, Z_next = generate_training_dataset()
    print(f"  done in {time.perf_counter() - stage_start:.3f} s")

    nz = Z.shape[1]
    print(f"Samples: {Z.shape[0]}")
    print(f"Lifted-state dimension: {nz}")

    stage_start = time.perf_counter()
    print("\n[1/3] Standard Koopman least squares...")
    A_ls, B_ls = identify_koopman_least_squares(Z, U, Z_next)
    print(f"  done in {time.perf_counter() - stage_start:.3f} s")

    stage_start = time.perf_counter()
    print("[2/3] Koopman with classical Lyapunov condition, P = I...")
    A_identity, B_identity, status_identity, objective_identity = (
        identify_koopman_p_identity(Z, U, Z_next)
    )
    print(f"  done in {time.perf_counter() - stage_start:.3f} s")

    stage_start = time.perf_counter()
    print("[3/3] Koopman with variable P + change of variables + Schur LMI...")
    A_p, B_p, P_p, status_p, objective_p = identify_koopman_variable_p(
        Z, U, Z_next
    )
    print(f"  done in {time.perf_counter() - stage_start:.3f} s")

    t_test, x0_test, U_test, X_real = build_test_trajectory()

    Z_ls = simulate_identified_model(A_ls, B_ls, x0_test, U_test)
    Z_identity = simulate_identified_model(
        A_identity,
        B_identity,
        x0_test,
        U_test,
    )
    Z_p = simulate_identified_model(A_p, B_p, x0_test, U_test)

    models = [
        ("Koopman LS", A_ls, B_ls, Z_ls),
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

    print("\nLMI verification")
    print("----------------")
    print(f"P=I solver status:                 {status_identity}")
    print(f"P=I optimization objective:        {objective_identity:.6e}")
    print(
        "max eig(A^T A - I):            "
        f"{identity_lyapunov_eig:.6e}  "
        "(< 0 means the P=I Lyapunov inequality is satisfied)"
    )
    print()
    print(f"Variable-P solver status:          {status_p}")
    print(f"Variable-P optimization objective: {objective_p:.6e}")
    print(
        "max eig(A^T P A - P):          "
        f"{variable_p_lyapunov_eig:.6e}  "
        "(< 0 means the Lyapunov inequality is satisfied)"
    )
    print(
        "eig(P) range:                   "
        f"[{torch.min(torch.linalg.eigvalsh(P_p)).item():.6e}, "
        f"{torch.max(torch.linalg.eigvalsh(P_p)).item():.6e}]"
    )

    print(f"\nTotal runtime: {time.perf_counter() - total_start:.3f} s")

    plot_comparison(
        t_test,
        X_real,
        [
            ("Koopman LS", Z_ls),
            ("Koopman P=I", Z_identity),
            ("Koopman variable P", Z_p),
        ],
    )


if __name__ == "__main__":
    main()
