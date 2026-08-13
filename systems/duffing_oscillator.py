# Duffing Oscillator System (as in SiShiAta 2026)
# dot{x1} = x2
# dot{x2} = -delta*x2 - x1*cos(x1 + x2) + u

import torch
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# For reproducibility purposes
torch.manual_seed(156)

# Parameters of simulation
dt = 0.01 # Sampling time
T = 20.0 # Total simulation time
N = int(T/dt) # Number of time steps
t = torch.linspace(0, T, N + 1, dtype=torch.float64) # Time vector
method = 'rk4' # Integration method
options = {'step_size': dt} # Integration options

# Parameters of the Duffing oscillator
delta = 2.0 # Damping coefficient
x0 = -1.0 + 2.0 * torch.rand(2, dtype=torch.float64)
u0 = -1.0 + 2.0 * torch.rand((), dtype=torch.float64)

# Parameters of the thin plate, using Nrbf = 8, as in SiShiAta 2026
Nrbf = 10 # Number of radial basis functions -> although the article states 8 NrBF, the code uses 10
c = -1.0 + 2.0 * torch.rand((Nrbf, 2), dtype=torch.float64) # RBF centers in [-1, 1]^2

# Parameters of the training dataset
N_trajectories = 20000 # Number of independent trajectories
N_steps = 2 # Number of simulation steps per trajectory

# Duffing oscillator dynamics
def dynamics(t, x, u):
    x1, x2 = x
    dx1 = x2
    dx2 = -delta*x2 - x1*torch.cos(x1 + x2) + u
    return torch.stack([dx1, dx2])

# Simulation of one time step with constant input u
def simulate_step(x, u):
    t_step = torch.tensor([0.0, dt], dtype=torch.float64)
    solution = odeint(
        lambda t, x: dynamics(t, x, u),
        x,
        t_step,
        method=method,
        options=options
    )
    x_next = solution[-1]
    return x_next


# Training dataset
X = [] # Current states x(k)
U = [] # Inputs u(k)
X_next = [] # Next states x(k+1)


# Generation of the training dataset
for trajectory in range(N_trajectories):

    x = -1.0 + 2.0 * torch.rand(
        2,
        dtype=torch.float64
    ) # Random initial state in [-1, 1]^2
    for step in range(N_steps):
        u = -1.0 + 2.0 * torch.rand(
            (),
            dtype=torch.float64
        ) # Random input in [-1, 1]
        x_next = simulate_step(x, u)
        X.append(x)
        U.append(u)
        X_next.append(x_next)
        x = x_next

# Conversion of the training dataset to tensors
X = torch.stack(X)
U = torch.stack(U)
X_next = torch.stack(X_next)

# Creation of the Koopman lifted state using thin plate radial basis functions
def create_koopman_state(x, c, Nrbf):
    phi = []
    for i in range(Nrbf):
        r = torch.norm(x - c[i]) # Euclidean distance to RBF center
        phi_i = r**2 * torch.log(r + 1e-6) # Thin plate radial basis function
        phi.append(phi_i)

    phi = torch.stack(phi)
    z = torch.cat((x, phi))
    return z

# Creation of the Koopman lifted states
Z = []

for i in range(X.shape[0]):
    z = create_koopman_state(X[i], c, Nrbf)
    Z.append(z)
Z = torch.stack(Z)

# Creation of the next Koopman lifted states
Z_next = []
for i in range(X_next.shape[0]):
    z_next = create_koopman_state(X_next[i], c, Nrbf)
    Z_next.append(z_next)
Z_next = torch.stack(Z_next)

# Koopman with least squares regression
U = U.unsqueeze(1) # Reshape U to be a column vector
Theta = torch.cat(
    (Z, U),
    dim=1
)
# Least squares solution
K = torch.linalg.lstsq(
    Theta,
    Z_next
).solution

Nz = Z.shape[1]

A = K[:Nz, :].T # Koopman operator for the lifted state
B = K[Nz:, :].T # Koopman operator for the input


# Koopman with soft constraint
lambda_soft = 500.0 # Weight of the soft stability penalty -> SiShiAta 2026 uses 200

# Conversion from PyTorch to NumPy for CVXPY
Z_numpy = Z.numpy()
U_numpy = U.numpy()
Z_next_numpy = Z_next.numpy()

# Optimization variables
A_soft_variable = cp.Variable((Nz, Nz)) # TODO: investigate types in python, "symmetric", etc etc...
B_soft_variable = cp.Variable((Nz, 1))

gamma = cp.Variable()

# Prediction error
residual = Z_next_numpy - (
    Z_numpy @ A_soft_variable.T
    +
    U_numpy @ B_soft_variable.T
)

# LMI used to impose norm 2 for the step
I = np.eye(Nz)
lmi = cp.bmat([
    [gamma * I, A_soft_variable],
    [A_soft_variable.T, gamma * I]
])

# Soft stability constraints
constraints = [
    lmi >> 0
]

# Objective function
objective = cp.Minimize(
    cp.sum_squares(residual)
    +
    lambda_soft * cp.square(gamma)
)

# Optimization problem
problem = cp.Problem(
    objective,
    constraints
)

# Solve the optimization problem
problem.solve(
    solver=cp.SCS,
    verbose=True
)

# Optimized Koopman matrices
A_soft = torch.tensor(
    A_soft_variable.value,
    dtype=torch.float64
)

B_soft = torch.tensor(
    B_soft_variable.value,
    dtype=torch.float64
)


# Analysis of eigenvalues

# Unconstrained Koopman
eigenvalues = torch.linalg.eigvals(A)
rho_A = torch.max(
    torch.abs(eigenvalues)
)
# Soft-constrained Koopman
eigenvalues_soft = torch.linalg.eigvals(A_soft)
rho_A_soft = torch.max(
    torch.abs(eigenvalues_soft)
)

print("rho of A:", rho_A.item())
print("rho of A_soft:", rho_A_soft.item())
print("soft optimization status:", problem.status)

# Test against real system

# Test parameters
T_test = 10.0
N_test = int(T_test / dt)
t_test = torch.arange(
    N_test + 1,
    dtype=torch.float64
) * dt
x0_test = torch.tensor(
    [-0.6, 1.4],
    dtype=torch.float64
) # same as in SiShiAta 2026
# Test input
U_test = []
for k in range(N_test):
    u = 0.8 * torch.sin(
        torch.tensor(
            0.2 * k,
            dtype=torch.float64
        ) # same as in SiShiAta 2026
    )
    U_test.append(u)
U_test = torch.stack(U_test)

# Initial states
x_real = x0_test.clone()
z_koopman = create_koopman_state(
    x0_test,
    c,
    Nrbf
)
z_soft = create_koopman_state(
    x0_test,
    c,
    Nrbf
)

# Initialization of the trajectories
X_real_test = [x_real]
Z_koopman_test = [z_koopman]
Z_soft_test = [z_soft]

# Simulation of the three models
for k in range(N_test):
    u = U_test[k]
    # Real nonlinear system
    x_real = simulate_step(
        x_real,
        u
    )
    # Unconstrained Koopman model
    z_koopman = (
        A @ z_koopman
        +
        B[:, 0] * u
    )
    # Soft-constrained Koopman model
    z_soft = (
        A_soft @ z_soft
        +
        B_soft[:, 0] * u
    )
    X_real_test.append(x_real)
    Z_koopman_test.append(z_koopman)
    Z_soft_test.append(z_soft)


# Conversion of the test trajectories to tensors
X_real_test = torch.stack(X_real_test)
Z_koopman_test = torch.stack(Z_koopman_test)
Z_soft_test = torch.stack(Z_soft_test)

# Recovery of the original states
X_koopman_test = Z_koopman_test[:, :2]
X_soft_test = Z_soft_test[:, :2]

# Comparison of the trajectories

plt.figure(figsize=(12, 5))
# State x1
plt.subplot(1, 2, 1)
plt.plot(
    t_test.numpy(),
    X_real_test[:, 0].numpy(),
    label='Real system'
)
plt.plot(
    t_test.numpy(),
    X_koopman_test[:, 0].numpy(),
    '--',
    label='Koopman'
)
plt.plot(
    t_test.numpy(),
    X_soft_test[:, 0].numpy(),
    '--',
    label='Koopman soft'
)
plt.title('State x1')
plt.xlabel('Time [s]')
plt.ylabel('x1')
plt.grid()
plt.legend()

# State x2
plt.subplot(1, 2, 2)
plt.plot(
    t_test.numpy(),
    X_real_test[:, 1].numpy(),
    label='Real system'
)
plt.plot(
    t_test.numpy(),
    X_koopman_test[:, 1].numpy(),
    '--',
    label='Koopman'
)
plt.plot(
    t_test.numpy(),
    X_soft_test[:, 1].numpy(),
    '--',
    label='Koopman soft'
)
plt.title('State x2')
plt.xlabel('Time [s]')
plt.ylabel('x2')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()