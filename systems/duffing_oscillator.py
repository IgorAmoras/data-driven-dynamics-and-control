# Duffing Oscillator System (as in SiShiAta 2026)
# dot{x1} = x2
# dot{x2} = -delta*x2 - x1*cos(x1 + x2) + u

import torch
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
x0 = -1.0 + 2.0 * torch.rand(2, dtype=torch.float64) # Initial state for static simulation (randomly chosen in the range [-1, 1] for both x1 and x2, with uniform distribution)
u0 = -1.0 + 2.0 * torch.rand((), dtype=torch.float64) # Initial input for static simulation (randomly chosen in the range [-1, 1] with uniform distribution)

# Parameters of the thin plate, using Nrbf = 8, as in SiShiAta 2026
Nrbf = 8 # Number of radial basis functions
c = -1.0 + 2.0 * torch.rand((Nrbf, 2), dtype=torch.float64) # Centers of the radial basis functions, randomly chosen in [-1, 1]^2

# Parameters of the training dataset
N_trajectories = 20000 # Number of independent trajectories
N_steps = 2 # Number of simulation steps per trajectory]

# Duffing oscillator dynamics 
def dynamics(t, x,u):
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

# Generation of the training dataset
for trajectory in range(N_trajectories):
    x = -1.0 + 2.0 * torch.rand(2, dtype=torch.float64) # Random initial state in [-1, 1]
    for step in range(N_steps):
        u = -1.0 + 2.0 * torch.rand((), dtype=torch.float64) # Random input in [-1, 1]
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
    phi = [] # List of radial basis function values
    for i in range(Nrbf):
        r = torch.norm(x - c[i]) # Euclidean distance between state x and center c[i]
        phi_i = r**2 * torch.log(r + 1e-6) # Thin plate radial basis function
        phi.append(phi_i)
    phi = torch.stack(phi) # Convert the list of RBF values into a tensor
    z = torch.cat((x, phi)) # Augment the original state with the RBF observables

    return z

# Creation of the Koopman lifted states for the training dataset
Z = [] 
for i in range(X.shape[0]):
    z = create_koopman_state(X[i], c, Nrbf)
    Z.append(z)
Z = torch.stack(Z) 
Z_next = [] 
for i in range(X_next.shape[0]):
    z_next = create_koopman_state(X_next[i], c, Nrbf)
    Z_next.append(z_next)
Z_next = torch.stack(Z_next) 

# Build the Koopman operator using least squares regression
U = U.unsqueeze(1) # Reshape U to be a column vector
Theta = torch.cat((Z, U), dim=1) # Concatenate Z and U to form the regression matrix

# Least squares solution for the Koopman operator K
K = torch.linalg.lstsq(Theta, Z_next).solution # Solve for K in the least squares sense

Nz = Z.shape[1]

A = K[:Nz, :].T# Koopman operator for the lifted state
B = K[Nz:, :].T # Koopman operator for the input

# Test against real system using SiShiAta 2026 parameters

# Parameters
T_test = 10.0 # Total test simulation time
N_test = int(T_test / dt) # Number of test simulation steps
t_test = torch.arange(N_test + 1, dtype=torch.float64) * dt # Test time vector

x0_test = torch.tensor([-0.6, 1.4], dtype=torch.float64) # Initial condition used for the test

U_test = []
for k in range(N_test):
    u = 0.8 * torch.sin(torch.tensor(0.2 * k, dtype=torch.float64)) # signal used in SiShiAta 2026
    U_test.append(u)
U_test = torch.stack(U_test)

x_real = x0_test.clone() # initiital state for real simulation
z_koopman = create_koopman_state(x0_test, c, Nrbf) # initial lifted state for Koopman simulation

# Initialization of the test trajectories

X_real_test = [x_real]
Z_koopman_test = [z_koopman]

# Simulation of the real and identified Koopman models
for k in range(N_test):
    u = U_test[k]
    # Real nonlinear system
    x_real = simulate_step(x_real, u)
    # Identified linear Koopman model
    z_koopman = A @ z_koopman + B[:, 0] * u
    X_real_test.append(x_real)
    Z_koopman_test.append(z_koopman)

X_real_test = torch.stack(X_real_test)
Z_koopman_test = torch.stack(Z_koopman_test)
X_koopman_test = Z_koopman_test[:, :2]

# Spectral analysis of the identified Koopman operator
eigenvalues = torch.linalg.eigvals(A) 
rho_A = torch.max(torch.abs(eigenvalues)) # Spectral radius of A
print("Spectral radius of A:", rho_A.item())

# Comparison between the real system and the Koopman model

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(t_test.numpy(), X_real_test[:, 0].numpy(), label='Real system')
plt.plot(t_test.numpy(), X_koopman_test[:, 0].numpy(), '--', label='Koopman model')

plt.title('State x1')
plt.xlabel('Time [s]')
plt.ylabel('x1')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(t_test.numpy(), X_real_test[:, 1].numpy(), label='Real system')
plt.plot(t_test.numpy(), X_koopman_test[:, 1].numpy(), '--', label='Koopman model')

plt.title('State x2')
plt.xlabel('Time [s]')
plt.ylabel('x2')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()