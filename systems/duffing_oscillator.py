# Duffing Oscillator System (as in SiShiAta 2026)

# dot{x1} = x2
# dot{x2} = -delta*x2 - x1*cos(x1 + x2) + u

# x = [x1, x2] in R^2
# u in R

import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Parameters
dt = 0.1 # Sampling time
delta = 1.0 # Damping coefficient
T = 20.0 # Total simulation time
N = T/dt # Number of time steps 
t = torch.linspace(0, T, int(N), dtype=torch.float64) # Time vector
method = 'rk4' # Integration method
options = {'step_size': dt} # Integration options

x0 = torch.tensor([-1.0, 1.0], dtype=torch.float64)
u0 = torch.tensor(0.0, dtype=torch.float64)

def dynamics(t, x,u):
    x1, x2 = x
    dx1 = x2
    dx2 = -delta*x2 - x1*torch.cos(x1 + x2) + u
    return torch.stack([dx1, dx2])

solution = odeint(
    lambda t, x: dynamics(t, x, u0),
    x0,
    t,
    method=method,
    options=options
)

print("Solution shape:", solution.shape)  # Should be (N, 2)

# Plotting the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t.numpy(), solution[:, 0].numpy(), label='x1 (Position)')
plt.title('Duffing Oscillator Position')
plt.xlabel('Time [s]')
plt.ylabel('x1')
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(t.numpy(), solution[:, 1].numpy(), label='x2 (Velocity)', color='orange')
plt.title('Duffing Oscillator Velocity')
plt.xlabel('Time [s]')
plt.ylabel('x2')
plt.grid()
plt.tight_layout()
plt.show()
