import numpy as np

def bacteria_phage_ode(t, x, mu, k_F, del_F, phi, eta, delta):
    b, F, p = x

    # Avoid division by zero
    F_term = F / (F + k_F) if (F + k_F) != 0 else 0

    dbdt = mu * b * F_term - phi * p * b
    dFdt = -mu * b * F_term * del_F
    dpdt = eta * p * b - delta * p

    return [dbdt, dFdt, dpdt]

# Parameters
mu = 0.5
k_F = 5
del_F = 0.9
phi = 1e-8
eta = 100
delta = 0.001

# Initial conditions: [bacteria, resources, phage]
x0 = [50, 10000, 10]

# Time span and evaluation points
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 1000000)  # uniform output times

from scipy.integrate import solve_ivp

sol = solve_ivp(
    fun=lambda t, x: bacteria_phage_ode(t, x, mu, k_F, del_F, phi, eta, delta),
    t_span=t_span,
    y0=x0,
    method="Radau",  # stiff-aware solver
    t_eval=t_eval
)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axs[0].plot(sol.t, sol.y[0], label="Bacteria")
axs[0].set_ylabel("Bacteria")
axs[1].plot(sol.t, sol.y[1], label="Resources")
axs[1].set_ylabel("Resources")
axs[2].plot(sol.t, sol.y[2], label="Phage")
axs[2].set_ylabel("Phage")
axs[2].set_xlabel("Time")
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.show()
