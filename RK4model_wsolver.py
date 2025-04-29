from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

def bacteria_phage_ode(t, x, mu, k_F, del_F, phi, eta, delta):
    b, F, p = x  # unpack the state vector

    # Avoid division by zero
    F_term = F / (F + k_F) if F + k_F != 0 else 0

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

# Initial conditions
b0 = 50
F0 = 10000
p0 = 10
x0 = [b0, F0, p0]

# Time span
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Solve
sol = solve_ivp(
    bacteria_phage_ode,
    t_span,
    x0,
    args=(mu, k_F, del_F, phi, eta, delta),
    t_eval=t_eval,
    method='RK45',  # or 'RK23', or 'Radau' for stiff systems
    vectorized=False
)

# Plot
fig, axs = plt.subplots(3,1)
axs[0].plot(sol.t, sol.y[0], label='Bacteria')
axs[1].plot(sol.t, sol.y[1], label='Resources')
axs[2].plot(sol.t, sol.y[2], label='Phage')
#plt.yscale('log')
#plt.ylim(0, 10000)
#plt.xlim(t0, tf)
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.show()
