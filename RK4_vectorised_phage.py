# using vectorisation to solve RK4 for single bacteria, limited resources and phage
import numpy as np
import matplotlib.pyplot as plt

# model as a vector function
def f(t, x, mu, k_F, del_F, phi, eta, delta):
    b, F, p = x

    # Avoid division by zero
    F_term = F / (F + k_F) if F + k_F != 0 else 0

    db = mu * b * F_term - phi * p * b
    dF = -mu * b * F_term * del_F
    dp = eta * p * b - delta * p

    return np.array([db, dF, dp])

def rk4_system(f, x0, t, args=(), threshold=1e-6, extinction_stop=True):
    x = np.zeros((len(t), len(x0)))
    x[0] = x0

    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        ti = t[i-1]
        xi = x[i-1]

        k1 = f(ti, xi, *args)
        k2 = f(ti + 0.5 * dt, xi + 0.5 * dt * k1, *args)
        k3 = f(ti + 0.5 * dt, xi + 0.5 * dt * k2, *args)
        k4 = f(ti + dt, xi + dt * k3, *args)

        x[i] = xi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Prevent negatives
        x[i] = np.maximum(x[i], 1e-8)

        # Stop if phage or bacteria extinct
        if extinction_stop and (x[i][0] < threshold or x[i][2] < threshold):
            x[i:] = x[i]  # hold the final value constant to match output shape
            break

    return x


# Parameters
mu = 0.5
k_F = 5
del_F = 0.9
phi = 1e-8
eta = 100
delta = 0.001

# Initial conditions
x0 = np.array([50, 10000, 10])  # [b, F, p]
dt = 0.00001  
t = np.linspace(0, 100, int(100 / dt))

# Integrate
X = rk4_system(f, x0, t, args=(mu, k_F, del_F, phi, eta, delta))

# Plot
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
axs[0].plot(t, X[:, 0], label='Bacteria')
axs[0].set_ylabel('Bacteria')
axs[1].plot(t, X[:, 1], label='Resources', color='orange')
axs[1].set_ylabel('Resources')
axs[2].plot(t, X[:, 2], label='Phage', color='green')
axs[2].set_ylabel('Phage')
axs[2].set_xlabel('Time')
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.show()