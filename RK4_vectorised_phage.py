# using vectorisation to solve RK4 for single bacteria, limited resources and phage
import numpy as np
import matplotlib.pyplot as plt

# model as a vector function
def f(t, x, mu, k_F, del_F, phi, eta, delta, phage_on=True):
    b, F, p = x

    # Avoid division by zero
    F_term = F / (F + k_F) if F + k_F != 0 else 0

    db = mu * b * F_term - phi * p * b
    dF = -mu * b * F_term * del_F
    dp = 0

    if phage_on:
        db -= phi * p * b
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
        print(x[i])
        # Stop if phage or bacteria extinct
        if extinction_stop and (x[i][0] < threshold or x[i][2] < threshold):
            x[i:] = x[i]  # hold the final value constant to match output shape
            break

    return x

def rk4_dilution(f, x0, t, args=(), dilution_hours = 2, threshold=1e-6, extinction_stop=True):
    x = np.zeros((len(t), len(x0)), dtype=np.float64)
    x[0] = x0
    dt = t[1] - t[0]
    dilution_interval = round(dilution_hours * 3600 / dt)  # number of steps for dilution)

    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        ti = t[i-1]
        xi = x[i-1]

        # RK4 integration
        k1 = f(ti, xi, *args)
        k2 = f(ti + 0.5 * dt, xi + 0.5 * dt * k1, *args)
        k3 = f(ti + 0.5 * dt, xi + 0.5 * dt * k2, *args)
        k4 = f(ti + dt, xi + dt * k3, *args)
        x[i] = xi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        #print(x[i])
        # Dilution Step
        if i % dilution_interval == 0:
            print(f"Dilution at t = {t[i]/3600:.2f} hours (step {i})")
            x[i][0] /= 10  # 10 fold dilution of bacteria
            x[i][2] /= 10  # 10 fold dilution of phage
            x[i][1] = 1    # refresh resources

        x[i] = np.maximum(x[i], 0) # was 1e-8 previously

        # if extinction_stop and (x[i][0] < threshold or x[i][2] < threshold):
        #     x[i:] = x[i]
        #     print('extinction')
        #     break

    return x

# Parameters
mu = 0.2 # bacterial growth rate
F0 = 1 # initial amount of resources (arbitrary)
k_F = 25*F0 # half-saturation constant - The resource concentration where growth rate is half-maximal
del_F = 1e-9 # resource depletion factor - How much resource is consumed per unit bacterial growth
phi = 1e-9 # phage infectivity - doesn't seem to affect rate of phage growth
eta = 0.00001/1800 # 100 phage produced per bacteria per half an hour - adjusted so that phage don't explode as rapidly
delta = 0.01 # phage decay rate
b0 = 1e5 # initial bacteria
p0 = 10 # initial phage
max_time_hours = 48
# Convert hours to seconds
max_time_seconds = max_time_hours * 3600
#max_time_seconds= 12000

# Initial conditions
x0 = np.array([b0, F0, p0], dtype = np.float64)  # [b, F, p]
dt = 0.1  
t = np.linspace(0, max_time_seconds, int(max_time_seconds / dt) + 1)  # time points

# Integrate
#X = rk4_system(f, x0, t, args=(mu, k_F, del_F, phi, eta, delta))
args_with_phage = (mu, k_F, del_F, phi, eta, delta, True)
args_without_phage = (mu, k_F, del_F, phi, eta, delta, False)
#X = rk4_dilution(f, x0, t, args=args_without_phage, dilution_hours=0.5)
X = rk4_dilution(f, x0, t, args=args_with_phage, dilution_hours=6)

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
#plt.savefig('RK4_vectorised_phage.png', dpi = 600)
plt.show()

# when max_time_seconds = 10, the output is: what you would expect? (see saved graph)
# when max_time_seconds = 48h the output is as before - goes to 0