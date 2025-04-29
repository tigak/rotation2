### 1 Bacteria type with resource depletion using Runge-Kutta 4th

import numpy as np
import matplotlib.pyplot as plt

def rk4_bacteria_resource(mu, k_F, del_F, b0, F0, t0, tf, dt):
    """
    Solves the bacteria + resource ODE system using 4th order Runge-Kutta.

    Parameters:
    - mu: growth rate
    - k_F: half-saturation constant
    - del_F: resource depletion factor
    - b0: initial bacteria amount
    - F0: initial food/resource amount
    - t0: initial time
    - tf: final time
    - dt: time step

    Returns:
    - t: array of time points
    - b: array of bacteria concentrations
    - F: array of resource concentrations
    """
    t = np.arange(t0, tf, dt) #array of time points
    # Bacteria
    b = np.empty_like(t) # array for storing bacteria values
    # Food
    F = np.empty_like(t) # array for resource values

    # Initial Conditions
    b[0] = b0 # initial value of bacteria
    F[0] = F0 # initial amount of resources

    # Main Loop
    for i in range(1, len(t)):
        k1_b= mu * b[i-1] * (F[i-1]/(F[i-1] + k_F))
        k1_f = -mu * b[i-1] * (F[i-1]/(F[i-1] + k_F)) * del_F

        k2_b = mu * (b[i-1] + 0.5 * dt * k1_b) * ((F[i-1] + 0.5 * dt * k1_f)/(F[i-1] + (0.5 * dt * k1_f) + k_F))
        k2_f = -mu * (b[i-1] + 0.5 * dt * k1_b) * ((F[i-1] + 0.5 * dt * k1_f)/(F[i-1] + (0.5 * dt * k1_f) + k_F)) * del_F

        k3_b = mu * (b[i-1] + 0.5 * dt * k2_b) * ((F[i-1] + 0.5 * dt * k2_f)/(F[i-1] + (0.5 * dt * k2_f) + k_F))
        k3_f = -mu * (b[i-1] + 0.5 * dt * k2_b) * ((F[i-1] + 0.5 * dt * k2_f)/(F[i-1] + (0.5 * dt * k2_f) + k_F)) * del_F
        
        k4_b = mu * (b[i-1] + dt * k3_b) * ((F[i-1] + dt * k3_f)/(F[i-1] + (dt * k3_f) + k_F))
        k4_f = -mu * (b[i-1] + dt * k3_b) * ((F[i-1] + dt * k3_f)/(F[i-1] + (dt * k3_f) + k_F)) * del_F

        # Update
        b[i] = b[i-1] + (dt/6) * (k1_b + 2*k2_b + 2*k3_b + k4_b)
        F[i] = F[i-1] + (dt/6) * (k1_f + 2*k2_f + 2*k3_f + k4_f)

    return t, b, F


def rk4_bacteria_phage(mu, k_F, del_F, b0, F0, t0, tf, dt, phi, eta, del_p, p0):
    import numpy as np

    t = np.arange(t0, tf, dt)
    b = np.empty_like(t)
    F = np.empty_like(t)
    p = np.empty_like(t)

    b[0] = b0
    F[0] = F0
    p[0] = p0

    for i in range(1, len(t)):
        # Current values
        b0_i = b[i-1]
        F0_i = F[i-1]
        p0_i = p[i-1]

        # --- k1 ---
        k1_b = mu * b0_i * (F0_i / (F0_i + k_F)) - phi * p0_i * b0_i
        k1_f = -mu * b0_i * (F0_i / (F0_i + k_F)) * del_F
        k1_p = (eta * p0_i * b0_i) - (del_p * p0_i)

        # --- k2 ---
        b1 = b0_i + 0.5 * dt * k1_b
        F1 = F0_i + 0.5 * dt * k1_f
        p1 = p0_i + 0.5 * dt * k1_p

        k2_b = mu * b1 * (F1 / (F1 + k_F)) - phi * p1 * b1
        k2_f = -mu * b1 * (F1 / (F1 + k_F)) * del_F
        k2_p = (eta * p1 * b1) - (del_p * p1)

        # --- k3 ---
        b2 = b0_i + 0.5 * dt * k2_b
        F2 = F0_i + 0.5 * dt * k2_f
        p2 = p0_i + 0.5 * dt * k2_p

        k3_b = mu * b2 * (F2 / (F2 + k_F)) - phi * p2 * b2
        k3_f = -mu * b2 * (F2 / (F2 + k_F)) * del_F
        k3_p = (eta * p2 * b2) - (del_p * p2)

        # --- k4 ---
        b3 = b0_i + dt * k3_b
        F3 = F0_i + dt * k3_f
        p3 = p0_i + dt * k3_p

        k4_b = mu * b3 * (F3 / (F3 + k_F)) - phi * p3 * b3
        k4_f = -mu * b3 * (F3 / (F3 + k_F)) * del_F
        k4_p = (eta * p3 * b3) - (del_p * p3)

        # --- Update ---
        b[i] = b0_i + (dt/6) * (k1_b + 2*k2_b + 2*k3_b + k4_b)
        F[i] = F0_i + (dt/6) * (k1_f + 2*k2_f + 2*k3_f + k4_f)
        p[i] = p0_i + (dt/6) * (k1_p + 2*k2_p + 2*k3_p + k4_p)

        # Prevent numerical blow-up
        b[i] = max(b[i], 0)
        F[i] = max(F[i], 0)
        p[i] = max(p[i], 0)

    return t, b, F, p

# Parameters
mu = 0.5 # bacterial growth rate
k_F = 5 # half-saturation constant - The resource concentration where growth rate is half-maximal
del_F = 0.9 # resource depletion factor - How much resource is consumed per unit bacterial growth
b0 = 50 # initial bacteria
F0 = 10000 # initial food
t0 = 0 #start time
tf = 10 #end time
dt = 0.001 #step size
phi = 1e-8 #infectivity
eta = 100 # phage replication rate
del_p = 0.001 # natural phage decay
p0 = 10 # initial phage

# Run solver
#t, b, F = rk4_bacteria_resource(mu, k_F, del_F, b0, F0, t0, tf, dt)
t, b, F, p = rk4_bacteria_phage(mu, k_F, del_F, b0, F0, t0, tf, dt, phi, eta, del_p, p0)

# Plot
fig, axs = plt.subplots(3,1)
axs[0].plot(t, b, label='Bacteria')
axs[1].plot(t, F, label='Resources')
axs[2].plot(t, p, label='Phage')
#plt.yscale('log')
#plt.ylim(0, 10000)
#plt.xlim(t0, tf)
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.show()

#Phage are dropping almost immediately to 0?