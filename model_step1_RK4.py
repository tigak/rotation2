### 1 Bacteria type with resource depletion using RK4

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


# Parameters
mu = 0.5 # bacterial growth rate
k_F = 5 # half-saturation constant - The resource concentration where growth rate is half-maximal
del_F = 0.9 # resource depletion factor - How much resource is consumed per unit bacterial growth
b0 = 100
F0 = 10000
t0 = 0
tf = 100
dt = 0.001

# Run solver
t, b, F = rk4_bacteria_resource(mu, k_F, del_F, b0, F0, t0, tf, dt)

# Plot
plt.plot(t, b, label='Bacteria')
plt.plot(t, F, label='Resources')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.show()