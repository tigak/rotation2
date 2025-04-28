### 1 Bacteria type with resource depletion using Forward Euler

import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 0.5 # bacterial growth rate
k_F = 5 # half-saturation constant - The resource concentration where growth rate is half-maximal
del_F = 0.9 # resource depletion factor - How much resource is consumed per unit bacterial growth

# Time
t0 = 0 # initial time
tf = 100 # end time
dt = 0.001 # time step
t = np.arange(t0, tf, dt) #array of time points
# Bacteria
b = np.empty_like(t) # array for storing bacteria values
b[0] = 100 # initial value of bacteria
# Food
F = np.empty_like(t) # array for resource values
F[0] = 10000 # initial amount of resources

# Main Loop
for i in range(1,len(t)):
    b[i] = b[i-1] + dt * (mu * b[i-1] * (F[i-1]/(F[i-1] + k_F)))
    F[i] = F[i-1] - dt * mu * b[i-1] * (F[i-1]/(F[i-1] + k_F)) * del_F
    # if i % 1000 == 0:
    #     print(f"t={t[i]:.2f}, b={b[i]:.2f}, F={F[i]:.2f}")



# Plot
plt.plot(t, b, label='Bacteria')
plt.plot(t, F, label='Resources')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.show()