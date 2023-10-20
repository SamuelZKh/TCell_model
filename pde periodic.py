import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import dblquad
import random
import os
import math
from numba import njit
# Create a directory named 'plots' to store the plot images
if not os.path.exists('plots'):
    os.mkdir('plots')
# Parameters
length_x = 1.0
length_y = 1.0

dt=0.0000006
num_points_x = 64
num_points_y = 64

dx = 0.01
dy = 0.01
time_end =0.01
num_steps = int(time_end/dt)
print(num_steps)
D = 0.00
k = 175
alpha = 288
Da=0.045
D_psi=0.003
k2=118
w=0.006
wd=0.3
va=0.34
kd=175
nf0=120
beta=0.036
V0=0.15
epsilon=10
delta=0.5

radius = 10
@njit
def integrand(x, y):
    # Convert continuous coordinates to integer indices
    i = int(round(x))
    j = int(round(y))
    if i==0:
        theta=np.pi/2
    else:
        theta = np.arctan(j / i)
    r=np.sqrt(i**2+j**2)
    # Use the integer indices to access the psi array
   # return r*(np.cos(theta))**2*psi[int(r*np.cos(theta)), int(r*np.cos(theta))]
    return psi[i,j]

def gradient(f, dx, dy):
    grad_x = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)
    grad_y = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dy)
    return grad_x, grad_y

# Define a function to wrap around indices for periodic boundary conditions
def periodic_index(i, size):
    if i < 0:
        return size + i
    elif i >= size:
        return i - size
    else:
        return i
# Initialize arrays to store the solutions for T and P
#T = np.ones((num_points_x, num_points_y))

#na = np.zeros((num_points_x, num_points_y))
#ni= np.zeros((num_points_x, num_points_y))
#psi= np.ones((num_points_x, num_points_y))
#P = np.ones((num_points_x, num_points_y, 2))

import numpy as np


center_x = num_points_x // 2
center_y = num_points_y // 2

# Create an array of zeros
psi = np.zeros((num_points_x, num_points_y))

# Define the radius of the circle
#radius = min(center_x, center_y)

# Set the elements in the circle around the center to 1
for i in range(num_points_x):
    for j in range(num_points_y):
        if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius ** 2:
            psi[i, j] = 1

na = np.zeros((num_points_x, num_points_y))
ni = np.zeros((num_points_x, num_points_y))
# Define the radius of the circle


# Set the elements in the circle around the center to 1
for i in range(num_points_x):
    for j in range(num_points_y):
        if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius ** 2:
            na[i, j] = nf0+random.uniform(-40,40)

# Print the resulting array
print(psi)

std_dev = 0.01  # You can adjust this value based on how "small" you want the random values to be

# Initialize the matrices with small positive random values from a normal distribution


#psi = np.abs(np.random.normal(loc=0.0, scale=std_dev, size=(num_points_x, num_points_y)))
#P = np.random.normal(loc=0.0, scale=3, size=(num_points_x, num_points_y, 2))





T=np.abs(np.random.normal(loc=0.0, scale=std_dev, size=(num_points_x, num_points_y)))
#P[int(num_points_x / 4):int(3 * num_points_x / 4), int(num_points_y / 4):int(3 * num_points_y / 4), 0] = -1.0
#P[int(num_points_x / 4):int(3 * num_points_x / 4), int(num_points_y / 4):int(3 * num_points_y / 4), 1] = 1.0
T = np.zeros((num_points_x, num_points_y))

for i in range(num_points_x):
    for j in range(num_points_y):
        if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius ** 2:
            T[i, j] = random.uniform(0,1)

#"""""
# Create the P array with random perturbations
P = np.zeros((num_points_x, num_points_y, 2))
for i in range(num_points_x):
    for j in range(num_points_y):
        # Calculate the relative position to the center
        relative_x = i - center_x
        relative_y = j - center_y

        # Calculate the magnitude of the relative position
        magnitude = 20*np.sqrt(relative_x ** 2 + relative_y ** 2)

        # Ensure that the magnitude is not zero to avoid division by zero
        if magnitude > 0:
            # Calculate the components of the vector P
            P[i, j, 0] = -10 * relative_x / magnitude
            P[i, j, 1] = -10 * relative_y / magnitude

            # Add a random perturbation to the direction
            random_angle = np.random.uniform(0,2*np.pi)
            perturbation = np.array([np.cos(random_angle), np.sin(random_angle)])
            P[i, j] = P[i, j] + perturbation*0.4

#"""""

"""""
radius = 50.0  # Change this to your desired radius
theta = np.pi/4  # Change this to your desired angle in radians

# Initialize the matrix P with vectors using radius and theta
x_component = radius * np.cos(theta)
y_component = radius * np.sin(theta)
P = np.array([[ [x_component, y_component] for _ in range(num_points_y)] for _ in range(num_points_x)])
#"""""
print(P)
print('asdasd',np.max(P),np.min(P))
# Main loop for time-stepping using FTCS method
for n in range(num_steps):
    print(n)

    grad_T_x, grad_T_y = gradient(T, dx, dy)

    divP_x = np.zeros((num_points_x, num_points_y))
    divP_y = np.zeros((num_points_x, num_points_y))
    laplacian_T = np.zeros((num_points_x, num_points_y))
    laplacian_na = np.zeros((num_points_x, num_points_y))
    laplacian_ni = np.zeros((num_points_x, num_points_y))
    laplacian_psi = np.zeros((num_points_x, num_points_y))
    laplacian_P = np.zeros((num_points_x, num_points_y, 2))

    for i in range(num_points_x):
        for j in range(num_points_y):
            i_minus_1 = (i - 1) % num_points_x
            i_plus_1 = (i + 1) % num_points_x
            j_minus_1 = (j - 1) % num_points_y
            j_plus_1 = (j + 1) % num_points_y

            divP_x[i, j] = (P[i_plus_1, j, 0] - P[i_minus_1, j, 0]) / dx
            divP_y[i, j] = (P[i, j_plus_1, 1] - P[i, j_minus_1, 1]) / dy

            laplacian_T[i, j] = (T[i_plus_1, j] - 2 * T[i, j] + T[i_minus_1, j]) / (dx ** 2) + (T[i, j_plus_1] - 2 * T[i, j] + T[i, j_minus_1]) / (dy ** 2)
            laplacian_na[i, j] = (na[i_plus_1, j] - 2 * na[i, j] + na[i_minus_1, j]) / (dx ** 2) + (na[i, j_plus_1] - 2 * na[i, j] + na[i, j_minus_1]) / (dy ** 2)
            laplacian_ni[i, j] = (ni[i_plus_1, j] - 2 * ni[i, j] + ni[i_minus_1, j]) / (dx ** 2) + (ni[i, j_plus_1] - 2 * ni[i, j] + ni[i, j_minus_1]) / (dy ** 2)
            laplacian_psi[i, j] = (psi[i_plus_1, j] - 2 * psi[i, j] + psi[i_minus_1, j]) / (dx ** 2) + (psi[i, j_plus_1] - 2 * psi[i, j] + psi[i, j_minus_1]) / (dy ** 2)

            laplacian_P[i, j, 0] = (P[i_plus_1, j, 0] - 2 * P[i, j, 0] + P[i_minus_1, j, 0]) / (dx ** 2) + (P[i, j_plus_1, 0] - 2 * P[i, j, 0] + P[i, j_minus_1, 0]) / (dy ** 2)
            laplacian_P[i, j, 1] = (P[i_plus_1, j, 1] - 2 * P[i, j, 1] + P[i_minus_1, j, 1]) / (dx ** 2) + (P[i, j_plus_1, 1] - 2 * P[i, j, 1] + P[i, j_minus_1, 1]) / (dy ** 2)

    grad_psi_x, grad_psi_y = gradient(psi, dx, dy)

    P_psi = np.zeros((num_points_x, num_points_y, 2))

    # Inside the loop for P_psi and integrated_psi calculations
    for i in range(1, num_points_x - 1):
        for j in range(1, num_points_y - 1):


            P_psi[i, j, 0] = P[i,j,0] * psi[i, j]

            P_psi[i, j, 1] = P[i, j, 1]*psi[i, j]

    # Inside the loop for integrated_psi calculations
    integrated_psi = np.zeros((num_points_x, num_points_y))
    for i in range(1, num_points_x - 1):
        for j in range(1, num_points_y - 1):
            x_min, x_max = i * dx, (i + 1) * dx
            y_min, y_max = j * dy, (j + 1) * dy

            integrated_value, _ = dblquad(integrand, x_min, x_max, y_min, y_max)

            integrated_psi[i, j] = integrated_value

    # Update T and P using the FTCS method
    delta = (delta + epsilon * (integrated_psi) - V0)
    ###############################################################################################################
    #print( (psi).shape)

    T_new = T + dt * (D*laplacian_T- k*psi*T -va*psi* (divP_x + divP_y)+alpha*psi*na)
    P_new = P + dt * (D * laplacian_P - k*P_psi-np.stack((grad_T_x*psi, grad_T_y*psi), axis=-1) * va)

    na_new = na+(Da * laplacian_na + psi*(1 + w * na ** 2) * ni - wd *psi* T * na)*dt
    ni_new = ni+(laplacian_ni - psi*(1 + w * na ** 2) * ni + wd * T * na*psi)*dt
    print('T',np.max(k * psi * T), '___________', np.max(va * psi * (divP_x + divP_y)), '_________',np.max(alpha * psi * na))
    print('P',np.max(k * P_psi), '__________', np.max(np.stack((grad_T_x * psi, grad_T_y * psi), axis=-1) * va))
    print('na',np.max(Da * laplacian_na), '__________', np.max(psi*(1 + w * na ** 2)),'__________',np.max(wd *psi* T * na))
    print('change in P',np.mean(P_new)-np.mean(P))
    if math.isnan(np.max(alpha * psi * na)):
        print("Found NaN, breaking out of the loop.")
        break
    #print(beta*(P[:, :, 0] * grad_psi_x + P[:, :, 1] * grad_psi_y))
    psi_new=psi+(k2*psi*(1-psi)*(psi-delta)+D_psi*laplacian_psi-beta*(P[:, :, 0] * grad_psi_x + P[:, :, 1] * grad_psi_y))*dt
   # print(psi)
    na = np.copy(na_new)
    ni = np.copy(ni_new)
    T = np.copy(T_new)
    P = np.copy(P_new)
    psi=np.copy(psi_new)
    #print(np.unique(psi))
    # Plot the final solution for T at the final time step
    x = np.linspace(0, length_x, num_points_x)
    y = np.linspace(0, length_y, num_points_y)
    X, Y = np.meshgrid(x, y)

    # Create the quiver plot for vector field P
    plt.quiver(X, Y, P[:, :, 0], P[:, :, 1], alpha=1)

    plt.imshow(T, cmap='jet', extent=[0, length_x, 0, length_y], origin='lower', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='T')
    plt.title(f'Scalar T with Gradient Term and Vector Field P (Time Step {n})')

    # Save the plot as an image file in the 'plots' directory
    plt.savefig(f'plots/T_and_P_plot_{n}.png')
    #plt.show()
    # Clear the current figure to prepare for the next iteration
    plt.clf()



# Plot the final solution for T at the final time step
x = np.linspace(0, length_x, num_points_x)
y = np.linspace(0, length_y, num_points_y)
X, Y = np.meshgrid(x, y)

# Create the quiver plot for vector field P
plt.quiver(X, Y, P[:, :, 0], P[:, :, 1], alpha=0.7)



plt.imshow(T, cmap='jet', extent=[0, length_x, 0, length_y], origin='lower', aspect='auto')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='T')
plt.title(f'Scalar T with Gradient Term and Vector Field P (Time Step {n})')

plt.show()

# Close any remaining open figures
plt.close()
