import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

a = 5.8e-4
b = 5e-3
tau = 0.1
k = -0.1

size = 100  # size of the 2D grid
dx = 2.0 / size  # space step
T = 9.0  # total time
dt = 0.001  # time step
n = int(T / dt)  # number of iterations

U = np.random.rand(size, size)
V = np.random.rand(size, size)

def laplacian(Z):
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx**2

fig, ax = plt.subplots(figsize=(8, 8))
image = ax.imshow(U, cmap=plt.cm.copper, interpolation='bilinear', extent=[-1, 1, -1, 1])
ax.set_axis_off()

def update(frame):
    global U, V
    for _ in range(step_plot):
        deltaU = laplacian(U)
        deltaV = laplacian(V)
        Uc = U[1:-1, 1:-1]
        Vc = V[1:-1, 1:-1]
        U[1:-1, 1:-1], V[1:-1, 1:-1] = Uc + dt * (a * deltaU + Uc - Uc**3 - Vc + k),Vc + dt * (b * deltaV + Uc - Vc) / tau
        for Z in (U, V):
            Z[0, :] = Z[1, :]
            Z[-1, :] = Z[-2, :]
            Z[:, 0] = Z[:, 1]
            Z[:, -1] = Z[:, -2]

    image.set_array(U)
    return image,

step_plot = n // 9  # Add this line to define step_plot
animation = FuncAnimation(fig, update, frames=n // step_plot, interval=10, blit=True)

plt.show()
