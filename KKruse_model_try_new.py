import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import dblquad
import random
import os
import math
from numba import njit
import numpy as np
# Create a directory named 'plots' to store the plot images
if not os.path.exists('plots'):
    os.mkdir('plots')
# Parameters
length_x = 1.0
length_y = 1.0

dt=0.0000001
num_points_x = 64
num_points_y = 64

dx = 1
dy = 1
time_end =100
num_steps = int(time_end/dt)
print(num_steps)

D = 0.00
k = 175
alpha = 588
Da=0.045
D_psi=0.003
k2=118
w1=0.006
wd=30
va=20
kd=175
nf0=120
beta=0.036

epsilon=10
delta=0.5

T=1
Tsig=0.1
NA=100
NAsig=20
NI=0
NIsig=0.1




center_x = num_points_x // 2
center_y = num_points_y // 2






# Main loop for time-stepping using FTCS method
@njit
def time_evolve(px,py,T,na,ni):

    dpx = np.zeros((num_points_x, num_points_x))
    dpy = np.zeros((num_points_x, num_points_x))
    dT = np.zeros((num_points_x, num_points_x))
    dna = np.zeros((num_points_x, num_points_x))
    dni = np.zeros((num_points_x, num_points_x))





    for i in range(num_points_x):
        for j in range(num_points_y):
            i_minus_1 = (i - 1) % num_points_x
            i_plus_1 = (i + 1) % num_points_x
            j_minus_1 = (j - 1) % num_points_y
            j_plus_1 = (j + 1) % num_points_y
            ### first equation terms

            laplacian_T = (T[i_plus_1, j] - 2 * T[i, j] + T[i_minus_1, j]) / (dx ** 2) + (T[i, j_plus_1] - 2 * T[i, j] + T[i, j_minus_1]) / (dy ** 2)
            laplacian_px = (px[i_plus_1, j] - 2 * px[i, j] + px[i_minus_1, j]) / (dx ** 2) + (px[i, j_plus_1] - 2 * px[i, j] + px[i, j_minus_1]) / (dy ** 2) ## also used in 2nd equation
            laplacian_py = (py[i_plus_1,j]-2*py[i, j]+py[i_minus_1,j])/(dx** 2)+(py[i, j_plus_1] - 2 * py[i, j] + py[i, j_minus_1]) / (dy ** 2)  ## also used in 2nd equation
            grad_py_x = (py[i_plus_1, j] - py[i_minus_1, j]) / (2 * dx)
            grad_py_y=(py[i, j_plus_1] - py[i, j_minus_1])/(2 * dy)
            grad_Tx = (T[i_plus_1, j] - T[i_minus_1, j]) / (2 * dx)
            grad_Ty = (T[i, j_plus_1] - T[i, j_minus_1]) / (2 * dy)
            ###########################
            #### second equation terms ########
            laplacian_na = (na[i_plus_1, j] - 2 * na[i, j] + na[i_minus_1, j]) / (dx ** 2) + (na[i, j_plus_1] - 2 * na[i, j] + na[i, j_minus_1]) / (dy ** 2)
            laplacian_ni = (ni[i_plus_1, j] - 2 * ni[i, j] + ni[i_minus_1, j]) / (dx ** 2) + (ni[i, j_plus_1] - 2 * ni[i, j] + ni[i, j_minus_1]) / (dy ** 2)
            ############



            dT[i,j] = dt *(D*laplacian_T-kd*(T[i,j])-va*(grad_py_x+grad_py_y)+alpha*na[i,j])
            dpx[i,j]=dt*(-D*laplacian_px-kd*px[i,j]-va*grad_Tx)
            dpy[i, j] = dt * (-D * laplacian_py - kd * py[i, j] - va * grad_Ty)
            dna[i,j]=dt*(Da*laplacian_na+(1+w1*na[i,j]**2)*ni[i,j]-wd*T[i,j]*na[i,j])
            dni[i, j] = dt * (laplacian_ni - (1 + w1 * na[i, j] ** 2) * ni[i, j] + wd * T[i, j] * na[i,j])

    return dT,dpx,dpy,dna,dni

std_dev=0.5
T = T*(np.ones((num_points_x,num_points_x))+ Tsig*np.random.rand(num_points_x,num_points_x))

na = NA*(np.ones((num_points_x,num_points_x))+ NAsig*np.random.rand(num_points_x,num_points_x))
ni = NI*(np.ones((num_points_x,num_points_x))+ NIsig*np.random.rand(num_points_x,num_points_x))
### disordered IC
theta = np.random.uniform(0, 2 * np.pi, (num_points_x, num_points_y))
px = np.cos(theta)
py = np.sin(theta)

### ordered IC
"""""
theta = np.zeros((num_points_x, num_points_y))
nx = np.cos(theta)
ny = np.sin(theta)
"""""
for t in range(num_steps):
    dT,dpx,dpy,dna,dni=time_evolve(px,py,T,na,ni)
    T_new = T + dT
    px_new = px + dpx
    py_new = py + dpy
    na_new=na+dna
    ni_new = ni + dni


    T = np.copy(T_new)
    px = np.copy(px_new)
    py = np.copy(py_new)
    na = np.copy(na_new)
    ni = np.copy(ni_new)
    if t%50000==0:
        x = np.linspace(0, length_x, num_points_x)
        y = np.linspace(0, length_y, num_points_y)
        X, Y = np.meshgrid(x, y)

        # Create the quiver plot for vector field P
        plt.quiver(X, Y, px,py, alpha=1)

        plt.imshow(T, cmap='coolwarm', extent=[0, length_x, 0, length_y], origin='lower', aspect='auto')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='T')
        plt.title(f'Scalar rho with Gradient Term and Vector Field n (Time Step {t})')

        # Save the plot as an image file in the 'plots' directory
        plt.savefig(f'plots1/T_and_P_plot_{t}.png')
        #plt.show()
        # Clear the current figure to prepare for the next iteration
        plt.clf()




# Close any remaining open figures
plt.close()
