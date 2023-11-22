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

dt=0.0001
num_points_x = 128
num_points_y = 128

dx = 1/8
dy = 1/8
time_end =10000
num_steps = int(time_end/dt)
print(num_steps)

l=4
Da = 1
Dm=1
W=-30 # -10 - -30
v0=0.1 #0.1-2

kb=0.25 # 0.25-16
ku=0.125 # 0.125-8

c_str=1/64
eta=25
eta0=1
lam=0
K=0.1  # 0.2-2
K2=0
ch=1/4

v=1

C=1
Csig=0.01
P=1
Psig=0.01
Nsig=0.001


center_x = num_points_x // 2
center_y = num_points_y // 2






# Main loop for time-stepping using FTCS method
@njit
def time_evolve(nx,ny,c,p):

    dnx = np.zeros((num_points_x, num_points_x))
    dny = np.zeros((num_points_x, num_points_x))
    dc = np.zeros((num_points_x, num_points_x))

    cnx=c*nx
    cny =c*ny

    n_sq=nx*nx+ny*ny

    alpha = v * (c / c_str - 1)
    beta = v * (1 + c / c_str)

    for i in range(num_points_x):
        for j in range(num_points_y):
            i_minus_1 = (i - 1) % num_points_x
            i_plus_1 = (i + 1) % num_points_x
            j_minus_1 = (j - 1) % num_points_y
            j_plus_1 = (j + 1) % num_points_y
            ### first equation terms

            laplacian_c = (c[i_plus_1, j] - 2 * c[i, j] + c[i_minus_1, j]) / (dx ** 2) + (c[i, j_plus_1] - 2 * c[i, j] + c[i, j_minus_1]) / (dy ** 2)

            grad_cnx=(cnx[i_plus_1, j] - cnx[i_minus_1, j]) / (2*dx)
            grad_cny = (cny[i, j_plus_1] - cny[i, j_minus_1])/(2 * dy)
            grad_px=(p[i_plus_1, j] - p[i_minus_1, j]) / (2*dx) ## also used in 2nd equation
            grad_py=(p[i, j_plus_1] - p[i, j_minus_1])/(2 * dy) ## also used in 2nd equation
            grad_cx = (c[i_plus_1, j] - c[i_minus_1, j]) / (2 * dx)
            grad_cy = (c[i, j_plus_1] - c[i, j_minus_1]) / (2 * dy)
            ###########################




            ##### third term #######
            grad_nx_x=(nx[i_plus_1, j] - nx[i_minus_1, j]) / (2 * dx)
            grad_ny_x = (ny[i_plus_1, j] - ny[i_minus_1, j]) / (2 * dx)
            grad_ny_y =(ny[i, j_plus_1] - ny[i, j_minus_1]) / (2 * dy)
            grad_nx_y = (nx[i, j_plus_1] - nx[i, j_minus_1]) / (2 * dy)
            laplacian_nx = (nx[i_plus_1, j] - 2 * nx[i, j] + nx[i_minus_1, j]) / (dx ** 2) + (nx[i, j_plus_1] - 2 * nx[i, j] + nx[i, j_minus_1]) / (dy ** 2)
            laplacian_ny = (ny[i_plus_1, j] - 2 * ny[i, j] + ny[i_minus_1, j]) / (dx ** 2) + (ny[i, j_plus_1] - 2 * ny[i, j] + ny[i, j_minus_1]) / (dy ** 2)

            del_x_sq_nx= (nx[i_plus_1, j] - 2 * nx[i, j] + nx[i_minus_1, j]) / (dx ** 2)
            del_y_sq_ny= (ny[i, j_plus_1] - 2 * ny[i, j] + ny[i, j_minus_1]) / (dy ** 2)
            del_x_dely_nx=(nx[i_plus_1, j_plus_1] + nx[i_minus_1, j_minus_1] -nx[i_plus_1, j_minus_1]- nx[i_minus_1, j_plus_1]) / (4*dx *dy)
            del_x_dely_ny = (ny[i_plus_1, j_plus_1] + ny[i_minus_1, j_minus_1] - ny[i_plus_1, j_minus_1] - ny[i_minus_1, j_plus_1]) / (4 * dx * dy)
            ############

            ### equations

            dc[i,j] = dt *(Da*laplacian_c-v0*(grad_cnx+grad_cny))

            dnx[i,j]=dt*(-lam*(nx[i,j]*grad_nx_x+ny[i,j]*grad_nx_y)+K*laplacian_nx+K2*(del_x_sq_nx+del_x_dely_nx)+eta*grad_cx+(alpha[i,j]-beta[i,j]*n_sq[i,j])*nx[i,j])
            dny[i,j] = dt * (-lam * (ny[i,j] * grad_ny_x + nx[i,j] * grad_ny_y) + K* laplacian_ny+K2*(del_y_sq_ny+del_x_dely_ny) + eta * grad_cy + (alpha[i,j]  - beta[i,j] * n_sq[i,j]) * ny[i,j])

    return dc,dnx,dny

std_dev=0.5
c = C*(np.ones((num_points_x,num_points_x))+ Csig*np.random.rand(num_points_x,num_points_x))

p = P*(np.ones((num_points_x,num_points_x))+ Psig*np.random.rand(num_points_x,num_points_x))

### disordered IC

theta = np.random.uniform(0, 2 * np.pi, (num_points_x, num_points_y))
nx = np.cos(theta)
ny = np.sin(theta)

### ordered IC
"""""
theta = np.zeros((num_points_x, num_points_y))+Nsig*np.random.uniform(0, 2 * np.pi, (num_points_x, num_points_y))
nx = np.cos(theta)
ny = np.sin(theta)
"""""
for t in range(num_steps):
    dc,dnx,dny=time_evolve(nx,ny,c,p)
    c_new = c + dc

    nx_new=nx+dnx
    ny_new = ny + dny

    """""
    print('T',np.max(k * psi * T), '___________', np.max(va * psi * (divP_x + divP_y)), '_________',np.max(alpha * psi * na))
    print('P',np.max(k * P_psi), '__________', np.max(np.stack((grad_T_x * psi, grad_T_y * psi), axis=-1) * va))
    print('na',np.max(Da * laplacian_na), '__________', np.max(psi*(1 + w * na ** 2)),'__________',np.max(wd *psi* T * na))
    print('change in P',np.mean(P_new)-np.mean(P))
    """""
    c = np.copy(c_new)

    nx = np.copy(nx_new)
    ny = np.copy(ny_new)
    if t%10000==0:
        x = np.linspace(0, length_x, num_points_x)
        y = np.linspace(0, length_y, num_points_y)
        X, Y = np.meshgrid(x, y)

        # Create the quiver plot for vector field P
        plt.quiver(X, Y, nx,ny, alpha=1)

        plt.imshow(c, cmap='jet', extent=[0, length_x, 0, length_y], origin='lower', aspect='auto')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='T')
        plt.title(f'Scalar c with Gradient Term and Vector Field n (Time Step {t})')

        # Save the plot as an image file in the 'plots' directory
        plt.savefig(f'plots/T_and_P_plot_{t}.png')
        #plt.show()
        # Clear the current figure to prepare for the next iteration
        plt.clf()




# Close any remaining open figures
plt.close()
