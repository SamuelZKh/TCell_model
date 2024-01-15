import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import dblquad
import random
import os
import math
import csv
from numba import njit
# Create a directory named 'plots' to store the plot images
if not os.path.exists('plots'):
    os.mkdir('plots')


# Parameters
length_x = 1.0
length_y = 1.0

dt=0.00001
num_points_x = 60
num_points_y = 60

dx = 1
dy = 1
time_end =10e4
num_steps = int(time_end/dt)
print(num_steps)

l=4
Da =2
Dm=2
W=-0 # -10 - -20
v0=0.9 #0.1-2

kb=2.25 # 0.25-16
ku=0.35 # 0.125-8

c_str=1/64
eta=0.5
eta0=2
lam=0
K=0.2 # 0.2-2
ch=1/4
v=0.3

epsilon=1
sig=30
D_psi=1

C=0.1
Csig=0.01
P=0.1
Psig=0.01
kappa=40
radius=num_points_y/3
gamma=2


center_x = num_points_x // 2
center_y = num_points_y // 2
import numpy as np

print('if',kb*np.abs(W),'>',Da*ku,'density clumping instability')
print('if',Da*v * (C / c_str - 1),'>',v0*eta*P,'density clumping instability')

if kb*np.abs(W)>Da*ku:
    print(kb*np.abs(W),'>',Da*ku,'!!!!!!!unstable!!!!!!!')

if Da*v * (C / c_str - 1)>v0*eta*P:
    print(Da*v * (C / c_str - 1),'>',v0*eta*P, '!!!!!!!unstable!!!!!!!!')



def set_below_threshold_to_zero(matrix, threshold):

    for i in range(len(matrix)):

        for j in range(len(matrix[i])):

            if matrix[i][j] < threshold:

                matrix[i][j] = 0




################################ File Create #########################
# Create folder name
folder_name = f'W={W},eta={eta},kb={kb},ku={ku},v={v},K={K},Da={Da},v0={v0}'

# Specify the path where you want to create the folder
 # Change this to your desired directory path

# Join the path and folder name
full_path = os.path.join(folder_name)

# Check if the folder already exists
if not os.path.exists(full_path):
    # Create the folder
    os.makedirs(full_path)
    print(f'Folder "{folder_name}" created successfully at {full_path}')
else:
    print(f'Folder "{folder_name}" already exists at {full_path}')
#############################################################################

# Main loop for time-stepping using FTCS method
@njit
def time_evolve(nx,ny,c,p,psi,V0):

    dnx = np.zeros((num_points_x, num_points_x))
    dny = np.zeros((num_points_x, num_points_x))
    dc = np.zeros((num_points_x, num_points_x))
    dp = np.zeros((num_points_x, num_points_x))
    dpsi= np.zeros((num_points_x, num_points_x))
    integral_psi = np.sum(psi >= 0.55)



    for i in range(num_points_x):
        for j in range(num_points_y):

            i_minus_1 = (i - 1) % num_points_x
            i_plus_1 = (i + 1) % num_points_x
            j_minus_1 = (j - 1) % num_points_y
            j_plus_1 = (j + 1) % num_points_y
            alpha = v * (c[i,j] / c_str - 1)
            beta = v * (1 + c[i,j] / c_str)
            ### first equation terms

            laplacian_c = (c[i_plus_1, j] - 2 * c[i, j] + c[i_minus_1, j]) / (dx ** 2) + (c[i, j_plus_1] - 2 * c[i, j] + c[i, j_minus_1]) / (dy ** 2)
            laplacian_p = (p[i_plus_1, j] - 2 * p[i, j] + p[i_minus_1, j]) / (dx ** 2) + (p[i, j_plus_1] - 2 * p[i, j] + p[i, j_minus_1]) / (dy ** 2) ## also used in 2nd equation
            grad_cnx=(c[i_plus_1, j]*nx[i_plus_1, j] - c[i_minus_1, j]*nx[i_minus_1, j]) / (2*dx)
            grad_cny = (c[i, j_plus_1]*ny[i, j_plus_1] - c[i, j_minus_1]*ny[i, j_minus_1])/(2 * dy)
            grad_px=(p[i_plus_1, j] - p[i_minus_1, j]) / (2*dx) ## also used in 2nd equation
            grad_py=(p[i, j_plus_1] - p[i, j_minus_1])/(2*dy) ## also used in 2nd equation

            grad_cx = (c[i_plus_1, j] - c[i_minus_1, j]) / (2*dx)
            grad_cy = (c[i, j_plus_1] - c[i, j_minus_1]) / (2*dy)

            grad_c_grad_px=(c[i_plus_1,j]*grad_px- c[i_minus_1,j]*grad_py) / (2*dx)
            grad_c_grad_py = (c[i, j_plus_1] * grad_px- c[i, j_minus_1] * grad_py) / (2 * dy)

            ###########################
            #### second equation terms ########
            grad_pnx = (p[i_plus_1, j]*nx[i_plus_1, j] - p[i_minus_1, j]*nx[i_minus_1, j]) / (2 * dx)
            grad_pny = (p[i, j_plus_1]*ny[i, j_plus_1] - p[i, j_minus_1]*ny[i, j_minus_1]) / (2 * dy)


            ##### third term #######
            grad_nx_x=(nx[i_plus_1, j] - nx[i_minus_1, j]) / (2 * dx)
            grad_ny_x = (ny[i_plus_1, j] - ny[i_minus_1, j]) / (2 * dx)
            grad_ny_y =(ny[i, j_plus_1] - ny[i, j_minus_1]) / (2 * dy)
            grad_nx_y = (nx[i, j_plus_1] - nx[i, j_minus_1]) / (2 * dy)
            laplacian_nx = (nx[i_plus_1, j] - 2 * nx[i, j] + nx[i_minus_1, j]) / (dx ** 2) + (nx[i, j_plus_1] - 2 * nx[i, j] + nx[i, j_minus_1]) / (dy ** 2)
            laplacian_ny = (ny[i_plus_1, j] - 2 * ny[i, j] + ny[i_minus_1, j]) / (dx ** 2) + (ny[i, j_plus_1] - 2 * ny[i, j] + ny[i, j_minus_1]) / (dy ** 2)
            ############
            laplacian_psi = (psi[i_plus_1, j] - 2 * psi[i, j] + psi[i_minus_1, j]) / (dx ** 2) + (psi[i, j_plus_1] - 2 * psi[i, j] + psi[i, j_minus_1]) / (dy ** 2)
            grad_psi_x = (psi[i_plus_1, j] - psi[i_minus_1, j]) / (2 * dx)
            grad_psi_y = (psi[i, j_plus_1] - psi[i, j_minus_1]) / (2 * dy)
            ### equations

            dc[i,j] = dt *(Da*laplacian_c-v0*(grad_cnx+grad_cny)+W*(c[i,j]*laplacian_p+grad_cx*grad_px+grad_py*grad_cy))*psi[i,j]
            #dc[i, j] = dt * (Da * laplacian_c - v0 * (grad_cnx + grad_cny) + W * (grad_c_grad_px + grad_c_grad_py)) * psi[ i, j]
            dp[i,j]=dt*(Dm*laplacian_p-v0*(grad_pnx+grad_pny)+kb*(c[i,j]/(c[i,j]+ch))-ku*(p[i,j]))*psi[i,j]
            dnx[i,j]=dt*(-lam*(nx[i,j]*grad_nx_x+ny[i,j]*grad_nx_y)+K*laplacian_nx-eta0*grad_cx+eta*grad_px+(alpha-beta*(nx[i,j]**2+ny[i,j]**2))*nx[i,j])*psi[i,j]
            dny[i,j] = dt * (-lam * (ny[i,j] * grad_ny_x + nx[i,j] * grad_ny_y) + K* laplacian_ny - eta0 * grad_cy  + eta * grad_py + (alpha - beta * (nx[i,j]**2+ny[i,j]**2)) * ny[i,j])*psi[i,j]

            delta = (0.5 + epsilon * (integral_psi - V0) - sig * (nx[i, j] ** 2 + ny[i, j] ** 2))
            dpsi[i, j] = (D_psi * laplacian_psi - kappa*(psi[i, j] * (1 - psi[i, j]) * (delta - psi[i, j])) - gamma* (nx[i, j] * grad_psi_x + ny[i, j] * grad_psi_y)) * dt

    return dc,dp,dnx,dny,dpsi



### disordered IC
theta = np.random.uniform(0, 2 * np.pi, (num_points_x, num_points_y))
nx = np.cos(theta)
ny = np.sin(theta)

### ordered IC
"""""
theta = np.zeros((num_points_x, num_points_y))
nx = np.cos(theta)
ny = np.sin(theta)
"""""
psi = np.zeros((num_points_x, num_points_y))

# Set the elements in the circle around the center to 1
for i in range(num_points_x):
    for j in range(num_points_y):
        if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius ** 2:
            psi[i, j] = 1

Vc_i = np.zeros((num_points_x, num_points_y))

# Set the elements in the circle around the center to 1
for i in range(num_points_x):
    for j in range(num_points_y):
        if (i - center_x) ** 2 + (j - center_y) ** 2 <= (radius) ** 2:
            Vc_i[i, j] = 1


V0=np.sum(psi == 1)

c = (C*(np.ones((num_points_x,num_points_x))+ Csig*np.random.rand(num_points_x,num_points_x)))*Vc_i

p = P*(np.ones((num_points_x,num_points_x))+ Psig*np.random.rand(num_points_x,num_points_x))*Vc_i



for t in range(num_steps):
    dc,dp,dnx,dny,dpsi=time_evolve(nx,ny,c,p,psi,V0)

    #if t<50000:
     #   psi_new=psi+dpsi
    #else:
    psi_new = psi+dpsi
    c_new = (c + dc)
    p_new = (p + dp)
    nx_new = (nx + dnx)
    ny_new = (ny+ dny)
    #psi_new[psi_new<0.001]=0

    """""
    print('T',np.max(k * psi * T), '___________', np.max(va * psi * (divP_x + divP_y)), '_________',np.max(alpha * psi * na))
    print('P',np.max(k * P_psi), '__________', np.max(np.stack((grad_T_x * psi, grad_T_y * psi), axis=-1) * va))
    print('na',np.max(Da * laplacian_na), '__________', np.max(psi*(1 + w * na ** 2)),'__________',np.max(wd *psi* T * na))
    print('change in P',np.mean(P_new)-np.mean(P))
    """""
    c = np.copy(c_new)
    p = np.copy(p_new)
    nx = np.copy(nx_new)
    ny = np.copy(ny_new)
    psi=np.copy(psi_new)
    if t%50000==0:
        print(t)
        """""
        print(t)
        print('dp',np.max(dp))
        print('dc', np.max(dc))
        print('dnx', np.max(dnx))
        print('dny', np.max(dny))
        """""
        x = np.linspace(0, length_x, num_points_x)
        y = np.linspace(0, length_y, num_points_y)
        X, Y = np.meshgrid(x, y)

        # Create the quiver plot for vector field P


        plt.imshow(psi, cmap='Blues', extent=[0, length_x, 0, length_y], origin='lower', aspect='auto')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='T')
        plt.title(f'Scalar p with Gradient Term and Vector Field n (Time Step {t})')

        print('sun:C',np.sum(c))
        print('sun:P', np.sum(p))
        print('sun:psi', np.sum(psi))
        if np.isnan(np.max(c)):
            print('code blew up')
            break



        # Save the plot as an image file in the 'plots' directory
        plt.savefig(f'plots_p/T_and_P_plot_{t}.png')
        # Save the rho matrix values to the CSV file
        with open('rho_field.txt', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(p)

        #plt.show()
        # Clear the current figure to prepare for the next iteration
        plt.clf()

        #plt.quiver(X, Y, c*nx/np.max(c), c*ny/np.max(c), alpha=0.5)
        plt.quiver(X, Y, nx*c, ny*c, alpha=1)
        plt.imshow(c, cmap='jet', extent=[0, length_x, 0, length_y], origin='lower', aspect='auto')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(label='T')
        plt.title(f'Scalar c with Gradient Term and Vector Field n (Time Step {t})')

        # Save the plot as an image file in the 'plots' directory
        plt.savefig(f'plots_c/T_and_C_plot_{t}.png')
        plt.clf()

        # Save the T matrix values to the CSV file
        with open('T_field.txt', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(c)
        # Save the T matrix values to the CSV file
        with open('nx_field.txt', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(nx)
        # Save the T matrix values to the CSV file
        with open('ny_field.txt', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(ny)






# Close any remaining open figures
plt.close()

