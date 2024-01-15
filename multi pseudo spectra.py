
from matplotlib import cm
import matplotlib.pyplot as plt

# plt.style.use(['science'])
# Choosing the time integration method
method = 'IMEX'


import numpy as np
from scipy.fft import fft2, ifft2
# Cahn-Hilliard model constants
W = 1.0
M = 1.0 # mobility
kappa = 0.1 #gradient coeficient
# Size of the system
N = 2**8 # 2**8 = 256


center_x=N/2
center_y=N/2
L = 16*np.pi
x = np.linspace(0,L,N)
dx = x[1]-x[0]
# The time step definition
h = 0.01
T = 1000
alpha=1
D=0.2
t2=0.4
t1=0.1
gamma=0.9
beta=1.3
epsilon=1
sig=1
radius=20

Nsteps = int(T/h)
dframes = 1.0 # time step to output
Nframes = int(T/dframes) #frames to the output
nframes = Nsteps//Nframes
print(Nsteps)

# The array of outputs
psi = np.empty((Nframes,N,N), dtype=np.float32)
px=np.empty((Nframes,N,N), dtype=np.float32)
py=np.empty((Nframes,N,N), dtype=np.float32)

# The Fourier variables
psi_k = np.empty((N,N), dtype=np.complex64)
px_k = np.empty((N,N), dtype=np.complex64)
py_k = np.empty((N,N), dtype=np.complex64)

kx = np.fft.fftfreq(N, d=dx)*2*np.pi
k = np.array(np.meshgrid(kx , kx ,indexing ='ij'), dtype=np.float32)
k2 = np.sum(k*k,axis=0, dtype=np.float32)

kmax_dealias = kx.max()*2.0/3.0 # The Nyquist mode
# Dealising matrix
dealias = np.array((np.abs(k[0]) < kmax_dealias )*(np.abs(k[1]) < kmax_dealias ),dtype =bool)


# The linear terms of PDE
psi_Loperator_k = D*k2
px_Loperator_k=D*k2-t1
py_Loperator_k=D*k2-t1
# The non-linear terms of PDE
def psi_Noperator_func(psi,px,py,delta):
    return fft2(-(1-psi)*(delta-psi)*psi-alpha*1j*(k[0]*px+k[1]*py))

def px_Noperator_func(psi,px,py):
    return fft2(t2*(1-psi**2)*px+gamma*(k[0]*px)*px)-beta*k[1]

def py_Noperator_func(psi,px,py):
    return fft2(t2*(1-psi**2)*py+gamma*(k[1]*py)*py)-beta*k[0]
# Defining the time marching operators arrays




psi_linear_k = 1.0/(1.0-h*psi_Loperator_k)
psi_non_k = dealias*h/(1.0-h*psi_Loperator_k)

px_linear_k = 1.0 / (1.0 - h * px_Loperator_k)
px_non_k = dealias * h / (1.0 - h * px_Loperator_k)

py_linear_k = 1.0 / (1.0 - h * py_Loperator_k)
py_non_k = dealias * h / (1.0 - h * py_Loperator_k)





# Initial condition
rng = np.random.default_rng(12345)
noise = 0.02
psi0 = 0.6
dtheta=0



# Set the elements in the circle around the center to 1psi[0] = psi0 +noise*rng.standard_normal(psi[0].shape)
for i in range(N):
    for j in range(N):
        if (i - center_x) ** 2 + (j - center_y) ** 2 <= (radius) ** 2:
            psi[0][i, j] = 1

V0 = np.sum(psi[0] >= 0.65)
theta=dtheta*rng.standard_normal(px[0].shape)
px[0] = np.cos(theta)
py[0] = np.sin(theta)



psi_Noperator_k = psi_k.copy() # auxiliary array
px_Noperator_k = px_k.copy() # auxiliary array
py_Noperator_k = py_k.copy() # auxiliary array


psi_n = psi[0].copy() # auxiliary array
px_n = px[0].copy() # auxiliary array
py_n = py[0].copy() # auxiliary array

psi_k[:] = fft2(psi[0]) # FT initial condition
px_k[:] = fft2(px[0]) # FT initial condition
py_k[:] = fft2(py[0]) # FT initial condition
# time evolution loop

for t in range(1,Nsteps):
    print(psi_n)
    integral_psi = np.sum(psi_n >= 0.65)
    delta = (0.5 + epsilon * (integral_psi - V0) - sig * (px_n ** 2 + py_n ** 2))
    # calculate the nonlinear operator (with dealising)
    psi_Noperator_k[:] = psi_Noperator_func(psi_n,px_n,py_n,delta)
    px_Noperator_k[:] = px_Noperator_func(psi_n,px_n,py_n)
    py_Noperator_k[:] = py_Noperator_func(psi_n,px_n,py_n)
    # updating in time
    psi_k[:] = psi_k*psi_linear_k + psi_Noperator_k*psi_non_k
    px_k[:] = px_k*px_linear_k + px_Noperator_k*px_non_k
    py_k[:] = py_k * py_linear_k + py_Noperator_k * py_non_k
    # IFT to next step
    psi_n[:] = ifft2(psi_k).real
    px_n[:] = ifft2(px_k).real
    py_n[:] = ifft2(py_k).real
    # test to output

    psi[t//nframes] = psi_n

    px[t // nframes] = px_n
    py[t // nframes] = py_n

    if t%1000==0:
        print(t)
        x = np.linspace(0, N, N)
        y = np.linspace(0, N, N)
        X, Y = np.meshgrid(x, y)

        #Create the quiver plot for vector field P
        plt.quiver(X, Y, px[t//nframes]*psi[t//nframes],py[t//nframes]*psi[t//nframes], alpha=0.3)

        plt.imshow(psi[t//nframes], cmap='Blues', extent=[0, N, 0, N], origin='lower', aspect='auto')

        plt.xlabel('x')
        plt.ylabel('y')
        print('sun:p', np.sum(psi))

        if np.isnan(np.max(psi)):
            print('code blew up')
            break

        # Save the plot as an image file in the 'plots' directory
        plt.savefig(f'plots_p/T_and_P_plot_{t}.png')
        plt.clf()