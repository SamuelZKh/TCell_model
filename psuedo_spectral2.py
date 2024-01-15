
from matplotlib import cm
import matplotlib.pyplot as plt

# plt.style.use(['science'])
# Choosing the time integration method
method = 'IF'


import numpy as np
from scipy.fft import fft2, ifft2

# Size of the system
N = 50


center_x=N/2
center_y=N/2
L = 5
x = np.linspace(0,L,N)
dx = x[1]-x[0]
# The time step definition
h = 0.00001
T = 1000
alpha=10
beta=1
D=1

eta=25

K1=0.1
#0.2-2
v0=2 #0.1-2
v=10
c_str=1/64


Nsteps = int(T/h)
dframes = 1.0 # time step to output
Nframes = int(T/dframes) #frames to the output
nframes = Nsteps//Nframes
print(Nsteps)

# The array of outputs
c = np.empty((Nframes,N,N), dtype=np.float32)
px=np.empty((Nframes,N,N), dtype=np.float32)
py=np.empty((Nframes,N,N), dtype=np.float32)

# The Fourier variables
c_k = np.empty((N,N), dtype=np.complex64)
px_k = np.empty((N,N), dtype=np.complex64)
py_k = np.empty((N,N), dtype=np.complex64)

kx = np.fft.fftfreq(N, d=dx)*2*np.pi
k = np.array(np.meshgrid(kx , kx ,indexing ='ij'), dtype=np.float32)
k2 = np.sum(k*k,axis=0, dtype=np.float32)

kmax_dealias = kx.max()*2.0/3.0 # The Nyquist mode
# Dealising matrix
dealias = np.array((np.abs(k[0]) < kmax_dealias )*(np.abs(k[1]) < kmax_dealias ),dtype =bool)


# The linear terms of PDE
c_Loperator_k = -k2*D
px_Loperator_k=-K1*k2+v
py_Loperator_k=-K1*k2+v
# The non-linear terms of PDE
def c_Noperator_func(c,px,py):
    return fft2(-v0*(1j*k[0]*c*px+1j*k[1]*c*py))

def px_Noperator_func(c,px,py):
    return -v*fft2((1+c/c_str)*(px**2+py**2)*px)+1j*eta*k[0]*fft2(c)+(v*fft2(c*px/c_str))

def py_Noperator_func(c,px,py):
    return -v*fft2((1+c/c_str)*(px**2+py**2)*px)+1j*eta*k[1]*fft2(c)+(v*fft2(c*px/c_str))
# Defining the time marching operators arrays


# can be calculated once
if method == 'IMEX':
    c_linear_k = 1.0 / (1.0 - h * c_Loperator_k)
    c_non_k = dealias * h / (1.0 - h * c_Loperator_k)

    px_linear_k = 1.0 / (1.0 - h * px_Loperator_k)
    px_non_k = dealias * h / (1.0 - h * px_Loperator_k)

    py_linear_k = 1.0 / (1.0 - h * py_Loperator_k)
    py_non_k = dealias * h / (1.0 - h * py_Loperator_k)
elif method == 'IF':
    c_linear_k = np.exp(h*c_Loperator_k )
    c_non_k = dealias*h*c_linear_k

    px_linear_k = np.exp(h * px_Loperator_k)
    px_non_k = dealias * h * px_linear_k

    py_linear_k = np.exp(h * py_Loperator_k)
    py_non_k = dealias * h * py_linear_k







# Initial condition
rng = np.random.default_rng(2)
noise = 0.01
c0 = 0.1
dtheta=np.pi/2

theta=dtheta*rng.uniform(size=px[0].shape)
px[0] = np.cos(theta)
py[0] = np.sin(theta)

print(px[0])

c[0] = c0 +noise*rng.standard_normal(c[0].shape)

c_Noperator_k = c_k.copy() # auxiliary array
px_Noperator_k = px_k.copy() # auxiliary array
py_Noperator_k = py_k.copy() # auxiliary array


c_n = c[0].copy() # auxiliary array
px_n = px[0].copy() # auxiliary array
py_n = py[0].copy() # auxiliary array

c_k[:] = fft2(c[0]) # FT initial condition
px_k[:] = fft2(px[0]) # FT initial condition
py_k[:] = fft2(py[0]) # FT initial condition
# time evolution loop

for t in range(0,Nsteps):


    # calculate the nonlinear operator (with dealising)
    c_Noperator_k[:] = c_Noperator_func(c_n,px_n,py_n)
    px_Noperator_k[:] = px_Noperator_func(c_n,px_n,py_n)
    py_Noperator_k[:] = py_Noperator_func(c_n,px_n,py_n)
    # updating in time
    c_k[:] = c_k*c_linear_k + c_Noperator_k*c_non_k
    px_k[:] = px_k*px_linear_k + px_Noperator_k*px_non_k
    py_k[:] = py_k * py_linear_k + py_Noperator_k * py_non_k
    # IFT to next step
    c_n[:] = ifft2(c_k).real
    px_n[:] = ifft2(px_k).real
    py_n[:] = ifft2(py_k).real
    # test to output

    c[t//nframes] = c_n

    px[t // nframes] = px_n
    py[t // nframes] = py_n

    if t%1000==0:
        print(t)
        x = np.linspace(0, N, N)
        y = np.linspace(0, N, N)
        X, Y = np.meshgrid(x, y)

        #Create the quiver plot for vector field P
        plt.quiver(X, Y, px[t//nframes]*c[t//nframes],py[t//nframes]*c[t//nframes], alpha=0.4)

        plt.imshow(c[t//nframes], cmap='jet', extent=[0, N, 0, N], origin='lower', aspect='auto')

        plt.xlabel('x')
        plt.ylabel('y')
        print('sun:c', np.sum(c[t//nframes]))

        if np.isnan(np.max(c)):
            print('code blew up')
            break

        # Save the plot as an image file in the 'plots' directory
        plt.savefig(f'plots_p/T_and_P_plot_{t}.png')
        plt.clf()