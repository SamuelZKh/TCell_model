
from matplotlib import cm
import matplotlib.pyplot as plt

# plt.style.use(['science'])
# Choosing the time integration method
method = 'IF'


import numpy as np
from scipy.fft import fft2, ifft2
# Cahn-Hilliard model constants
W = 1.0
M = 1.0 # mobility
kappa = 0.1 #gradient coeficient
# Size of the system
N = 2**8 # 2**8 = 256
L = 16*np.pi
x = np.linspace(0,L,N)
dx = x[1]-x[0]
# The time step definition
h = 0.01
T = 1500
Nsteps = int(T/h)
dframes = 1.0 # time step to output
Nframes = int(T/dframes) #frames to the output
nframes = Nsteps//Nframes
print(Nsteps)

# The array of outputs
n = np.empty((Nframes,N,N), dtype=np.float32)
# The Fourier variables
n_k = np.empty((N,N), dtype=np.complex64)
kx = np.fft.fftfreq(N, d=dx)*2*np.pi
k = np.array(np.meshgrid(kx , kx ,indexing ='ij'), dtype=np.float32)


k2 = np.sum(k*k,axis=0, dtype=np.float32)
kmax_dealias = kx.max()*2.0/3.0 # The Nyquist mode
# Dealising matrix
dealias = np.array((np.abs(k[0]) < kmax_dealias )*(np.abs(k[1]) < kmax_dealias ),dtype =bool)


# The linear terms of PDE
Loperator_k = -M*(kappa*k2**2+2*W*k2)
# The non-linear terms of PDE
def Noperator_func(n):
    return -2*M*W*k2*fft2(-3*n**2+2*n**3)
# Defining the time marching operators arrays



# can be calculated once
if method == 'IMEX':
    Tlinear_k = 1.0/(1.0-h*Loperator_k)
    Tnon_k = dealias*h/(1.0-h*Loperator_k)
elif method == 'IF':
    Tlinear_k = np.exp(h*Loperator_k)
    Tnon_k = dealias*h*Tlinear_k
elif method == 'ETD':
    Tlinear_k = np.exp(h*Loperator_k)
    def myexp(x):
        if x == 1:
            return 1.0
        else:
            return (x-1.0)/np.log(x)
    vmyexp = np.vectorize(myexp) # vectorize myexp (could be jitted)
    Tnon_k = dealias*h*vmyexp(Tlinear_k)

else: print('ERROR: Undefined Integrator')


# Initial condition
rng = np.random.default_rng(12345)
noise = 0.02
n0 = 0.6
n[0] = n0 +noise*rng.standard_normal(n[0].shape)
Noperator_k = n_k.copy() # auxiliary array
nn = n[0].copy() # auxiliary array
n_k[:] = fft2(n[0]) # FT initial condition
# time evolution loop
for i in range(1,Nsteps):
    if i%10000==0:
        print(i)
    # calculate the nonlinear operator (with dealising)
    Noperator_k[:] = Noperator_func(nn)
    # updating in time
    n_k[:] = n_k*Tlinear_k + Noperator_k*Tnon_k
    # IFT to next step
    nn[:] = ifft2(n_k).real
    # test to output
    if (i % nframes) == 0:
        n[i//nframes] = nn
    if i%1000==0:
        print(i)
        x = np.linspace(0, N, N)
        y = np.linspace(0, N, N)
        X, Y = np.meshgrid(x, y)

        #Create the quiver plot for vector field P


        plt.imshow(n[i//nframes], cmap='Blues', extent=[0, N, 0, N], origin='lower', aspect='auto')

        plt.xlabel('x')
        plt.ylabel('y')


        # Save the plot as an image file in the 'plots' directory
        plt.savefig(f'plots_p/T_and_P_plot_{i}.png')
        plt.clf()


j=10
plt.imshow(n[j],cmap='RdBu_r')

plt.xticks([])
plt.yticks([])
plt.show()
