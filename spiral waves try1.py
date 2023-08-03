from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from scipy.spatial import ConvexHull
from scipy.interpolate import UnivariateSpline

N=200

grid=np.zeros((N,N))
center = grid.shape[0] // 2

center=[100,100]
sim_steps=100
patch_coordinate_cm=center
patch_size=10
patch_size=100

center_fixed=100
grid_store=[]
r=1
dr=0.4
d_theta=0.1
x_list=[]
y_list=[]
pos_time=[]
TCR_count=100
particle_store=[]
particle_bound_status=[]
for i in range(TCR_count):
    angle = random.uniform(0, 2 * np.pi)  # Random angle for each particle

    radius = random.uniform(0,r + 30)  # Random radius within the donut

    x = int(center_fixed + radius * np.cos(angle))
    y = int(center_fixed + radius * np.sin(angle))
    particle_store.append([x, y])
    particle_bound_status.append(0)


x = [p[1] for p in  particle_store]
y = [p[0] for p in  particle_store]
plt.scatter(x, y, color='blue',s=1)
particle_store_time=[]
for t in range(sim_steps):

    grid_store.append(grid)
    x,y=patch_coordinate_cm
    for p in range(len(particle_store)):

        xp=(particle_store[p][0])
        yp = (particle_store[p][1])
        if np.sqrt((x-xp)**2+(y-yp)**2)<patch_size:
            particle_bound_status[p]==1



    r=r+dr
    angle=t*d_theta

    x=x+r*np.cos(angle)
    y=y+r*np.sin(angle)

    for p in range(len(particle_store)):

        xp = (particle_store[p][0])
        yp = (particle_store[p][1])
        if particle_bound_status[p] == 1:
            xp = xp + r*np.cos(angle)
            yp=yp+r*np.sin(angle)
            particle_store[p]==[xp,yp]

    particle_store_time.append(particle_store)

    x_list.append(x)
    y_list.append(y)
    pos_time.append([x,y])

plt.plot(x_list,y_list)
plt.show()






