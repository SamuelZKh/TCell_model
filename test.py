from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from scipy.spatial import ConvexHull
from PIL import Image
grid_size = 400
initial_r = 20
initial_dr = 0.1
grid = np.zeros([grid_size, grid_size])

tot_actin = 1000
initial_length = 6
actin_store = []
sim_time = 100
polymerise_prob = 0.4
depolymeris_prob = 0.3

TCR_bind_prob=1
TCR_unbind_prob=0.2
polymer_size=5

no_neighbour=2
center = grid.shape[0] // 2
polymerisation_prob_list=[]
print(center)

boundary_points = []

particle_count = 150  # Number of particles
particle_radius = 2 # Radius of the particle
particle_bound_status=[]
particle_store = []
for i in range(particle_count):
    angle = random.uniform(0, 2 * np.pi)  # Random angle for each particle
    radius = random.uniform(initial_r, initial_r+20)  # Random radius within the donut
    x = int(center + radius * np.cos(angle))
    y = int(center + radius * np.sin(angle))
    particle_store.append([x, y])
    particle_bound_status.append(0)

# Update the outermost points for each frame
def update_boundary(frame):
    current_frame = grid_store[frame]
    outer_points = np.argwhere(current_frame == 1)
    if len(outer_points) > 0:
        hull = ConvexHull(outer_points)
        boundary_points.append(outer_points[hull.vertices])




## initialization
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if np.sqrt((i - center)**2 + (j - center)**2) >= initial_r - initial_dr and np.sqrt((i - center)**2 + (j - center)**2) <= initial_r + initial_dr:
            #grid[i, j] = 1
            actin_coord = []
            # grow initial branch
            for k in range(1, initial_length):
                x = i + k * (i - center) // initial_r
                y = j + k * (j - center) // initial_r
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    #grid[x, y] = 1
                    actin_coord.append([x, y])
            actin_store.append(actin_coord)
print(actin_store[0])
print(len(actin_store))
# time evolution
grid_store=[]
particle_time=[]
for t in range(sim_time):
    print('time',t)
    polymerise_prob = 0.4
    depolymeris_prob = 0.3
    grid_time = np.zeros([grid_size, grid_size])
    new_actin_store = []
    branch_polymer = []
    for i,branch in enumerate(actin_store) :
        neighbour_lengths = []

        for j in range(-no_neighbour, no_neighbour + 1):
            if t%3==0:
                if j != 0:
                    if i + j >= 0 and i + j < len(actin_store):
                        neighbour_lengths.append(len(actin_store[i + j]))

                current_length = len(branch)
                avg_neighbour_length = np.mean(neighbour_lengths) if neighbour_lengths else current_length
                length_difference = current_length - avg_neighbour_length

                # Calculate the curvature based on the length difference
                if current_length!=0:
                    curvature = length_difference / current_length

                    # Apply the restoring force if the curvature is positive (current polymer is longer)
                    if curvature > 0:
                        restoring_force = 0.1 * curvature
                        polymerise_prob -= restoring_force

        polymerise = random.uniform(0, 1)
        branch_polymer.append(polymerise)
        if polymerise < polymerise_prob and tot_actin>0:
            if len(branch)!=0:
                last_point = branch[-1]
                x, y = last_point
                # grow radially from the last point
                for k in range(1,polymer_size):

                    dx = k * (x - center) // initial_r
                    dy = k * (y - center) // initial_r
                    new_x = x + dx
                    new_y = y + dy
                    if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                        #grid[new_x, new_y] = 1
                        branch.append([new_x, new_y])
                        tot_actin-=1
                        actin_store[i]=branch

        depolymerise = random.uniform(0, 1)
        if depolymerise < depolymeris_prob :
            actin_store[i] = actin_store[i][polymer_size:]
            tot_actin += 1
    for i in range(len(actin_store)):
        for y in actin_store[i]:
            grid_time[y[0], y[1]] = 1
    grid_store.append(grid_time)
    #actin_store = new_actin_store



    particle_movement_factor = 0.1  # Controls the movement of the particles
    particle_in_time=[n for n in particle_store]
    for j, particle in enumerate(particle_store):
        x, y = particle
        for k, branch in enumerate(actin_store):
            for m in range(len(branch) - 1):
                monomer_x, monomer_y = branch[m]

                if np.sqrt((x - monomer_x) ** 2 + (y - monomer_y) ** 2) <= particle_radius and particle_bound_status[
                    j] == 0:
                    particle[0] = monomer_x
                    particle[1] = monomer_y
                    particle_in_time[j] = [monomer_x, monomer_y]
                    particle_bound_status[j] = 1

                if particle_bound_status[j] == 1 and branch_polymer[k] < polymerise_prob:
                    particle[0] = monomer_x
                    particle[1] = monomer_y
                    particle_in_time[j] = [particle[0], particle[1]]

        if particle_bound_status[j] == 0:
            dx = x - center
            dy = y - center
            magnitude = np.sqrt(dx ** 2 + dy ** 2)
            if magnitude > particle_radius:
                particle[0] -= particle_movement_factor * dx / magnitude
                particle[1] -= particle_movement_factor * dy / magnitude
                particle_in_time[j] = [particle[0], particle[1]]

    particle_store = particle_in_time
    particle_time.append(particle_in_time)
    # Plot the particles
print(actin_store[0])


plt.imshow(grid)
plt.show()

# Function to update the plot for each frame
# Function to update the plot for each frame
# Function to update the plot for each frame
def update_frame(frame):
    plt.cla()  # Clear the current plot
    plt.imshow(grid_store[frame], cmap='binary')  # Plot the grid at the current frame
    #plt.imshow(particle_time[frame])  # Plot the grid at the current frame

    particles = particle_time[frame]
    x = [p[1] for p in particles]
    y = [p[0] for p in particles]
    plt.scatter(x, y, color='red', s=1)

    plt.title('Frame {}'.format(frame))
    plt.axis('off')  # Turn off the axis
    plt.title('Frame {}'.format(frame))
    plt.axis('off')  # Turn off the axis

    # Plot the boundary
    if frame < len(boundary_points):
        boundary = boundary_points[frame]
        boundary_x = [p[1] for p in boundary]
        boundary_y = [p[0] for p in boundary]
        plt.plot(boundary_x, boundary_y, color='red')

    # Update the particle positions

print(particle_store)
# Create a figure and axes for the plot
fig, ax = plt.subplots()

# Create the animation
anim = animation.FuncAnimation(fig, update_frame, frames=len(grid_store), interval=200)

# Update the boundary points for each frame
for frame in range(len(grid_store)):
    update_boundary(frame)

# Save the animation as a GIF
anim.save('grid_movie.gif', writer='pillow')

# Alternatively, show the animation in a pop-up window
plt.show()