import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
import copy
patch_area=0.5
sim_time=1000
dr= 0.0006
d_theta= 0.1
frame_threshold1 = 300  # Number of frames after which the second patch appears
frame_threshold2 = 600
new_patch_offset = 100  # Offset for the new patch of particle1s to distinguish them
D=0.01


def random_polar_coordinates(radius):
    r = np.sqrt(np.random.uniform(0, radius**2))
    theta = np.random.uniform(0, 2 * np.pi)
    return r, theta

# Function to generate random direction for particle2
def random_angle():
    return np.random.uniform(0, 2 * np.pi)

# Function to generate random polar coordinates within the circle's radius

# Define the number of particle1s and particle2s
num_particles = 50
radius_circle = 10
# Initialize particle1 positions


particle_1_initial=[]
for i in range(num_particles):
    xi = np.random.normal(0, 0.2)
    yi = np.random.normal(0, 0.2)
    particle_1_initial.append([xi, yi])
num_TCR=100


particle2 = [[np.random.uniform(-10, 10), np.random.uniform(-10, 10)] for _ in range(num_TCR)]

wave_frequency=np.linspace(0,sim_time,3)

def random_angle():
    return np.random.uniform(0, 2 * np.pi)


particle1_save=copy.copy(particle_1_initial)
particle1_save2=copy.copy(particle_1_initial)
particle1_save3=copy.copy(particle_1_initial)

def update(frame):
    print(frame)
    global particle1_positions, particle2
    particle1_positions = []
    # Update particle1 positions

    if frame >= frame_threshold1:
        x_fix_new = []
        y_fix_new = []

        for i in range(num_particles):
            randomx = np.random.normal(0, 0.2)
            randomy = np.random.normal(0, 0.2)
            x_fix_new.append(randomx)
            y_fix_new.append(randomy)

    x_fix = []
    y_fix = []

    for i in range(num_particles):
        randomx = np.random.normal(0, 0.2)
        randomy = np.random.normal(0, 0.2)
        x_fix.append(randomx)
        y_fix.append(randomy)
    xy_pos=[]
    for i in range(num_particles):

        xi=particle1_save[i][0]
        yi=particle1_save[i][1]
        # Move particle1 in a spiral
        r = dr * frame
        angle = d_theta * frame
        x1 =xi+ r * np.cos(angle)+np.random.normal(0, 0.02)
        y1 =yi+r * np.sin(angle)+np.random.normal(0, 0.02)





        random_no=random.uniform(0,1)
        if np.sqrt((x1) ** 2 + (y1) ** 2) < radius_circle :
            particle1_positions.append([x1, y1])

        particle1_save[i][0]=x1
        particle1_save[i][1] = y1

        if frame >= frame_threshold1:
            r_new = dr * (frame - frame_threshold1)
            xi = particle1_save[i][0]
            yi = particle1_save[i][1]
            # Move particle1 in a spiral
            r = dr * frame
            angle = d_theta * frame
            x1 = xi + r_new * np.cos(angle) + np.random.normal(0, 0.02)
            y1 = yi + r_new * np.sin(angle) + np.random.normal(0, 0.02)

            random_no = random.uniform(0, 1)
            if np.sqrt((x1) ** 2 + (y1) ** 2) < radius_circle:
                particle1_positions.append([x1, y1])

            particle1_save[i][0] = x1
            particle1_save[i][1] = y1


        if frame >= frame_threshold2:
            r_new = dr * (frame - frame_threshold1)
            xi = particle1_save3[i][0]
            yi = particle1_save3[i][1]
            # Move particle1 in a spiral
            r = dr * frame
            angle = d_theta * frame
            x1 = xi + r_new * np.cos(angle) + np.random.normal(0, 0.02)
            y1 = yi + r_new * np.sin(angle) + np.random.normal(0, 0.02)

            random_no = random.uniform(0, 1)
            if np.sqrt((x1) ** 2 + (y1) ** 2) < radius_circle:
                particle1_positions.append([x1, y1])

            particle1_save3[i][0] = x1
            particle1_save3[i][1] = y1



    # Update particle2 positions
    for i in range(len(particle1_positions)):
        x1, y1 = particle1_positions[i]
        for j in range(len(particle2)):
            x2, y2 = particle2[j]
            dx = x1 - x2
            dy = y1 - y2
            dist = np.sqrt(dx**2 + dy**2)
            if dist < patch_area:
            # print('move')

                # Calculate the attraction force based on the distance
                attraction_strength = 3
                force_x = attraction_strength * dx
                force_y = attraction_strength * dy

                # Move particle2 in the direction of particle1 with attraction force
                particle2[j] = [x1,y1]

            else:
                # If not, move particle2 in a random direction
                angle2 = random_angle()
                dx2 = D * np.cos(angle2)
                dy2 = D * np.sin(angle2)
                particle2[j] = [x2 + dx2, y2 + dy2]

    # Update particle1 data for all particle1s
    if len(particle1_positions)>0:
        particle1.set_data(*zip(*particle1_positions))

    # Update particle2_dots data
    particle2_dots.set_data(*zip(*particle2))

    return particle1, particle2_dots

# Initialize particle1 position
x1, y1 = 3, 0

# Initialize particle2 positions uniformly inside a circle


particle2 = [random_polar_coordinates(radius_circle) for _ in range(num_TCR)]
particle2 = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in particle2]

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-radius_circle, radius_circle)
ax.set_ylim(-radius_circle, radius_circle)

# Create particle1 as a red dot and particle2 as blue dots
particle1, = ax.plot(x1, y1, 'ro', markersize=2)
particle2_dots, = ax.plot([], [], 'bo', markersize=1, linestyle='')

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=sim_time, interval=10, blit=True)

plt.show()