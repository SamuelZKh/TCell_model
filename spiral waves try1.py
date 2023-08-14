import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
import copy
patch_area=0.2
sim_time=200
dr= 0.01
d_theta= 0.2
frame_threshold1 = 50  # Number of frames after which the second patch appears
frame_threshold2 = 80
new_patch_offset = 100  # Offset for the new patch of particle1s to distinguish them
D=0.001
num_particles = 30
radius_circle = 3
N=200
num_TCR=1000
x_grid=np.linspace(-3,3,100)
y_grid=np.linspace(-3,3,100)
grid=np.zeros((N,N))

def random_polar_coordinates(radius):
    r = np.sqrt(np.random.uniform(0, radius**2))
    theta = np.random.uniform(0, 2 * np.pi)
    return r, theta

# Function to generate random direction for particle2
def random_angle():
    return np.random.uniform(0, 2 * np.pi)

# Function to generate random polar coordinates within the circle's radius

# Define the number of particle1s and particle2s

# Initialize particle1 positions


particle_1_initial=[]
for i in range(num_particles):
    angle = np.random.uniform(0,np.pi)
    distance = np.random.uniform(0.1,0.3)
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)
    particle_1_initial.append([x, y])

particle1_save=copy.copy(particle_1_initial)



particle2 = [[np.random.uniform(-10, 10), np.random.uniform(-10, 10)] for _ in range(num_TCR)]

wave_frequency=np.linspace(0,sim_time,3)

def random_angle():
    return np.random.uniform(0, 2 * np.pi)


particle_movement_type=[0 for _ in range(len(particle1_save))]
time_to_add_particles=400
particle1_save=[]
particle_movement_type=[]

r_store=[]
def update(frame):
    particle_1_initial = []

    print(frame)

    global particle1_positions, particle2
    if frame<num_particles:
        particle1_save.append([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])
        particle_movement_type.append(0)

    particle1_positions = []
    # Update particle1 positions
    if frame == frame_threshold1:


        for _ in range(num_particles):
            particle1_save.append([random.uniform(-0.2,0.2),random.uniform(-0.2,0.2)])
            particle_movement_type.append(1)
    if frame == frame_threshold2:


        for _ in range(num_particles):
            particle1_save.append([random.uniform(-0.2,0.2),random.uniform(-0.2,0.2)])
            particle_movement_type.append(2)

    for i in range(len(particle1_save)):

        xi=particle1_save[i][0]
        yi=particle1_save[i][1]
        # Move particle1 in a spiral
        if particle_movement_type[i]==0:
            r = dr * frame
            angle = d_theta * frame
        if particle_movement_type[i]==1:
            r = dr * (frame-frame_threshold1)
            angle = d_theta * (frame-frame_threshold1)
        if particle_movement_type[i] == 2:
            r = dr * (frame - frame_threshold2)
            angle = d_theta * (frame - frame_threshold2)
        x1 =xi+ r * np.cos(angle)+np.random.normal(0, 0.02)
        y1 =yi+r * np.sin(angle)+np.random.normal(0, 0.02)
        frac=0.7
        x2 = xi + r*frac * np.cos(angle*frac) + np.random.normal(0, 0.02)
        y2 = yi + r*frac * np.sin(angle*frac) + np.random.normal(0, 0.02)
        """""
        random_no=random.uniform(0,1)
        if np.sqrt((x1) ** 2 + (y1) ** 2) < radius_circle and random_no<0.3:
            particle1_positions.append([x1, y1])
        elif np.sqrt((x1) ** 2 + (y1) ** 2) < radius_circle and random_no >=0.3 and random_no<0.6 :
            particle1_positions.append([x2, y2])
        elif np.sqrt((x1) ** 2 + (y1) ** 2) < radius_circle and random_no >=0.6 :
            particle1_positions.append([xi, yi])
        """""
        if np.sqrt((x1) ** 2 + (y1) ** 2) < radius_circle :
            particle1_positions.append([x1, y1])
        if np.sqrt((x1) ** 2 + (y1) ** 2) < radius_circle :
            particle1_positions.append([xi, yi])
        particle1_save[i][0]=x1
        particle1_save[i][1]=y1
        if frame==sim_time:
            break





    # Update particle2 positions
    r_t=[]
    for j in range(len(particle2)):
        r_t.append(np.sqrt(x1 ** 2 + y1 ** 2))
    r_store.append(np.mean(r_t))
    for i in range(len(particle1_save)):
        x1 = particle1_save[i][0]
        y1 = particle1_save[i][1]


        for j in range(len(particle2)):
            x2, y2 = particle2[j]

            dx = x1 - x2
            dy = y1 - y2
            dist = np.sqrt(dx**2 + dy**2)
            dist_center=np.sqrt(x2**2 + y2**2)
            if dist < patch_area:
            # print('move')

                # Calculate the attraction force based on the distance
                attraction_strength = 5
                force_x = attraction_strength * dx
                force_y = attraction_strength * dy

                # Move particle2 in the direction of particle1 with attraction force
                if np.sqrt((x2+force_x)**2 + (y2+force_y)**2)<=radius_circle:
                    particle2[j] = [x2+force_x,y2+force_y]

            elif dist > patch_area and dist_center<=radius_circle:
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


particle2 = [random_polar_coordinates(radius_circle-1) for _ in range(num_TCR)]
particle2 = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in particle2]

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-radius_circle, radius_circle)
ax.set_ylim(-radius_circle, radius_circle)

# Create particle1 as a red dot and particle2 as blue dots
particle1, = ax.plot(x1, y1, 'ro', markersize=5)
particle2_dots, = ax.plot([], [], 'bo', markersize=3, linestyle='')




def plot_circle(center, radius):
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('spiral wave')
    plt.grid(True)
    plt.axis('equal')  # Set equal scaling for x and y axes to make the circle look circular
    plt.show()
# Create the animation
ani = animation.FuncAnimation(fig, update, frames=sim_time, interval=50)


# Example usage
center = [0, 0]  # Center coordinates of the circle
radius = radius_circle+0.5 # Radius of the circle
plot_circle(center, radius)
ani.save('grid_movie.gif', writer='pillow')

# Alternatively, show the animation in a pop-up window
plt.show()

#plt.show()
plt.plot(r_store)
plt.title('Hypothesis1_spiral')
plt.xlabel('time')
plt.ylabel('Average radial Distance of TCR')

plt.show()
