from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from scipy.spatial import ConvexHull
from scipy.interpolate import UnivariateSpline
from PIL import Image
def interpolate_boundary_1(x, y, smoothing_factor):
    # Fit a smoothed curve to the boundary
    t = np.arange(len(x))
    spl = UnivariateSpline(t, x, k=smoothing_factor)
    inter_x = spl(t)
    spl = UnivariateSpline(t, y, k=smoothing_factor)
    inter_y = spl(t)


    return  inter_x, inter_y


def interpolate_boundary(x, y, smoothing_factor):
    num_points=1000
    # Fit a smoothed curve to the boundary
    t = np.linspace(0, len(x) - 1, num_points)
    spl = UnivariateSpline(np.arange(len(x)), x, k=smoothing_factor)
    inter_x = spl(t)
    spl = UnivariateSpline(np.arange(len(y)), y, k=smoothing_factor)
    inter_y = spl(t)

    return inter_x, inter_y
np.random.seed(0)

grid_size = 400
initial_r = 80
initial_dr =0.6
grid = np.zeros([grid_size, grid_size])
max_polymer_radius = 100  # Set the maximum radius for polymerization to stop

tot_actin = 3000
initial_length = 20
actin_store = []
sim_time =150

TCR_bind_prob = 1
TCR_unbind_prob = 0.2
polymer_size = 4
D = 0.5
no_neighbour = 2
V=4
polymerise_region = 8
center = grid.shape[0] // 2
print('cc',center)
polymerisation_prob_list = []
print(center)

boundary_points = []

TCR_count = 200  # Number of particles
particle_radius = 0.7  # Radius of the particle
particle_bound_status = []
particle_store = []
actin_pool_positions = []


def delete_element(lst, i):
    if i < 0 or i >= len(lst):
        raise IndexError("Invalid index")

    del lst[i]
    return lst


for i in range(TCR_count):
    angle = random.uniform(0, 2 * np.pi)  # Random angle for each particle
    radius = random.uniform(initial_r + 10, initial_r + 50)  # Random radius within the donut
    x = int(center + radius * np.cos(angle))
    y = int(center + radius * np.sin(angle))
    particle_store.append([x, y])
    particle_bound_status.append(0)


# Update the outermost points for each frame
def update_boundary(frame):
    current_frame = grid_store[frame]
    inner_points = np.argwhere(current_frame == 1)
    if len(inner_points) > 0:
        hull = ConvexHull(inner_points)
        boundary_points.append(inner_points[hull.vertices])


F_initial_store = []
## initialization
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if np.sqrt((i - center) ** 2 + (j - center) ** 2) >= initial_r - initial_dr and np.sqrt(
                (i - center) ** 2 + (j - center) ** 2) <= initial_r + initial_dr:
            grid[i, j] = 1
            actin_coord = []
            F_initial_n = []
            # grow initial branch
            for k in range(1, initial_length):
                x = i + k * (i - center) // initial_r
                y = j + k * (j - center) // initial_r
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    # grid[x, y] = 1
                    actin_coord.append([x, y])
                    F_initial_n.append([x, y])
            actin_store.append(actin_coord)
            F_initial_store.append(actin_coord)

print(actin_store)
print(actin_store[0])
print(len(actin_store))
actin_pool_status = []
for i in range(tot_actin - sum(int(len(sublist) / polymer_size) for sublist in actin_store)):
    angle = random.uniform(0, 2*np.pi)  # Random angle for each particle
    radius = random.uniform(initial_r, initial_r + 40)  # Random radius within the donut
    x = int(center + radius * np.cos(angle))
    y = int(center + radius * np.sin(angle))

    actin_pool_positions.append([x, y])
    actin_pool_status.append(0)
# time evolution
grid_store = []
particle_time = []
actin_pool_time = []

actin_last_store = []
actin_first_store = []

branch_bound = [0 for _ in particle_store]
boundary_store=[]

branch_type=[]
for _ in range(len(actin_store)):
    rand_no=random.uniform(0,1)
    if rand_no<0.5:
        branch_type.append(0)
    else:
        branch_type.append(1)


for t in range(sim_time):
    print('time', t)
    # polymerise_prob = 0
    # depolymeris_prob = 0.5
    shortening = 0.05
    grid_time = np.zeros((grid_size, grid_size))
    new_actin_store = []
    branch_polymer = []
    branch_polymer_prob = []
    actin_last_time = []
    actin_first_time = []
    actin_pool_now = list(actin_pool_positions)

    for k, mono in enumerate(actin_pool_now):
        if actin_pool_status[k]==0 or actin_pool_status[k]==3:
            mono[0] += np.random.normal(0, D)
            mono[1] += np.random.normal(0, D)
            actin_pool_now[k] = [mono[0], mono[1]]
        if actin_pool_status[k]==1:
            f=1
            dx = mono[0] - center
            dy = mono[1] - center
            magnitude = np.sqrt(dx ** 2 + dy ** 2)
            if magnitude > particle_radius:
                x_new=mono[0] + f* dx / magnitude
                y_new=mono[1] + f* dy / magnitude
                actin_pool_now[k] = [x_new,y_new]




       # if actin_pool_status[k]==2:

           # mono[0] += V+np.random.normal(-D, D)  # Bias towards the right
            #mono[1] += V+np.random.normal(-D, D)
            #actin_pool_now[k] = [mono[0], mono[1]]
    boundary_x_time=[]
    boundary_y_time =[]
    for i, branch in enumerate(actin_store):
        boundary_x_time.append(branch[-1][0])
        boundary_y_time.append(branch[-1][1])
        polymerise_prob = 0.9
        depolymeris_prob = 0.1

        depolymerise = random.uniform(0, 1)
        polymerise = random.uniform(0, 1)

        branch_polymer_prob.append(polymerise_prob)
        branch_polymer.append(
            polymerise)  ### store to later check if the branch undergoes poymerisation at this time point or not
        if len(branch) > 0:
            ## impose a limit to growth after which polymerisation stops
            actin_last_time.append(branch[0])
            actin_first_time.append(branch[-1])
            branch_radius = np.max(
                np.sqrt((np.array(branch)[:, 0] - center) ** 2 + (np.array(branch)[:, 1] - center) ** 2))

            #if branch_radius >= max_polymer_radius:
            if len(branch) <= 6:
                available_monomer = []
                ind_available_monomer = []
                # print(actin_pool_now)
                for k, mono in enumerate(actin_pool_now):
                    # print(np.sqrt((mono[0]-branch[-1][0])**2+(mono[1]-branch[-1][1])))

                    if len(branch) > 0 and np.sqrt(
                            (mono[0] - branch[-1][0]) ** 2 + (mono[1] - branch[-1][1]) ** 2) <= polymerise_region:
                        # print(len(branch) > 0 and np.sqrt((mono[0] - branch[-1][0]) ** 2 + (mono[1] - branch[-1][1])))
                        available_monomer.append(mono)
                        ind_available_monomer.append(k)
                        break

                # print(available_monomer)
                # print(len(available_monomer))
                if polymerise < polymerise_prob and len(available_monomer) > 0 and len(actin_pool_now) > 0:
                    first_point = branch[0]
                    x, y = first_point
                    for k in range(1, polymer_size):

                        dx = k * (x - center) // initial_r
                        dy = k * (y - center) // initial_r
                        new_x = x - dx
                        new_y = y - dy
                        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                            # grid[new_x, new_y] = 1

                            branch.insert(0, [new_x, new_y])
                            tot_actin -= 1
                            actin_store[i] = branch
                """""
                if depolymerise < depolymeris_prob:
                    depolymerized_monomer = actin_store[i][:polymer_size]
                    actin_store[i] = actin_store[i][polymer_size:]
                    tot_actin += 1

                    # Reset depolymerized monomer to initial position
                    first_point = F_initial_store[i][-1]
                    x, y = first_point
                    depolymerized_monomer[0] = [x, y]
                    actin_pool_now.append([x, y])
                    # Polymerize from the initial position
                    for k in range(1, 2):
                        dx = k * (x - center) // initial_r
                        dy = k * (y - center) // initial_r
                        new_x = x + dx
                        new_y = y + dy
                        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                            depolymerized_monomer.append([new_x, new_y])

                    # Add depolymerized monomer back to actin store
                    actin_store[i] = depolymerized_monomer + actin_store[i]
                """""
                continue  # Skip de polymerization for this branch
            if branch_radius >= max_polymer_radius and branch_type[i]==0:
                if depolymerise < depolymeris_prob:
                    depolymerized_monomer = actin_store[i][:polymer_size]
                    actin_store[i] = actin_store[i][polymer_size:]
                    tot_actin += 1

                    # Reset depolymerized monomer to initial position
                    first_point = F_initial_store[i][-1]
                    x, y = first_point
                    depolymerized_monomer[0] = [x, y]

                    # Polymerize from the initial position
                    for k in range(1, 2):
                        dx = k * (x - center) // initial_r
                        dy = k * (y - center) // initial_r
                        new_x = x + dx
                        new_y = y + dy
                        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                            depolymerized_monomer.append([new_x, new_y])

                    # Add depolymerized monomer back to actin store
                    actin_store[i] = depolymerized_monomer + actin_store[i]

                continue  # Skip polymerization for this branch
        neighbour_lengths = []

        ## implement curvature force. if a bracnh is longer than n of its surroundings the polymerisation propability stops
        for j in range(-no_neighbour, no_neighbour + 1):
            if t % 9 == 0:
                if j != 0:
                    if i + j >= 0 and i + j < len(actin_store):
                        neighbour_lengths.append(len(actin_store[i + j]))

                current_length = len(branch)
                avg_neighbour_length = np.mean(neighbour_lengths) if neighbour_lengths else current_length
                length_difference = current_length - avg_neighbour_length

                # Calculate the curvature based on the length difference
                if current_length != 0:
                    curvature = length_difference / current_length

                    # Apply the restoring force if the curvature is positive (current polymer is longer)
                    if curvature > 0:
                        restoring_force = 0.1 * curvature
                        polymerise_prob -= restoring_force

        ## polymerisation implementation if polymerisation event occours then
        # print(polymerise_prob)
        available_monomer = []
        ind_available_monomer = []
        # print(actin_pool_now)
        for k, mono in enumerate(actin_pool_now):
            # print(np.sqrt((mono[0]-branch[-1][0])**2+(mono[1]-branch[-1][1])))

            if len(branch) > 0 and np.sqrt(
                    (mono[0] - branch[-1][0]) ** 2 + (mono[1] - branch[-1][1]) ** 2) <= polymerise_region:
                # print(len(branch) > 0 and np.sqrt((mono[0] - branch[-1][0]) ** 2 + (mono[1] - branch[-1][1])))
                available_monomer.append(mono)
                ind_available_monomer.append(k)
                break

        # print(available_monomer)
        # print(len(available_monomer))
        if polymerise < polymerise_prob and len(available_monomer) > 0 and len(actin_pool_now) > 0:
            actin_pool_status.pop(ind_available_monomer[0])
            actin_pool_now.pop(ind_available_monomer[0])
            # if len(actin_pool_positions) > 0:
            #   random_index = random.randint(0, len(actin_pool_positions) - 1)
            #   del actin_pool_positions[random_index]
            if len(branch) != 0:
                last_point = branch[-1]
                x, y = last_point
                # grow radially from the last point
                for k in range(1, polymer_size):

                    dx = k * (x - center) // initial_r
                    dy = k * (y - center) // initial_r
                    new_x = x + dx
                    new_y = y + dy
                    if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                        # grid[new_x, new_y] = 1
                        branch.append([new_x, new_y])
                        #   print('polymerise')

                        actin_store[i] = branch

        if depolymerise < depolymeris_prob:
            if len(actin_store[i]) > 0:
                depolymerized_monomer = actin_store[i][:polymer_size]
                actin_store[i] = actin_store[i][polymer_size:]
                #print(depolymerized_monomer[-1])
                ##Reset depolymerized monomer to initial position
                first_point = F_initial_store[i][0]
                x, y = depolymerized_monomer[-1]
               # depolymerized_monomer[0] = [x, y]

                """""
                available_monomer = []
                ind_available_monomer = []
                for k, mono in enumerate(actin_pool_now):
                    # print(np.sqrt((mono[0]-branch[-1][0])**2+(mono[1]-branch[-1][1])))

                    if len(branch) > 0 and np.sqrt(
                            (x - branch[0][0]) ** 2 + (y - branch[0][1])) <= polymerise_region:
                        available_monomer.append(mono)
                        ind_available_monomer.append(k)
                        break

                # Polymerize from the initial position
                for k in range(1, 2):
                    dx = k * (x - center) // initial_r
                    dy = k * (y - center) // initial_r
                    new_x = x + dx
                    new_y = y + dy
                    if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                        depolymerized_monomer.append([new_x, new_y])

                # Add depolymerized monomer back to actin store
                actin_store[i] = depolymerized_monomer + actin_store[i]
                 """""
                fraction_moving=0.1
                actin_pool_now.append([x, y])
                type_segregation=random.uniform(0,1)
                if type_segregation<0.5:
                    actin_pool_status.append(1)
                else:
                    actin_pool_status.append(0)



        ### if not plymerising can experience contraction
        ### check if any free monomers nearby
        available_monomer = []
        ind_available_monomer = []
        # print(actin_pool_now)
        for k, mono in enumerate(actin_pool_now):
            # print(np.sqrt((mono[0]-branch[-1][0])**2+(mono[1]-branch[-1][1])))

            if len(branch) > 0 and np.sqrt(
                    (mono[0] - branch[-1][0]) ** 2 + (mono[1] - branch[-1][1]) ** 2) <= polymerise_region:
                # print(len(branch) > 0 and np.sqrt((mono[0] - branch[-1][0]) ** 2 + (mono[1] - branch[-1][1])))
                available_monomer.append(mono)
                ind_available_monomer.append(k)
                break
        ## if no free monomers nearby implement contraction (we just borrow the random number generated form depolymerise check)
        if depolymerise < shortening and len(available_monomer) == 0:
            if len(actin_store[i]) > polymer_size:
                actin_store[i] = actin_store[i][:-polymer_size]
                tot_actin += 1

                first_point = actin_store[i][0]
                x, y = first_point

                actin_pool_now.append([x, y])
                if y>center:
                    actin_pool_status.append(1)
                if y<=center:
                    actin_pool_status.append(1)

    print(boundary_x_time)
    angle=[]
    for ib in range(len(boundary_x_time)):
       # print(boundary_x_time[ib])
        if boundary_x_time[ib]!=center:
            theta=abs(np.arctan((boundary_y_time[ib]-center)/(boundary_x_time[ib]-center)))**57.2958
        else:
            theta=90
        dx = (boundary_x_time[ib] - center)
        dy = (boundary_y_time[ib]-center)

        if dx > 0 and dy > 0:
            new_angle = theta
            angle.append(new_angle)
        if dx < 0 and dy > 0:
            new_angle = 180 - theta
            angle.append(new_angle)
        if dx < 0 and dy < 0:
            new_angle = 180 + theta
            angle.append(new_angle)
        if dx > 0 and dy < 0:
            new_angle = 360 - theta
            angle.append(new_angle)

    points = zip(boundary_x_time, boundary_y_time, angle)
    sorted_points = sorted(points, key=lambda x: x[2])
    sorted_boundary_x_time, sorted_boundary_y_time, sorted_angle = zip(*sorted_points)

    x_bound, y_bound = interpolate_boundary( sorted_boundary_x_time,  sorted_boundary_y_time, 3)
    interpolated_cords_time = []
    for xb, yb in zip(x_bound, y_bound):
        interpolated_cords_time.append([xb, yb])
    boundary_store.append(interpolated_cords_time)




    bound_monomer_count = sum(int(len(sublist) / polymer_size) for sublist in actin_store)
    free_monomer = len(actin_pool_now)

    print(bound_monomer_count + free_monomer)
   # print(len(actin_pool_now),len(actin_pool_status))
    ## store everything in for this time
    actin_pool_positions = actin_pool_now
    actin_pool_time.append(actin_pool_positions)
    actin_last_store.append(actin_last_time)
    actin_first_store.append(actin_first_time)

    for n in range(len(actin_store)):
        for y in actin_store[n]:
            grid_time[y[0], y[1]] = 1
    grid_store.append(grid_time)
    # actin_store = new_actin_store
    #### simulate TCR movements  ###########

    particle_movement_factor = 1  # Controls the movement of the particles
    particle_in_time = [n for n in particle_store]

    for j, particle in enumerate(particle_store):
        x, y = particle
        if particle_bound_status[j] == 0:
            particle[0] += np.random.normal(0, 0.1)
            particle[1] += np.random.normal(0, 0.1)
            particle_in_time[j] = [particle[0], particle[1]]

        for k, branch in enumerate(actin_store):
            for m in range(len(branch) - 1):
                monomer_x, monomer_y = branch[m]

                if np.sqrt((x - monomer_x) ** 2 + (y - monomer_y) ** 2) <= particle_radius and particle_bound_status[
                    j] == 0:
                    particle[0] = monomer_x
                    particle[1] = monomer_y
                    particle_in_time[j] = [monomer_x, monomer_y]
                    particle_bound_status[j] = 1
                    branch_bound[j] = k

        # if j == 6 and branch_bound[j] != 0:
        # print(polymerise_prob, particle_in_time[j], actin_store[branch_bound[j]], branch_bound[j])
        # print(particle_in_time[j],actin_store[branch_bound[j]],branch_bound[j])
        if (particle_in_time[j]) in actin_store[branch_bound[j]] and particle_bound_status[j] == 1 and branch_bound[
            j] != 0 and branch_polymer[j] < polymerise_prob and branch_type[branch_bound[j]]==1:
            index = actin_store[branch_bound[j]].index(particle_in_time[j])
            branch_ind = branch_bound[j]
            # print(index,len((actin_store[branch_bound[j]])))
            if index < len(actin_store[branch_bound[j]]) - polymer_size and len(actin_store[branch_bound[j]]) > 0:
                # print('move')

                # print(len(actin_store),branch_bound[j],index+1,len(actin_store[branch_bound[j]]))
                # print(index+1,len(actin_store[branch_bound[j]])+polymer_size+1)

                x_move, y_move = actin_store[branch_ind][index + polymer_size - 1]
                particle[0] = x_move
                particle[1] = y_move
                particle_in_time[j] = [x_move, y_move]

        if (particle_in_time[j]) not in actin_store[branch_bound[j]] and particle_bound_status[j] == 1 and branch_bound[
            j] != 0:
            # print('diss')
            # print([monomer_x, monomer_y],branch)
            particle_bound_status[j] = 2   ### change to 2 to couple
        # if (particle_in_time[j])  in actin_store[branch_bound[j]] and particle_bound_status[j] == 1:
        #   print('bound')

        if particle_bound_status[j] == 2:### change to 2 to couple
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


# plt.imshow(grid)
# plt.show()


def update_frame(frame):
    plt.cla()  # Clear the current plot
    plt.imshow(grid_store[frame], cmap='binary')  # Plot the grid at the current frame
    # plt.imshow(particle_time[frame])  # Plot the grid at the current frame

    bound = boundary_store[frame]
    x = [p[1] for p in bound]
    y = [p[0] for p in bound]
    #plt.scatter(x, y, color='blue',s=1)


    particles = particle_time[frame]
    x = [p[1] for p in particles]
    y = [p[0] for p in particles]
    plt.scatter(x, y, color='red', s=1)

    free_monomer = actin_pool_time[frame]
    x = [p[1] for p in free_monomer]
    y = [p[0] for p in free_monomer]
    plt.scatter(x, y, color='green', s=0.2)

    act_pool_last = actin_last_store[frame]

    xl = [p[1] for p in act_pool_last]
    yl = [p[0] for p in act_pool_last]
    plt.scatter(xl, yl, color='blue', s=0.3)

    act_pool_first = actin_first_store[frame]

    xi = [p[1] for p in act_pool_first]
    yi = [p[0] for p in act_pool_first]
   # plt.scatter(xi, yi, color='blue', s=0.3)

    # plt.title('Frame {}'.format(frame))
    # plt.axis('off')  # Turn off the axis
    # plt.title('Frame {}'.format(frame))
    # plt.axis('off')  # Turn off the axis

    # Plot the boundary
    if frame < len(boundary_points):
        boundary = boundary_points[frame]
        boundary_x = [p[1] for p in boundary]
        boundary_y = [p[0] for p in boundary]
        plt.plot(boundary_x, boundary_y, color='blue')


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
