import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib import animation
repulsion_distance = 20
# Width, height of the image.
nx, ny = 512, 512

alpha = 0.5
beta = 0.5
gamma = 0.5
r = 150  # Radius of the circular region
# Function to generate coordinates in a circle
def generate_circle_coordinates(radius, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(angles) + nx // 2
    y = radius * np.sin(angles) + ny // 2
    return x, y

# Generate coordinates in a circle
num_points = 100  # Number of points in the circle
circle_x, circle_y = generate_circle_coordinates(r, num_points)

def update(p, arr,circle_x,circle_y):
    q = (p + 1) % 2  #    It alternates between 0 and 1 in the update() function: q = (p + 1) % 2.It's used as an index to update the arrays representing the concentrations at different time steps.
    s = np.zeros((3, ny, nx)) ## store the different conc of the chemical fields each s[k] store convoluted array for each species
    m = np.ones((3, 3)) / 9 ## averaging for concolutional pourposes
    for k in range(3):
        s[k] = convolve2d(arr[p, k], m, mode='same', boundary='wrap')

    arr[q, 0] = s[0] + s[0] * (alpha * s[1] - gamma * s[2])
    arr[q, 1] = s[1] + s[1] * (beta * s[2] - alpha * s[0])
    arr[q, 2] = s[2] + s[2] * (gamma * s[0] - beta * s[1])
    #print(arr[q,0])
    # Set values outside the circular region to 0
    for grid_x in range(nx):
        for grid_y in range(ny):
            if arr[q, 0][grid_x, grid_y] > 0.95:
                # Calculate distances between grid points and circle points
                distances = np.sqrt((circle_x - grid_x) ** 2 + (circle_y - grid_y) ** 2)

                # Repel circle points within the specified distance
                circle_x = np.where(distances < repulsion_distance, circle_x + (circle_x - grid_x) * 0.1, circle_x)
                circle_y = np.where(distances < repulsion_distance, circle_y + (circle_y - grid_y) * 0.1, circle_y)

    y, x = np.ogrid[:ny, :nx]
    mask = (x - nx // 2) ** 2 + (y - ny // 2) ** 2 > r ** 2
    # print(np.max(arr[q, 0]))
    for i in range(3):
        arr[q, i][mask] = 0

    np.clip(arr[q], 0, 1, arr[q])  ### limits the array to a value between 0 and 1 of >1=1 if <0=0
    return arr,circle_x,circle_y

# Initialize the array with random values
arr = np.random.uniform(0, 1, size=(3, 3, ny, nx))

fig, ax = plt.subplots()
im = ax.imshow(arr[0, 1], cmap=plt.cm.jet)
ax.axis('off')

# Plot the initial circle coordinates
scatter = ax.scatter(circle_x, circle_y, color='red', s=5)


def animate(i, arr, circle_x, circle_y):
    arr, new_circle_x, new_circle_y = update(i % 2, arr, circle_x, circle_y)
    im.set_array(arr[i % 2, 0])

    # Update scatter plot of evolved circle coordinates at each time point
    scatter.set_offsets(np.vstack((new_circle_x, new_circle_y)).T)

    return [im, scatter]

anim = animation.FuncAnimation(fig, animate, frames=100, interval=100, blit=False, fargs=(arr, circle_x, circle_y))

plt.show()

# To save the animation as an MP4 movie, uncomment this line
# anim.save(filename='bz.mp4', fps=30)