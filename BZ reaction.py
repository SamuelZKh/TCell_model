import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib import animation

# Width, height of the image.
nx, ny = 512, 512

alpha = 1
beta = 1
gamma = 1
r = 200  # Initial radius of the circular region

def update(p, arr):
    global r  # Declare r as a global variable to modify it within the function
    q = (p + 1) % 2
    s = np.zeros((3, ny, nx))
    m = np.ones((3, 3)) / 9
    for k in range(3):
        s[k] = convolve2d(arr[p, k], m, mode='same', boundary='wrap')

    arr[q, 0] = s[0] + s[0] * (alpha * s[1] - gamma * s[2])
    arr[q, 1] = s[1] + s[1] * (beta * s[2] - alpha * s[0])
    arr[q, 2] = s[2] + s[2] * (gamma * s[0] - beta * s[1])

    # Dynamically adjust circular boundary based on the concentration of species 0
    concentration_species_0 = arr[q, 0]
    mean_concentration = np.mean(concentration_species_0)
    threshold = 0.6  # You can adjust the threshold to suit your needs
    if mean_concentration > threshold and r < 300:  # Define an upper limit for the radius
        r += 1  # Increase the radius
    elif mean_concentration <= threshold and r > 100:  # Define a lower limit for the radius
        r -= 1  # Decrease the radius

    y, x = np.ogrid[:ny, :nx]
    mask = (x - nx // 2) ** 2 + (y - ny // 2) ** 2 > r ** 2
    for i in range(3):
        arr[q, i][mask] = 0

    np.clip(arr[q], 0, 1, arr[q])
    return arr

# Initialize the array with random values
arr = np.random.uniform(0, 1, size=(3, 3, ny, nx))

fig, ax = plt.subplots()
im = ax.imshow(arr[0, 1], cmap=plt.cm.jet)
ax.axis('off')

def animate(i, arr):
    arr = update(i % 2, arr)
    im.set_array(arr[i % 2, 0])
    return [im]

anim = animation.FuncAnimation(fig, animate, frames=100, interval=100, blit=False, fargs=(arr,))

plt.show()
