import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib import animation

# Width, height of the image.
nx, ny = 512, 512

alpha = 0.5
beta = 0.5
gamma = 1
r = 200  # Radius of the circular region

def update(p, arr):
    q = (p + 1) % 2  #    It alternates between 0 and 1 in the update() function: q = (p + 1) % 2.It's used as an index to update the arrays representing the concentrations at different time steps.
    s = np.zeros((3, ny, nx)) ## store the different conc of the chemical fields each s[k] store convoluted array for each species
    m = np.ones((3, 3)) / 9 ## averaging for concolutional pourposes
    for k in range(3):
        s[k] = convolve2d(arr[p, k], m, mode='same', boundary='wrap')

    arr[q, 0] = s[0] + s[0] * (alpha * s[1] - gamma * s[2])
    arr[q, 1] = s[1] + s[1] * (beta * s[2] - alpha * s[0])
    arr[q, 2] = s[2] + s[2] * (gamma * s[0] - beta * s[1])

    # Set values outside the circular region to 0
    y, x = np.ogrid[:ny, :nx]
    mask = (x - nx // 2) ** 2 + (y - ny // 2) ** 2 > r ** 2
    for i in range(3):
        arr[q, i][mask] = 0

    np.clip(arr[q], 0, 1, arr[q])  ### limits the array to a value between 0 and 1 of >1=1 if <0=0
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

# To save the animation as an MP4 movie, uncomment this line
# anim.save(filename='bz.mp4', fps=30)
