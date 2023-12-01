import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

# Load field data from the CSV file
def load_field_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        field_data = [list(map(float, row)) for row in csvreader]
    return field_data

# Define parameters
t_file_path = 'T_field.csv'
rho_file_path = 'rho_field.csv'

# Load data from the CSV files
t_field_data = load_field_from_csv(t_file_path)
rho_field_data = load_field_from_csv(rho_file_path)

# Find the maximum length of rows for both fields
max_t_row_length = max(len(row) for row in t_field_data)
max_rho_row_length = max(len(row) for row in rho_field_data)

# Fill missing values with NaN and convert to NumPy arrays
t_field_array = np.array([row + [np.nan] * (max_t_row_length - len(row)) for row in t_field_data])
rho_field_array = np.array([row + [np.nan] * (max_rho_row_length - len(row)) for row in rho_field_data])

# Calculate the number of frames
num_frames = len(t_field_array) // 128  # Assuming 128 rows per frame

# Create a figure and axis for T field
fig_t, ax_t = plt.subplots()
image_t = ax_t.imshow(t_field_array[:128],
                      cmap='Reds', extent=[0, 128, 0, 128],
                      origin='lower', aspect='auto')
ax_t.set_title('Scalar T Field (Time Step 0)')
plt.colorbar(image_t, label='T')

# Create a figure and axis for rho field
fig_rho, ax_rho = plt.subplots()
image_rho = ax_rho.imshow(rho_field_array[:128],
                          cmap='Blues', extent=[0, 128, 0, 128],
                          origin='lower', aspect='auto')
ax_rho.set_title('Scalar rho Field (Time Step 0)')
plt.colorbar(image_rho, label='rho')

# Update functions for the animations
def update_t(frame):
    start_row = frame * 128
    end_row = (frame + 1) * 128
    frame_data = t_field_array[start_row:end_row]
    image_t.set_array(frame_data)
    ax_t.set_title(f'Scalar T Field (Time Step {frame})')

def update_rho(frame):
    start_row = frame * 128
    end_row = (frame + 1) * 128
    frame_data = rho_field_array[start_row:end_row]
    image_rho.set_array(frame_data)
    ax_rho.set_title(f'Scalar rho Field (Time Step {frame*10000})')

# Create the animations
animation_t = FuncAnimation(fig_t, update_t, frames=num_frames, interval=200, repeat=False)
animation_rho = FuncAnimation(fig_rho, update_rho, frames=num_frames, interval=200, repeat=False)

# Save the animations as video files
animation_t.save('T_field_evolution.mp4', writer='ffmpeg', fps=5)
animation_rho.save('rho_field_evolution.mp4', writer='ffmpeg', fps=5)

