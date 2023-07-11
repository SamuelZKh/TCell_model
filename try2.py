import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

import numpy as np
import math
# Specify the file path


def calculate_surface_roughness_mean(x_coordinates, y_coordinates):
    # Convert coordinates to NumPy arrays
    x = np.array(x_coordinates)
    y = np.array(y_coordinates)

    # Calculate the total number of points
    num_points = len(x)

    # Calculate the average distance between consecutive points
    average_distance = np.mean(np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2))

    # Calculate the difference between the actual distance and the average distance for each point
    differences = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2) - average_distance

    # Calculate the root mean square (RMS) of the differences
    surface_roughness = np.sqrt(np.mean(differences**2))

    return surface_roughness

def calculate_surface_roughness_interpolate(x, y, smoothing_factor):
    # Fit a smoothed curve to the boundary
    t = np.arange(len(x))
    spl = UnivariateSpline(t, x, k=smoothing_factor)
    smooth_x = spl(t)
    spl = UnivariateSpline(t, y, k=smoothing_factor)
    smooth_y = spl(t)

    # Calculate the deviation of each point from the smoothed curve
    deviations = np.sqrt((x - smooth_x) ** 2 + (y - smooth_y) ** 2)
      
    # Calculate the average surface roughness
    surface_roughness = np.mean(deviations)
    return surface_roughness, smooth_x, smooth_y

def calculate_surface_roughness_time_avg(x, y, window_size):
    # Calculate the time average of points using a sliding window
    smooth_x = np.convolve(x, np.ones(window_size)/window_size, mode='valid')
    smooth_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')

    # Calculate the deviation of each point from the time-averaged curve
    deviations = np.sqrt((x[window_size-1:] - smooth_x) ** 2 + (y[window_size-1:] - smooth_y) ** 2)

    # Calculate the average surface roughness
    surface_roughness = np.mean(deviations)
    return surface_roughness, smooth_x, smooth_y

def final_function(file_path):
    

    # Read the tab-delimited file into a DataFrame
    data = pd.read_csv(file_path, delimiter='\t')
    data
    num_columns = data.shape[1]
    # Display the DataFrame
    #%matplotlib widget
    x_coords=[]
    y_coords=[]
    for i in range(num_columns):
        if i%2==0 or i==0:
            x = data.iloc[:, i].tolist()
        
            x_coords.append(x)
        else:
            y=data.iloc[:, i].tolist()
            y_coords.append(y)
    #for i in range(len(x_coords)):
        #plt.scatter(x_coords[i],y_coords[i],s=0.01)
    #plt.show()
    
    x_store = [[value for value in sublist if not math.isnan(value)] for sublist in x_coords]
    y_store = [[value for value in sublist if not math.isnan(value)] for sublist in y_coords]


    roughness_mean_store=[]
    roughness_interpolate_store=[]
    roughness_time_avg_store=[]
    for i in range(len(x_store)):
        x_list =x_store[i]
        y_list =y_store[i]
    
        smoothing = 5  # Smoothing factor, adjust as needed
        window_size = 20
    
        roughness_mean = calculate_surface_roughness_mean(x_list, y_list)
        roughness_interpolate, smooth_x, smooth_y = calculate_surface_roughness_interpolate(x_list, y_list, smoothing)
        roughness_time_avg, smooth_x, smooth_y = calculate_surface_roughness_time_avg(x_store[i], y_store[i], window_size)
    
        roughness_mean_store.append(roughness_mean)
        roughness_interpolate_store.append(roughness_interpolate)
        roughness_time_avg_store.append(roughness_time_avg)
    
    #%matplotlib widget
    fig, axs = plt.subplots(2, 1, figsize=(3, 4))

    # Plot roughness_mean_store
    axs[0].plot(roughness_mean_store)
    axs[0].set_title('Roughness Mean Store')

    # Plot roughness_interpolate_store
    #axs[1].plot(roughness_interpolate_store)
    #axs[1].set_title('Roughness Interpolate Store')

    # Plot roughness_time_avg_store
    axs[1].plot(roughness_time_avg_store)
    axs[1].set_title('Roughness Time Avg Store')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()
    
    return   roughness_mean_store

