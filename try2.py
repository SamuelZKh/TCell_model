from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

from collections import defaultdict
import PIL
from sklearn.cluster import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson
from itertools import chain
from scipy.special import i0




from tqdm import tqdm
import scipy.cluster.hierarchy as hcluster
from scipy import ndimage
# change the following to %matplotlib notebook for interactive plotting
%matplotlib notebook
from scipy.signal import find_peaks
# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import cv2
import pims
import trackpy as tp

## import images

def filter_elements(A, B, threshold):
    filtered_list = []
    for element_A in A:
        for element_B in B:
            if abs(element_A - element_B) <= threshold:
                filtered_list.append(element_A)
                break  # Break out of the inner loop once a match is found
    return filtered_list
def count_ones(matrix):
    count = 0
    tot=0
    for row in matrix:
        for element in row:
            tot+=1
            if element == 1:
                count += 1
    return (count/tot)*100



def subtract_with_previous(lst):
    result = []
    for i in range(1, len(lst)):
        diff = np.abs(lst[i] - lst[i - 1])
        result.append(diff)
    return result

def find_deviation(file):
    @pims.pipeline
    def gray(image):
        return image[:, :, 1]  # Take just the green channel

    frames = pims.open(file)
    first_frame = frames[1]
    blurred_frame = cv2.GaussianBlur(first_frame, (5, 5),100)  # Applying Gaussian blur with kernel size (5, 5)
    threshold = 20  # Set desired threshold value

    binary_img = np.where(blurred_frame > threshold, 1,0)
    
  
    Area_initial = count_ones(binary_img)
    center_of_mass_initial = ndimage.measurements.center_of_mass(binary_img)
    Area_store=[]
    av_vec_store=[]
    r_cm_store=[]
    var_vector_store=[]
    cm_coordinates_x=[]
    cm_coordinates_y=[]
    for n in range(0,100):
        first_frame = frames[n]
        blurred_frame = cv2.GaussianBlur(first_frame, (5, 5),100)  # Applying Gaussian blur with kernel size (5, 5)
        threshold = 20  # Set desired threshold value

        binary_img = np.where(blurred_frame > threshold, 1,0)
        Area = count_ones(binary_img)
        Area_store.append(Area)
        
        %matplotlib notebook
        # Load the binary image
        image = binary_img
        image = cv2.convertScaleAbs(image)
        center_of_mass = ndimage.measurements.center_of_mass(binary_img)
        
        cm_coordinates_x.append(center_of_mass[0])
        cm_coordinates_y.append(center_of_mass[1])
        r_cm_store.append(np.sqrt(center_of_mass[0]**2+center_of_mass[1]**2))
        
        # Apply Canny edge detection
        edges = cv2.Canny(image, 0, 1)  # Adjust the thresholds as needed
       
        # Convert edge image to binary format
        edges = edges.astype(np.uint8)

        # Find contours of the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Combine all edge coordinates into a single list
        edge_coordinates = []
        for contour in contours:
            contour_coordinates = []
            for point in contour:
                x, y = point[0]
                contour_coordinates.append((x, y))
            edge_coordinates.extend(contour_coordinates)

        # Convert edge_coordinates to a NumPy array
        edge_coordinates = np.array(edge_coordinates)
        coordinates = edge_coordinates
        coordinates = np.array(coordinates)

        # Calculate radial distance from the origin
        distances=[]
        for pos in edge_coordinates:
            r=np.sqrt((pos[0]-center_of_mass_initial[0])**2+(pos[1]-center_of_mass_initial[1])**2)
            distances.append(r)
        av_dist_time=np.mean(distances)
        
        # Print the radial distances
        #print(np.mean(distances))
        av_vec_store.append(np.mean(distances))
        var_time=[]
        for r in distances:
            var_time.append(np.abs(r-av_dist_time))
        
        var_vector_store.append(np.mean(var_time))
    
    difference = subtract_with_previous(av_vec_store)
    area_change= subtract_with_previous(Area_store)
    
    variance_change=subtract_with_previous(var_vector_store)
    cm_coord_change_x=subtract_with_previous(cm_coordinates_x)
    cm_coord_change_y=subtract_with_previous(cm_coordinates_y)
    ##r_cm_change= subtract_with_previous(r_cm_store)
    r_cm_change= [i**2+j**2 for i,j in zip(cm_coord_change_x,cm_coord_change_x)]
    Area_initial=1
    normalised_difference=[i/Area_initial for i in difference ]
    normalised_area_change=[i/Area_initial for i in  area_change]
    normalised_r_cm_change=[i/Area_initial for i in  r_cm_change]
    normalised_variance_change=[i/Area_initial for i in  variance_change]
    
    return normalised_difference,normalised_area_change,normalised_r_cm_change,normalised_variance_change

def boundary_tracking(file):
    m_p=1
    Fps=1
    @pims.pipeline
    def gray(image):
        return image[:, :, 1]  # Take just the green channel

    frames = pims.open(file)
    first_frame = frames[0]

    blurred_frame = cv2.GaussianBlur(first_frame, (5, 5),100)  # Applying Gaussian blur with kernel size (5, 5)
    threshold = 20  # Set desired threshold value

    binary_img = np.where(blurred_frame > threshold, 1,0)

    center_of_mass = ndimage.measurements.center_of_mass(binary_img)
    surface_store=[]
    rmin_frame=[]
    for n in range(0,100):
        first_frame = frames[n]
        blurred_frame = cv2.GaussianBlur(first_frame, (5, 5),100)  # Applying Gaussian blur with kernel size (5, 5)
        threshold = 20  # Set desired threshold value

        binary_img = np.where(blurred_frame > threshold, 1,0)
        
        
        %matplotlib notebook
        # Load the binary image
        image = binary_img
        image = cv2.convertScaleAbs(image)
        
        
        
        # Apply Canny edge detection
        edges = cv2.Canny(image, 0, 1)  # Adjust the thresholds as needed
       
        # Convert edge image to binary format
        edges = edges.astype(np.uint8)
    
        # Find contours of the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Combine all edge coordinates into a single list
        edge_coordinates = []
        for contour in contours:
            contour_coordinates = []
            for point in contour:
                x, y = point[0]
                contour_coordinates.append((x, y))
            edge_coordinates.extend(contour_coordinates)

        # Convert edge_coordinates to a NumPy array
        edge_coordinates = np.array(edge_coordinates)
        angle_to_fix=np.linspace(0,360,25)
        edge_coordinates_fixed=[]
        angle=[]
        
        
        for pos in edge_coordinates:
            x, y = pos
            rmin_frame.append(np.sqrt((x-center_of_mass[1])**2+(y-center_of_mass[0])**2))
            
            theta=abs(np.arctan((y-center_of_mass[0])/(x-center_of_mass[1]))*57.2958)
            dx=(x-center_of_mass[1])
            dy=(y-center_of_mass[0])
        
            if dx>0 and dy>0:
                new_angle=theta
                angle.append(new_angle)
            if dx<0 and dy>0:
                new_angle=180-theta
                angle.append(new_angle)
            if dx<0 and dy<0:
                new_angle=180+theta
                angle.append(new_angle)
            if dx>0 and dy<0:
                new_angle=360-theta
                angle.append(new_angle)
        
            for ang in angle_to_fix:
                if abs(new_angle-ang)<2:
                    edge_coordinates_fixed.append([x,y])

            #####################################################################
       
        
        for pos in edge_coordinates_fixed:
            x, y = pos
            first_frame[y, x] =  first_frame[y, x] = 255  # Increase brightness by 50, capped at 255
        surface_store.append( first_frame)
    
  
    
    f = tp.batch(surface_store,11,minmass=255);
    t = tp.link(f,5, memory=8)

    t1 = tp.filter_stubs(t,30)
    ## subtract any drift in image(look at image from the center of mass)
    d = tp.compute_drift(t1)
    tm = tp.subtract_drift(t1.copy(), d) 
    tm.rename(columns = {'size':'area'}, inplace = True)
    data = pd.DataFrame()
    
    for item in set(tm.particle):
    
        sub = tm[tm.particle==item]
    
        dx = m_p*np.diff(sub.x)/1. # differnece between to frame for X postion
        dy = m_p*np.diff(sub.y)/1. #wdiffernece between to frame for Y postion
        dt = np.diff(sub.frame)/Fps #to calcul the time
        dr=((np.sqrt(dy**2 + dx**2)))
        tan = (dy/dx) # to calcul the slope
        ang_d=abs((np.arctan(tan)*57.2958))
        #print(np.diff(sub.frame))
        v = (((np.sqrt(dy**2 + dx**2)))/dt) # to calcul the velocity  12 microns per pixel
        
        for x, y, dx, dy,disp, v,ang_disp,area, dt, frame in zip(sub.x[:-1], sub.y[:-1], abs(dx), abs(dy),dr ,v,ang_d,sub.area[:-1] ,dt,sub.frame[:-1],):
            data = data.append([{'dx': dx,
            'dy': dy,
             'x': x,
             'y': y,
            'frame': frame,
             'particle': item,
              'disp':disp,                  
             'dt' : dt,
             'area':area,
              'ang_disp':ang_disp,                   
             'v' : v}])
       
    
    
    utraj = np.unique(tm.particle)
    num_traj = utraj.size
    cdict = {}
    return tm,data,center_of_mass[1],center_of_mass[0],np.mean(rmin_frame)
def find_ee_distance(data):  ## takes in dataframe and a max lag time
    disp_part_i=[]
    xi=[]
    yi=[]
    part_id=[]
    ## see all particle numbers that have trajectory 
    part_temp=list(data['particle'])
    part = list(set(part_temp))
    for i in part: ## loop over all trajectories
        del_t=[]
        msd_part_i=[]
        df=data.loc[data['particle'] ==i]  ## dataframe of all rows with particle =i
        x_list=list(df['x'])## x positions of particle i
        y_list=list(df['y'])## y positions of particle i
        dx=(x_list[0]-x_list[-1])
        dy=(y_list[0]-y_list[-1])
        disp=np.sqrt(dx**2+dy**2) ## find displacement
        disp_part_i.append(disp)
        xi.append(x_list[-1])
        yi.append(y_list[-1])
        part_id.append(i)
    return xi,yi,part_id    
def find_rad_distance(data,xc,yc):  ## takes in dataframe and a max lag time
    disp_part_i=[]
    ri_list=[]
    rf_list=[]
    part_id=[]
    ## see all particle numbers that have trajectory 
    part_temp=list(data['particle'])
    part = list(set(part_temp))
    for i in part: ## loop over all trajectories
        del_t=[]
        msd_part_i=[]
        df=data.loc[data['particle'] ==i]  ## dataframe of all rows with particle =i
        x_list=list(df['x'])## x positions of particle i
        y_list=list(df['y'])## y positions of particle i
        dxi=(x_list[0]-xc)
        dyi=(y_list[0]-yc)
        dxf=(x_list[-1]-xc)
        dyf=(y_list[-1]-yc)  
        ri=np.sqrt(dxi**2+dyi**2) ## find displacement
        rf=np.sqrt(dxf**2+dyf**2) ## find displacement
        ri_list.append(ri)
        rf_list.append(rf)
        part_id.append(i)
    return ri_list,rf_list,part_id

def in_out(ri,rf,xe,ye,iidr,data,plot,rmin,file):
    frames = pims.open(file)
    first_frame = frames[0]
    blurred_frame = cv2.GaussianBlur(first_frame, (5, 5),100)  # Applying Gaussian blur with kernel size (5, 5)
    threshold = 20  # Set desired threshold value

    binary_img = np.where(blurred_frame > threshold, 1,0)
    
  
    Area_initial = count_ones(binary_img)
    %matplotlib notebook
    iid_in=[]
    iid_out=[]
    ee_distance=[]
    V_filtered_list=[]
    for i in range(0,len(ri)):
        
        if ri[i]>rf[i] and ri[i]>=rmin :
          
            df=data.loc[data['particle'] ==iidr[i]]
            trajx=df['x'].tolist()
            trajy=df['y'].tolist()
            dx=( trajx[0]- trajx[-1])
            dy=( trajy[0]- trajy[-1])
            disp=np.sqrt(dx**2+dy**2) ## find displacement
            ee_distance.append(disp/ Area_initial )
            
            v_list=df['v'].tolist()
            
            for j in (v_list):
                V_filtered_list.append(j)
            if plot==1:
                plt.scatter(xe[i],ye[i],color='yellow',s=10)
                plt.plot(trajx,trajy,color='brown')
            iid_in.append(i)
          
        if ri[i]<rf[i] and ri[i]>=rmin : 
        
            df=data.loc[data['particle'] ==iidr[i]]
            trajx=df['x'].tolist()
            trajy=df['y'].tolist()
            
            dx=( trajx[0]- trajx[-1])
            dy=( trajy[0]- trajy[-1])
            disp=np.sqrt(dx**2+dy**2) ## find displacement
            ee_distance.append(disp/ Area_initial )
            v_list=df['v'].tolist()
            if plot==1:
                plt.scatter(xe[i],ye[i],color='blue',s=10)
                plt.plot(trajx,trajy,color='brown')
            for j in (v_list):
                V_filtered_list.append(j)
           
            iid_out.append(i)
           
         #plt.ylim(max(ye), min(ye))

    if plot==1:
        plt.imshow(frames[0]);
    plt.show()
    return V_filtered_list,ee_distance
    
def smooth_list(lst, window_size):
    smoothed = []
    for i in range(len(lst)):
        window_start = max(0, i - window_size + 1)
        window_end = i + 1
        window_values = lst[window_start:window_end]
        window_average = sum(window_values) / len(window_values)
        smoothed.append(window_average)
    return smoothed


#def average_lists_by_label(A, B):
#    grouped_lists = defaultdict(list)
#    
#    for sublist, label in zip(A, B):
#        grouped_lists[label].append(sublist)
    
#    averaged_lists = {}
#    for label, sublists in grouped_lists.items():
#        averaged_lists[label] = [sum(elements) / len(elements) for elements in zip(*sublists)]
    
 #   return averaged_lists

def average_lists_by_label(A, B):
    grouped_lists = defaultdict(list)

    for sublist, label in zip(A, B):
        grouped_lists[label].append(sublist)

    averaged_lists = {}
    variances= {}
    for label, sublists in grouped_lists.items():
        averaged_lists[label] = [np.mean(elements) for elements in zip(*sublists)]
        variances[label] = [np.var(elements) for elements in zip(*sublists)]
        

    return  averaged_lists,variances


