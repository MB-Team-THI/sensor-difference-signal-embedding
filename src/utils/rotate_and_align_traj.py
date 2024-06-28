from src.utils.rot_points import rot_points
import numpy as np
import matplotlib.pyplot as plt



def transform_objects(obj_camera, obj_lidar, obj_ego, obj_gt, rotation_type='ego'):
    # Decide based on what the object the rotation and centering should be done
    if rotation_type == 'ego':
        # Starting point is the position and heading of the 'ego'-vehicle (at the time the first object is seen)
        min_idx = int(min(obj_lidar['time_idx_in_scenario_frame'][0], obj_camera['time_idx_in_scenario_frame'][0]))
        starting_point = {'x':       obj_ego['x'][min_idx],
                          'y':       obj_ego['y'][min_idx],
                          'heading': obj_ego['heading'][min_idx]}
    

    elif rotation_type == 'first_obj':
        # Starting point is the position and heading of the first detected obj (camera or lidar)
        if obj_lidar['time_idx_in_scenario_frame'][0] < obj_camera['time_idx_in_scenario_frame'][0]:
            starting_point = {'x':       obj_lidar['x'][0],
                              'y':       obj_lidar['y'][0],
                              'heading': obj_lidar['heading'][0]}
        else:
            starting_point = {'x':       obj_camera['x'][0],
                              'y':       obj_camera['y'][0],
                              'heading': obj_camera['heading'][0]}

    else:
        raise ValueError('rotation_type not supported')
    
    # Center and rotate the trajectories objects of camera, lidar, ego and gt based on the 'starting_point'
    # TODO bbox dimension and lateral and longitudinal velocity do not match anymore
    obj_camera = center_and_rotate_trajectory(starting_point, obj=obj_camera)    
    obj_lidar  = center_and_rotate_trajectory(starting_point, obj=obj_lidar)
    obj_ego    = center_and_rotate_trajectory(starting_point, obj=obj_ego)
    if obj_gt != []:
        obj_gt     = center_and_rotate_trajectory(starting_point, obj=obj_gt)

    return obj_camera, obj_lidar, obj_ego, obj_gt


def center_and_rotate_trajectory(starting_point, obj):
    # Idx=0: x, idx=1: y, idx=2: timestamp, idx=3: heading
    # Set the center and rotate the coordinate system based on the starting points
    # Center
    obj['x'] = obj['x'] - starting_point['x']
    obj['y'] = obj['y'] - starting_point['y']  

    # Rotation
    len_obj   = len(obj['y'])
    new_pos_y = np.zeros(len_obj)
    new_pos_x = np.zeros(len_obj)
    new_heading = np.zeros(len_obj)
    # Note: the incoming angle must be negative
    angle = starting_point['heading']
    for idx in range(len_obj):
        new_pos_x[idx], new_pos_y[idx] = rot_points([obj['x'][idx], obj['y'][idx]], angle)
        new_heading[idx] = - obj['heading'][idx] + angle
        # TODO rotate v_lat, v_lon
        

    obj['x']       = new_pos_x
    obj['y']       = new_pos_y
    obj['heading'] = new_heading

    if False:        
        plt.scatter(obj['x'] , obj['y'] )
        r = 5 
        for idx in range(len(obj['x'])):
            plt.arrow(obj['x'][idx], 
                      obj['y'][idx],
                      r*np.cos(new_pos_heading[idx]), 
                      r*np.sin(new_pos_heading[idx]))

    return obj

def center_single_instance(obj_ego, obj_c, obj_l, idx_t=0, idx_x=1, idx_y=2): 
    # Center the camera and lidar objects for each instance of the ego obj
    # Just return the overlapping camera and lidar objects 

    intersections = sorted(set(obj_c[idx_t]).intersection(obj_l[idx_t]))
    intersection_idx = [int(x) for x in intersections]
    obj_c_new = []
    obj_l_new = []
    for idx in intersection_idx:
        idx_l = list(obj_l[idx_t]).index(idx)
        x = obj_l[idx_x][idx_l] - obj_ego[idx_x][idx] 
        y = obj_l[idx_y][idx_l] - obj_ego[idx_y][idx] 
        obj_l_new.append([x,y])
        
        idx_c = list(obj_c[idx_t]).index(idx)
        x = obj_c[idx_x][idx_c] - obj_ego[idx_x][idx] 
        y = obj_c[idx_y][idx_c] - obj_ego[idx_y][idx] 
        obj_c_new.append([x,y])

    return obj_c_new, obj_l_new, intersection_idx
    
    
