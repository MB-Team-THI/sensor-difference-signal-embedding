
import cv2
import math
import torch
import numpy as np
from PIL import Image  

import numpy as np
from matplotlib.pyplot import imshow
from matplotlib.image import imsave

from src.utils.image_creation import SignalToImageConverter
# from src.utils.image_creation import plot_signals
from src.train.loss.loss_152 import calculate_error_signal_full_with_padding

max_vals = {'diff_x':                   -1,
            'diff_y':                   -1,
            'diff_heading':             -1,
            'diff_v':                   -1, 
            'mean_eucl_dist_obj_ego':   -1, 
            'diff_radius':              -1, 
            'diff_azimuth':             -1,
            'diff_radius_c':            -1,
            'diff_azimuth_c':           -1, 
            } 
            

def trajectory_to_image(self, obj_camera=None, obj_lidar=None):
    # Settings
    # RGB color scheme:
    color_channel_camera = 0    # = red
    color_channel_lidar = 1     # = green --> 2 = blue
    line_thickness = 1

    image = np.zeros((3, self.bbox_pixel[1], self.bbox_pixel[0]), dtype=int)
    resolution_x = image.shape[2] / self.bbox_meter[1]
    resolution_y = image.shape[1] / self.bbox_meter[0]
    

    # The mix of the two colors is yellow = overlaying trajectory of the two objects
    background_color = (255, 255, 255)

    if obj_camera is not None:
        obj_camera_pixel = np.empty((2, len(obj_camera[1, :])), dtype=int)    
        obj_camera_pixel[0, :] = ((obj_camera[0,:] - self.min_x_meter) * resolution_x).astype(int)
        obj_camera_pixel[1, :] = (image.shape[1] - 1) - ((obj_camera[1,:]- self.min_y_meter) * resolution_y).astype(int)
        # camera is the red channel
        image[color_channel_camera, :, :] = cv2.polylines(image[0, :, :], [obj_camera_pixel.T], False, background_color, line_thickness)
    
    if obj_lidar is not None:
        obj_lidar_pixel = np.empty((2, len(obj_lidar[1, :])), dtype=int)
        obj_lidar_pixel[0, :] = ((obj_lidar[0, :] - self.min_x_meter) * resolution_x).astype(int)
        obj_lidar_pixel[1, :] = (image.shape[1] - 1) - ((obj_lidar[1, :] - self.min_y_meter) * resolution_y).astype(int)        
        # lidar is the green channel
        image[color_channel_lidar, :, :]  = cv2.polylines(image[1, :, :], [obj_lidar_pixel.T],  False, background_color, line_thickness)

    image = np.moveaxis(image, 0, 2)

    return image


def error_signal_to_image(calc_error_signal, norm_max_values = [], obj_camera=[], obj_lidar=[], obj_ego=None, pre_calc_error_signal=[], included_features_types = ['diff_x'], data_order_dict=[]):

    # TODO adapt error in the visualization of the error signal plot 
    if calc_error_signal:
        # Calculate the error signal based on the object pair

        ### Obtain error signal
        if False:
            # For "eval_152.py"
            batch_camera = [torch.moveaxis(torch.from_numpy(obj_camera).to(torch.float).t(), 0, 1)]
            batch_lidar  = [torch.moveaxis(torch.from_numpy(obj_lidar).to(torch.float).t(), 0, 1)]
            batch_ego    = [torch.moveaxis(torch.from_numpy(obj_ego).to(torch.float).t(), 0, 1)]
        else:
            # To save images from the dataloader
            batch_camera = [torch.from_numpy(obj_camera).to(torch.float).t()]
            batch_lidar  = [torch.from_numpy(obj_lidar).to(torch.float).t()]
            batch_ego    = [torch.from_numpy(obj_ego).to(torch.float).t()]

        error_signal_batch_padded, x_lengths, error_sgn_idx = calculate_error_signal_full_with_padding(batch_camera             = batch_camera,
                                                                                                       batch_lidar              = batch_lidar,
                                                                                                       batch_ego                = batch_ego,
                                                                                                       included_features_types  = included_features_types,
                                                                                                       enable_padding           = False,
                                                                                                       padding_value            = 10,
                                                                                                       data_order_dict          = data_order_dict)
    else:
        # Use the precalculated error signal
        error_signal_batch_padded = pre_calc_error_signal

    # TODO make this work for the image_creation as well again
    signal      = [error_signal_batch_padded[0][:, i].cpu().detach().numpy() for i in range(len(included_features_types))]
    signal_dict = {key: signal[idx] for idx, key in enumerate(included_features_types)}

    # Determine the settings for the plot of each sample
    # max_values_y_axis determine the resolution on the y-axis
    for key in included_features_types:
        if max(abs(signal_dict[key])) > max_vals[key]:
            max_vals[key] = max(abs(signal_dict[key]))
    
    if norm_max_values == []:
        # Val values for val and train set
        norm_max_values = {'diff_x':                   32.3455,
                           'diff_y':                   32.3133,
                           'diff_heading':              3.1416,
                           'diff_v':                   13.7660,
                           'diff_eucl_dist_obj_ego':   31.6163, 
                           'mean_eucl_dist_obj_ego':   49.9590,
                           'diff_radius':              25.5196,
                           'diff_azimuth':              3.1185,
                           'diff_radius_c':            32.6163,
                           'diff_azimuth_c':            2.6685, 
                           }      

    exp_functions_params = {'diff_x':                 {'a': 8.7, 'b':1.3, 'd': 0.5, 'shift_y': 62.48, 'shift_x': 15},
                            'diff_y':                 {'a': 8.7, 'b':1.3, 'd': 0.5, 'shift_y': 62.48, 'shift_x': 15},
                            'diff_heading':           {'a': 8.7, 'b':1.3, 'd': 3.6, 'shift_y': 64,    'shift_x': 2.1},
                            'diff_v':                 {'a': 8.7, 'b':1.3, 'd': 0.9, 'shift_y': 64.7,  'shift_x': 8.5},
                            'mean_eucl_dist_obj_ego': {'a': 8.7, 'b':1.3, 'd': 0.5, 'shift_y': 62.3,  'shift_x': 15},
                            'diff_radius':            {'a': 8.7, 'b':1.3, 'd': 0.5, 'shift_y': 62.48, 'shift_x': 15},
                            'diff_azimuth':           {'a': 8.7, 'b':1.3, 'd': 3.6, 'shift_y': 64,    'shift_x': 2.1},
                            'diff_radius_c':          {'a': 8.7, 'b':1.3, 'd': 0.5, 'shift_y': 62.48, 'shift_x': 15},
                            'diff_azimuth_c':         {'a': 8.7, 'b':1.3, 'd': 3.6, 'shift_y': 64,    'shift_x': 2.1},
                            }   

    max_values  = {'x': 40,   'y': norm_max_values}
    image_size  = {'x': 256,  'y': 256}

    color_codes = {'diff_x':                  {'r': 255, 'g': 0,   'b': 0},        # red            = diff_x
                   'diff_y':                  {'r': 0,   'g': 254, 'b': 0},        # green          = diff_y
                   'diff_heading':            {'r': 237, 'g': 229, 'b': 184},      # dutch white    = diff_heading
                   'diff_v':                  {'r': 252, 'g': 200, 'b': 78},       # Yellow         = diff_v
                   'mean_eucl_dist_obj_ego':  {'r': 219, 'g': 93,  'b': 15},       # Flame / orange = mean_eucl_dist_obj_ego    
                   'diff_radius':             {'r': 153, 'g': 204, 'b': 255},      # Light blue
                   'diff_azimuth':            {'r': 255, 'g': 153, 'b': 255},      # Light Purple 
                   'diff_radius_c':           {'r': 0,   'g': 255, 'b': 255},      # Cyan
                   'diff_azimuth_c':          {'r': 244, 'g': 8,   'b': 252},}     # Dark Purple  

    
    plot_modes  = ['bresenham_plot', 'pyplotlib']
    converter   = SignalToImageConverter(signal_dict, 
                                         plot_mode              = plot_modes[1],
                                         image_size             = image_size,
                                         max_vals               = max_values, 
                                         color_codes            = color_codes,
                                         exp_functions_params   = exp_functions_params)
    paired_data = converter.signal_to_pair()
    if converter.plot_mode == "bresenham_plot":
        image       = converter.bresenham_pair_to_image(paired_data)

    elif converter.plot_mode == "pyplotlib":
        image = converter.plot_signals(signal1 = paired_data['diff_x'], 
                                       signal2 = paired_data['diff_y'])
    return image

