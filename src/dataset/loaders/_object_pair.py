
import os
import copy
import numpy as np

from scipy.io import loadmat
import matplotlib.pyplot as plt
from src.utils.rotate_and_align_traj import transform_objects
from src.utils.filter_both_traj_available import both_traj_available
from src.utils.create_subdivision import create_subdivision_distance

from src.utils.utils_object_pair_dataloader import add_polar_coords
from src.utils.utils_object_pair_dataloader import add_dist_ego_obj


def _visualize_trajectories(old, new, obj_pair_name):
    output_dir = 'output_files\\verify_trajectory_rotation\\'

    fig = plt.figure()
    fig.set_size_inches(15, 10.5)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Left subplot
    obj_camera  = old['obj_camera']
    obj_lidar   = old['obj_lidar']
    obj_ego     = old['obj_ego']
    obj_gt      = old['obj_gt']
    ax1.plot(obj_camera['x'], obj_camera['y'], 'r', label='Camera Object')
    ax1.plot(obj_camera['x'][0], obj_camera['y'][0], 'p', color='r')
    ax1.plot(obj_lidar['x'], obj_lidar['y'], 'g', label='Lidar Object')
    ax1.plot(obj_lidar['x'][0], obj_lidar['y'][0], 'p', color='g')
    ax1.plot(obj_ego['x'], obj_ego['y'], 'pink', label='EGO')
    ax1.plot(obj_ego['x'][0], obj_ego['y'][0], 'p', color='pink')
    ego_seeing_object_pair = obj_ego['time_idx_in_scenario_frame'].tolist().index(obj_camera['time_idx_in_scenario_frame'][0])
    ax1.plot(obj_ego['x'][ego_seeing_object_pair], obj_ego['y'][ego_seeing_object_pair], 'x', color='black')
    if obj_gt != []:
        ax1.plot(obj_gt['x'], obj_gt['y'], 'b', label='GT Object', alpha=0.5)
        ax1.plot(obj_gt['x'][0], obj_gt['y'][0], 'p', color='b', alpha=0.5)
    ax1.axis('equal')
    ax1.legend()
    ax1.grid()
    ax1.set_title('original object pair')

    # Right subplot
    obj_camera  = new['obj_camera']
    obj_lidar   = new['obj_lidar']
    obj_ego     = new['obj_ego']
    obj_gt      = new['obj_gt']
    ax2.plot(obj_camera['x'], obj_camera['y'], 'r', label='Camera Object')
    ax2.plot(obj_camera['x'][0], obj_camera['y'][0], 'p', color='r')
    ax2.plot(obj_lidar['x'], obj_lidar['y'], 'g', label='Lidar Object')
    ax2.plot(obj_lidar['x'][0], obj_lidar['y'][0], 'p', color='g')
    ax2.plot(obj_ego['x'], obj_ego['y'], 'pink', label='EGO')
    ax2.plot(obj_ego['x'][0], obj_ego['y'][0], 'p', color='pink')
    ego_seeing_object_pair = obj_ego['time_idx_in_scenario_frame'].tolist().index(obj_camera['time_idx_in_scenario_frame'][0])
    ax2.plot(obj_ego['x'][ego_seeing_object_pair], obj_ego['y'][ego_seeing_object_pair], 'x', color='black')
    if obj_gt != []:
        ax2.plot(obj_gt['x'], obj_gt['y'], 'b', label='GT Object', alpha=0.5)
        ax2.plot(obj_gt['x'][0], obj_gt['y'][0], 'p', color='b', alpha=0.5)
    ax2.axis('equal')
    ax2.legend()
    ax2.grid()
    ax2.set_title('rotated and shifted object pair')
    fig.show()

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = obj_pair_name + '.png'
    plt.savefig(output_dir + filename)

    plt.close()




def _dict_to_ndarray(obj_dict, features_to_load):
    dict_keys = list(obj_dict.keys())

    obj_array = np.ndarray(shape=(len(features_to_load), len(obj_dict[dict_keys[0]])))

    idx = 0
    for key in features_to_load:
        if key in dict_keys:
            obj_array[idx, :] = obj_dict[key]
            idx += 1

    return obj_array


def _load_sample_object_pair(self, filename_sample, features_to_load=['x', 'y', 'time_idx_in_scenario_frame'], 
                             only_overlaying=False, create_subdivisions=False, subdivisions_params=None, rotation_type='ego'):
    
    # Load Object Pair ==================================================================================
    mat_temp = loadmat(filename_sample,
                       variable_names=["general_info", "scene_info", "obj_ego", "obj_lidar",
                                       "obj_camera", "obj_gt", "obj_pair_global_idx", "obj_pair_name"],
                       verify_compressed_data_integrity=False)
        
    # META===============================================================================================
    # INFO-----------------------------------------------------------------------------------------------
    obj_pair_name       = mat_temp['obj_pair_name'][0]
    obj_pair_global_idx = mat_temp['obj_pair_global_idx'][0][0]

    scene_info_keys     = mat_temp['scene_info']['scene'][0][0][0].__dir__.__self__.dtype.names
    scene_info_vals     = [mat_temp['scene_info']['scene'][0][0][0][0][i][0][:] for i in range(len(scene_info_keys))]
    scene_info_vals[2]  = scene_info_vals[2][0]
    scene_info          = {k: v for (k, v) in zip(scene_info_keys, scene_info_vals)}

    map_info_keys       = mat_temp['scene_info']['map'][0][0][0].__dir__.__self__.dtype.names
    map_info_vals       = [mat_temp['scene_info']['map'][0][0][0][0][i][0][:] for i in range(len(map_info_keys))]
    map_info            = {k: v for (k, v ) in zip(map_info_keys, map_info_vals)}

    # DYNAMICS============================================================================================
    # Object EGO -----------------------------------------------------------------------------------------
    temp_ego        = mat_temp['obj_ego']
    data_order_ego  = list(temp_ego[0][0].__dir__.__self__.dtype.names)
    obj_ego         = {data_order_ego[idx]: temp_ego[0][0][idx][0] for idx in range(len(data_order_ego))}
    obj_ego_org     = copy.deepcopy(obj_ego)

    # Object Ground Truth --------------------------------------------------------------------------------
    temp_gt = mat_temp['obj_gt']
    if len(temp_gt) > 0:
        # TODO check for test-set if this applies like this
        data_order_gt   = list(temp_gt[0][0].__dir__.__self__.dtype.names)
        obj_gt          = {data_order_gt[idx]: temp_gt[0][0][idx][0] for idx in range(len(data_order_gt))}
    else:
        data_order_gt   = []
        obj_gt          = []
    obj_gt_org = copy.deepcopy(obj_gt)

    # Object Camera -------------------------------------------------------------------------------------
    temp_camera         = mat_temp['obj_camera']
    data_order_camera   = list(temp_camera[0][0].__dir__.__self__.dtype.names)
    obj_camera          = {data_order_camera[idx]: temp_camera[0][0][idx][0] for idx in range(len(data_order_camera))}

    # Object Lidar -------------------------------------------------------------------------------------
    temp_lidar          = mat_temp['obj_lidar']
    data_order_lidar    = list(temp_camera[0][0].__dir__.__self__.dtype.names)
    obj_lidar           = {data_order_lidar[idx]: temp_lidar[0][0][idx][0] for idx in range(len(data_order_lidar))}

    # General Info -------------------------------------------------------------------------------------
    general_info        = {'data_order_camera':     features_to_load,
                           'data_order_lidar':      features_to_load,
                           'data_order_ego':        features_to_load,
                           'data_order_gt':         features_to_load,
                           'data_order_ego_org':    data_order_ego,
                           'data_order_gt_org':     data_order_gt}


    # ALTER DATA =========================================================================================
    # get rid of 'pred_class'
    obj_camera.pop('pred_class', None)
    obj_lidar.pop('pred_class', None)
    if obj_gt != []:
        obj_gt.pop('pred_class', None)
        obj_gt.pop('gt_attribute', None)
        obj_gt.pop('gt_category', None)


    # Only use the frames where both objects are visible =================================================
    if only_overlaying:
        obj_camera, obj_lidar = both_traj_available(obj_camera, obj_lidar)

    # Rotate and align trajectory ========================================================================
    old_objects = {'obj_camera': copy.deepcopy(obj_camera), 
                   'obj_lidar':  copy.deepcopy(obj_lidar), 
                   'obj_ego':    copy.deepcopy(obj_ego), 
                   'obj_gt':     copy.deepcopy(obj_gt)}
    obj_camera, obj_lidar, obj_ego, obj_gt = transform_objects(obj_camera    = obj_camera,
                                                               obj_lidar     = obj_lidar,
                                                               obj_ego       = obj_ego, 
                                                               obj_gt        = obj_gt,
                                                               rotation_type = rotation_type)

    # Add some additional to the gt information
    if obj_gt != []:
        obj_gt_org['tracking_score_camera'] = obj_camera['tracking_score']
        obj_gt_org['tracking_score_lidar']  = obj_lidar['tracking_score']

    # Confirm proper object processing and transformation by visualization
    if False:
        new_objects = {'obj_camera': obj_camera, 
                       'obj_lidar':  obj_lidar, 
                       'obj_ego':    obj_ego, 
                       'obj_gt':     obj_gt}

        _visualize_trajectories(old=old_objects, new=new_objects, obj_pair_name=obj_pair_name)

    if 'dist_ego_obj' in features_to_load:
        obj_camera, obj_lidar, obj_ego, obj_gt = add_dist_ego_obj(obj_camera    = obj_camera,
                                                                  obj_lidar     = obj_lidar,
                                                                  obj_ego       = obj_ego, 
                                                                  obj_gt        = obj_gt)
        
    if ('azimuth' in features_to_load) or ('radius' in features_to_load):
        obj_camera, obj_lidar, obj_ego, obj_gt = add_polar_coords(obj_camera           = obj_camera,
                                                                  obj_lidar            = obj_lidar,
                                                                  obj_ego              = obj_ego, 
                                                                  obj_gt               = obj_gt,
                                                                  center_for_each_step = False)
        
    if ('azimuth_c' in features_to_load) or ('radius_c' in features_to_load):
        obj_camera, obj_lidar, obj_ego, obj_gt = add_polar_coords(obj_camera           = obj_camera,
                                                                  obj_lidar            = obj_lidar,
                                                                  obj_ego              = obj_ego, 
                                                                  obj_gt               = obj_gt,
                                                                  center_for_each_step = True)
        

    # Create subdivision =================================================================================
    # TODO should I treat subdivisions as ODDs ?
    if create_subdivisions:
        subdivision = create_subdivision_distance(subdivisions_params, 
                                                  obj_ego    = obj_ego, 
                                                  obj_lidar  = obj_lidar, 
                                                  obj_camera = obj_camera)
    else:
        subdivision = None
    
    # Sanity check
    for obj in (obj_ego, obj_lidar, obj_camera, obj_gt):
        for key in obj:
            assert len(obj[list(obj.keys())[0]]) == len(obj[key]), "all features within an object should have same length"
        

    # Filter for the 'features_to_load' and convert the dict to an array =================================
    obj_array_camera  = _dict_to_ndarray(obj_camera, features_to_load)
    obj_array_lidar   = _dict_to_ndarray(obj_lidar, features_to_load)
    obj_array_ego     = _dict_to_ndarray(obj_ego, features_to_load)
    if obj_gt != []:
        obj_array_gt = _dict_to_ndarray(obj_gt, features_to_load)
    else:
        obj_array_gt = []

    # Output =============================================================================================
    # Output dimensions for obj_camera, obj_lidar and obj_ego
    # Dimensions: N_features x N_timestamps
    out_dict = {'obj_pair_global_idx':  obj_pair_global_idx,
                'obj_pair_name':        obj_pair_name,
                'general_info':         general_info,
                'scene_info':           scene_info,
                'map_info':             map_info,
                'obj_camera':           obj_array_camera,
                'obj_lidar':            obj_array_lidar,
                'obj_ego':              obj_array_ego,
                'obj_gt':               obj_array_gt,
                'obj_ego_org':          obj_ego_org,
                'obj_gt_org':           obj_gt_org,
                'subdivision':          subdivision}
    
    return out_dict
