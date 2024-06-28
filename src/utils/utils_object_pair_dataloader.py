
from scipy.spatial.distance import euclidean


from src.utils.polar_coordinates import cart2pol

def add_polar_coords(obj_camera, obj_lidar, obj_ego, obj_gt, center_for_each_step):

    if obj_gt != []:
        objects = [obj_camera, obj_lidar, obj_gt]
    else:
        objects = [obj_camera, obj_lidar]

    if center_for_each_step:
        # For each step: subtract the ego-pos from obj_camera, obj_lidar and obj_gt
        for time_idx in obj_ego['time_idx_in_scenario_frame']:
            for obj in objects:
                
                if time_idx in obj['time_idx_in_scenario_frame']:
                    obj_time_idx = list(obj['time_idx_in_scenario_frame']).index(time_idx)
                    obj['x'][obj_time_idx] = obj['x'][obj_time_idx] - obj_ego['x'][int(time_idx)]
                    obj['y'][obj_time_idx] = obj['y'][obj_time_idx] - obj_ego['y'][int(time_idx)]
    else:
        # Do nothing
        pass

    # Convert Cartesian coordinates to Polar coordinates and add them to the object dict
    for obj in objects:

        obj_cart = [[x,y] for x, y in zip(obj['x'], obj['y'])]
        obj_pol = cart2pol(obj_cart)

        if center_for_each_step:
            obj['radius_c']  = [pos[0] for pos in obj_pol]
            obj['azimuth_c'] = [pos[1] for pos in obj_pol]
        else:
            obj['radius']  = [pos[0] for pos in obj_pol]
            obj['azimuth'] = [pos[1] for pos in obj_pol]



    return obj_camera, obj_lidar, obj_ego, obj_gt



def add_dist_ego_obj(obj_camera, obj_lidar, obj_ego, obj_gt, metric='euclidean'):
    if metric == 'euclidean':
        distance_function = euclidean
    else:
        assert False, "TBD"

    obj_ego['dist_ego_obj'] =  [-1 for x in range(len(obj_ego['x']))]

    if obj_gt != []:
        objects = [obj_camera, obj_lidar, obj_gt]
    else:
        objects = [obj_camera, obj_lidar]

    for obj_cur in objects:

        # assert obj_1.shape == obj_2.shape, "the Dimensions should be the same"
        intersection_idx = sorted(set(obj_cur['time_idx_in_scenario_frame']).intersection(obj_ego['time_idx_in_scenario_frame']))

        distance_list = []
        for idx in intersection_idx:
            idx_cur = obj_cur['time_idx_in_scenario_frame'].tolist().index(idx)
            idx_ego = obj_ego['time_idx_in_scenario_frame'].tolist().index(idx)
            dist    = distance_function([obj_cur['x'][idx_cur], obj_cur['y'][idx_cur]], 
                                        [obj_ego['x'][idx_ego], obj_ego['y'][idx_ego]])
            distance_list.append(dist)

        assert len(distance_list) == len(obj_cur['x']), "length must match"
        obj_cur['dist_ego_obj'] = distance_list



    return obj_camera, obj_lidar, obj_ego, obj_gt
