
import torch
import numpy as np

from scipy.spatial.distance import euclidean


def get_distance_between_objects(obj_1, obj_2, data_order, metric='euclidean'):
    # Return the distance between obj_1 and obj_2 for every common timestep

    if metric == 'euclidean':
        dist_function = euclidean
    else:
        assert False, 'TBD'

    if torch.is_tensor(obj_1):
        obj_1 = obj_1.cpu().detach().numpy()
    if torch.is_tensor(obj_2):
        obj_2 = obj_2.cpu().detach().numpy()

    # assert obj_1.shape == obj_2.shape, "the Dimensions should be the same"
    intersection_idx = sorted(set(obj_1[:, 0]).intersection(obj_2[:, 0]))

    idx_x = data_order['data_order_camera'].index('x')
    idx_y = data_order['data_order_camera'].index('y')
    distance_list = []

    for idx in intersection_idx:
        idx_1 = obj_1[:, 0].tolist().index(idx)
        idx_2 = obj_2[:, 0].tolist().index(idx)
        dist = dist_function([obj_1[idx_1][idx_x], obj_1[idx_1][idx_y]], 
                             [obj_2[idx_2][idx_x], obj_2[idx_2][idx_y]])
        distance_list.append(dist)

    return distance_list


def get_distance_to_ego(obj_ego, obj_camera, obj_lidar, data_order, metric='euclidean', calc='mean'):
    # Return mean distance of lidar and camera to ego

    distance_list_ego_obj_camera = get_distance_between_objects(obj_1=obj_ego, obj_2=obj_camera, data_order=data_order, metric=metric)
    distance_list_ego_obj_lidar  = get_distance_between_objects(obj_1=obj_ego, obj_2=obj_lidar,  data_order=data_order, metric=metric)

    if calc == 'mean':
        distance_ego_obj_camera = np.mean(distance_list_ego_obj_camera)
        distance_ego_obj_lidar  = np.mean(distance_list_ego_obj_lidar)
    else:
        assert False, 'TBD'

    return np.mean([distance_ego_obj_camera, distance_ego_obj_lidar])

