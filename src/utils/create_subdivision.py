import json
from scipy.spatial.distance import euclidean

# TODO thinks about subdivision that include also more domain-knowledge (driving left, right or straight ( object being, left, right, in front, behind)

def create_subdivision_distance(subdivisions_params, obj_ego, obj_lidar, obj_camera):
    # Divide the object pairs in subdivisions based on the distance between the objects and the ego.
    threshold_distance = subdivisions_params['distance_threshold']

    # Get the distance in the middle of the detection time
    camera_middle_time_idx = int(obj_camera.shape[1]/2)
    lidar_middle_time_idx = int(obj_lidar.shape[1]/2)
    middle_idx = int((camera_middle_time_idx + lidar_middle_time_idx) / 2)
    # Calculate the distance
    # distance_camera_obj = euclidean([obj_ego[0][camera_middle_time_idx], obj_ego[1][camera_middle_time_idx]],
    #                                 [obj_camera[0][camera_middle_time_idx], obj_camera[1][camera_middle_time_idx]])
    # distance_lidar_obj = euclidean([obj_ego[0][lidar_middle_time_idx], obj_ego[1][lidar_middle_time_idx]],
    #                                 [obj_lidar[0][lidar_middle_time_idx], obj_lidar[1][lidar_middle_time_idx]])    
    distance_camera_obj = euclidean([obj_ego[0][middle_idx], obj_ego[1][middle_idx]],
                                    [obj_camera[0][middle_idx], obj_camera[1][middle_idx]])
    distance_lidar_obj = euclidean([obj_ego[0][middle_idx], obj_ego[1][middle_idx]],
                                   [obj_lidar[0][middle_idx], obj_lidar[1][middle_idx]])
    mean_distance_obj_ego = (distance_camera_obj + distance_lidar_obj) / 2.0

    if mean_distance_obj_ego <= threshold_distance:
        subdivision = "near-field, " + str(threshold_distance)+"m"
    else:
        subdivision = "far-field, " + str(threshold_distance)+"m"

    # TODO throws error when used within the SCENARIO-NET training setup

    return subdivision


def create_subdivision_day_night(scene_info):
    base_dir = 'output_files\\'
    filename = 'scene_trainval_test.json'
    f = open(base_dir + filename)
    data = json.load(f)
    f.close()

    res = None
    for entry in data:
        if entry['token'] == scene_info['token']:
            res = entry['day_night']
            break

    return res


def create_subdivision_rain_no_rain(scene_info):
    base_dir = 'output_files\\'
    filename = 'scene_trainval_test.json'
    f = open(base_dir + filename)
    data = json.load(f)
    f.close()

    res = None
    for entry in data:
        if entry['token'] == scene_info['token']:
            res = entry['rain_no_rain']
            break

    return res
