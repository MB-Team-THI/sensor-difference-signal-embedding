
def both_traj_available(obj_camera, obj_lidar):
    # Only return the time steps where both trajectories are available.
    intersecting_time_idx = sorted(set(obj_camera['time_idx_in_scenario_frame']).intersection(obj_lidar['time_idx_in_scenario_frame']))
    
    assert len(intersecting_time_idx) != 0, "Error, no overlaying frames found!"

    overlay_idx_camera = [i in intersecting_time_idx for i in obj_camera['time_idx_in_scenario_frame']]
    overlay_idx_lidar  = [i in intersecting_time_idx for i in obj_lidar['time_idx_in_scenario_frame']]
    assert sum(overlay_idx_camera) == sum(overlay_idx_lidar),  "Idx-list is of different length"

    for key in obj_camera:
        obj_camera[key] = obj_camera[key][overlay_idx_camera]
        obj_lidar[key]  = obj_lidar[key][overlay_idx_lidar]

    assert obj_camera != [] or obj_lidar != [],         "obj_camera or obj_lidar are empty, after only_overlaying."
    assert len(obj_camera['x']) == len(obj_lidar['x']), "After overlaying, the object have somehow different length."

    return obj_camera, obj_lidar

