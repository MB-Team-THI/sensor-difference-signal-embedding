import math
import torch
from scipy.spatial.distance import euclidean


architecture_types          = ["Encoder", "AE", "AE_advanced_phase1", "AE_advanced_phase2", "AE_error_signal"]
decoderT_types              = ['LSTM-Decoder', 'CNN-Decoder', 'MLP']
loss_types                  = ['reconstruction_loss', 'VICReg', 'SiamSim']
processed_objects_types     = ['obj_pair', 'obj_lidar', 'obj_camera']
loss_target_recon_types     = ['obj_pair', 'obj_lidar', 'obj_camera', 'calc_error_signal', 'recon_error_signal']
error_signal_types          = ['euclidean', 'difference_for_each_dim']
error_signal_feature_types = ['diff_x', 'diff_y', 'diff_heading', 'diff_v', 'directed_distance', 'euclidean_distance', 
                              'mean_distance_to_ego_x', 'mean_distance_to_ego_y', 'diff_eucl_dist_obj_ego', 'mean_eucl_dist_obj_ego', 
                              'diff_radius', 'diff_azimuth', 'diff_radius_c', 'diff_azimuth_c']


def get_directed_distance_from_vectors(vec_1, vec_2):
    vec_directed = torch.sub(vec_2, vec_1)

    sign = torch.sign(torch.sum(vec_directed))
    distance_abs = torch.sqrt(torch.sum(torch.square(vec_directed)))

    return torch.multiply(distance_abs, sign)


def calculate_error_signal_idx(obj_camera, obj_lidar, idx_c, idx_l, data_order_dict, included_features_types, obj_ego=None, idx_e=None):
    error_signal_out, error_signal_out_legend = [], []
    idx_pos_x         = data_order_dict['data_order_camera'].index('x')
    idx_pos_y         = data_order_dict['data_order_camera'].index('y')
    idx_pos_heading   = data_order_dict['data_order_camera'].index('heading')
    idx_pos_v         = data_order_dict['data_order_camera'].index('v')
    idx_pos_radius    = data_order_dict['data_order_camera'].index('radius')
    idx_pos_azimuth   = data_order_dict['data_order_camera'].index('azimuth')
    idx_pos_radius_c  = data_order_dict['data_order_camera'].index('radius_c')
    idx_pos_azimuth_c = data_order_dict['data_order_camera'].index('azimuth_c')
    
    pos_camera = torch.tensor((obj_camera[idx_c][idx_pos_x], obj_camera[idx_c][idx_pos_y])) #.to(device)
    pos_lidar  = torch.tensor((obj_lidar[idx_l][idx_pos_x],  obj_lidar[idx_l][idx_pos_y])) #.to(device)
    pos_ego    = torch.tensor((obj_ego[idx_e][idx_pos_x],    obj_ego[idx_e][idx_pos_y])) #.to(device)

    if obj_camera.get_device() != -1:
        device = "cuda:"+str(obj_camera.get_device())
        pos_camera = pos_camera.to(device)
        pos_lidar  = pos_lidar .to(device)
        pos_ego    = pos_ego   .to(device) 
                              
    # feature_types: 'diff_x'
    if error_signal_feature_types[0] in included_features_types:
        error_signal_out.append((obj_lidar[idx_l][idx_pos_x] - obj_camera[idx_c][idx_pos_x]))
        error_signal_out_legend.append(error_signal_feature_types[0])

    # feature_types: 'diff_y'
    if error_signal_feature_types[1] in included_features_types:
        error_signal_out.append((obj_lidar[idx_l][idx_pos_y] - obj_camera[idx_c][idx_pos_y]))
        error_signal_out_legend.append(error_signal_feature_types[1])

    # feature_types: 'diff_heading'
    if error_signal_feature_types[2] in included_features_types:
        diff_heading = (obj_lidar[idx_l][idx_pos_heading] - obj_camera[idx_c][idx_pos_heading])
        if abs(diff_heading) > math.pi:
            diff_heading = (2 * math.pi - abs(diff_heading)) * (-1)
        error_signal_out.append(diff_heading)
        error_signal_out_legend.append(error_signal_feature_types[2])

    # feature_types: 'diff_v'
    if error_signal_feature_types[3] in included_features_types:
        error_signal_out.append((obj_lidar[idx_l][idx_pos_v] - obj_camera[idx_c][idx_pos_v]))
        error_signal_out_legend.append(error_signal_feature_types[3])

    # feature_types: 'directed_distance' - directed_distance (+-) between lidar and camera object
    if error_signal_feature_types[4] in included_features_types:
        vec_camera = torch.stack([obj_camera[idx_c][idx_pos_x], obj_camera[idx_c][idx_pos_y]])
        vec_lidar  = torch.stack([obj_lidar[idx_l][idx_pos_x], obj_lidar[idx_l][idx_pos_y]])
        distance_directed = get_directed_distance_from_vectors(vec_camera, vec_lidar)
        error_signal_out.append(distance_directed)
        error_signal_out_legend.append(error_signal_feature_types[4])

    # feature_types: 'euclidean_distance'
    if error_signal_feature_types[5] in included_features_types:
        eucl_dist = (pos_lidar - pos_camera).pow(2).sum().sqrt()
        error_signal_out.append(eucl_dist)
        error_signal_out_legend.append(error_signal_feature_types[5])

    # feature_types: 'mean_distance_to_ego_x',
    if error_signal_feature_types[6] in included_features_types:
        mean_distance_to_ego_x = obj_ego[idx_e][idx_pos_x] - (obj_camera[idx_c][idx_pos_x] + obj_lidar[idx_l][idx_pos_y]) / 2.0
        error_signal_out.append(mean_distance_to_ego_x)
        error_signal_out_legend.append(error_signal_feature_types[6])

    # feature_types: 'mean_distance_to_ego_y',
    if error_signal_feature_types[7] in included_features_types:
        mean_distance_to_ego_y = obj_ego[idx_e][idx_pos_y] - (obj_camera[idx_c][idx_pos_y] + obj_lidar[idx_l][idx_pos_y]) / 2.0
        error_signal_out.append(mean_distance_to_ego_y)
        error_signal_out_legend.append(error_signal_feature_types[7])

    # feature_type: 'diff_eucl_dist_obj_ego'
    if ((error_signal_feature_types[8] in included_features_types) or 
        (error_signal_feature_types[9] in included_features_types)):
        dist_ego_camera = (pos_ego - pos_camera).pow(2).sum().sqrt()
        dist_ego_lidar  = (pos_ego - pos_lidar) .pow(2).sum().sqrt()


    if error_signal_feature_types[8] in included_features_types:        
        error_signal_out.append(dist_ego_camera - dist_ego_lidar)
        error_signal_out_legend.append(error_signal_feature_types[8])

    # feature_type: 'mean_dist_obj_ego'
    if error_signal_feature_types[9] in included_features_types:
        error_signal_out.append((dist_ego_camera + dist_ego_lidar) / 2.0)
        error_signal_out_legend.append(error_signal_feature_types[9])

    # feature_type: 'diff_radius'
    if error_signal_feature_types[10] in included_features_types:        
        error_signal_out.append((obj_lidar[idx_l][idx_pos_radius] - obj_camera[idx_c][idx_pos_radius]))
        error_signal_out_legend.append(error_signal_feature_types[10])
    
    # feature_type: 'diff_azimuth'
    if error_signal_feature_types[11] in included_features_types:
        diff_azimuth = (obj_lidar[idx_l][idx_pos_azimuth] - obj_camera[idx_c][idx_pos_azimuth])
        if abs(diff_azimuth) > math.pi:
            diff_azimuth = (2 * math.pi - abs(diff_azimuth)) * (-1)
        error_signal_out.append(diff_azimuth)
        error_signal_out_legend.append(error_signal_feature_types[11])

        # feature_type: 'diff_radius_c'
    if error_signal_feature_types[12] in included_features_types:
        error_signal_out.append((obj_lidar[idx_l][idx_pos_radius_c] - obj_camera[idx_c][idx_pos_radius_c]))
        error_signal_out_legend.append(error_signal_feature_types[12])
    
    # feature_type: 'diff_azimuth_c'
    if error_signal_feature_types[13] in included_features_types:
        diff_azimuth_c = (obj_lidar[idx_l][idx_pos_azimuth_c] - obj_camera[idx_c][idx_pos_azimuth_c])
        if abs(diff_azimuth_c) > math.pi:
            diff_azimuth_c = (2 * math.pi - abs(diff_azimuth_c)) * (-1)
        error_signal_out.append(diff_azimuth_c)
        error_signal_out_legend.append(error_signal_feature_types[13])
    
    return error_signal_out, error_signal_out_legend



def calculate_error_signal(error_signal, obj_camera, obj_lidar, idx_c, idx_l, included_features_types):
    # feature_types: 'diff_x'
    # TODO get this information from the 'general_info' - data_order
    idx_pos_x = 1
    if error_signal_feature_types[0] in included_features_types:
        error_signal[error_signal_feature_types[0]].append(
            (obj_lidar[idx_l][idx_pos_x] - obj_camera[idx_c][idx_pos_x]))

    # feature_types: 'diff_y'
    # TODO get this information from the 'general_info' - data_order
    idx_pos_y = 2
    if error_signal_feature_types[1] in included_features_types:
        error_signal[error_signal_feature_types[1]].append(
            (obj_lidar[idx_l][idx_pos_y] - obj_camera[idx_c][idx_pos_y]))

    # feature_types: 'directed_distance' - directed_distance (+-) between lidar and camera object
    if error_signal_feature_types[2] in included_features_types:
        vec_camera = torch.stack(
            [obj_camera[idx_c][idx_pos_x], obj_camera[idx_c][idx_pos_y]])
        vec_lidar = torch.stack(
            [obj_lidar[idx_l][idx_pos_x], obj_lidar[idx_l][idx_pos_y]])
        distance_directed = get_directed_distance_from_vectors(
            vec_camera, vec_lidar)
        error_signal[error_signal_feature_types[2]].append(distance_directed)

    # feature_types: 'euclidean_distance'
    if error_signal_feature_types[3] in included_features_types:
        # TODO adapt for tensors
        eucl_dist = euclidean([obj_lidar[idx_l][idx_pos_x],  obj_lidar[idx_l][idx_pos_y]],
                              [obj_camera[idx_c][idx_pos_x], obj_camera[idx_c][idx_pos_y]])
        error_signal[error_signal_feature_types[6]].append(eucl_dist)

    return error_signal


def calculate_error_signal_full_with_padding(batch_lidar, batch_camera, batch_ego, data_order_dict, included_features_types=['diff_x', 'diff_y'],
                                             enable_padding=False, padding_value=15):
    error_signal_batch = []
    overlap_idx_batch = []
    idx_pos_time = data_order_dict['data_order_camera'].index('time_idx_in_scenario_frame')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for idx in range(len(batch_camera)):
        obj_camera = batch_camera[idx]
        obj_lidar = batch_lidar[idx]
        obj_ego = batch_ego[idx]

        error_signal = []
        overlap_idx_list = []
        # Basic error signal and case: obj_camera > obj_lidar
        for idx_c in range(obj_camera.shape[0]):
            error_signal_idx = None
            for idx_l in range(obj_lidar.shape[0]):
                if obj_camera[idx_c][idx_pos_time] == obj_lidar[idx_l][idx_pos_time]:

                    if idx_c != 0 and obj_lidar[idx_l][idx_pos_time] == 0 and obj_camera[idx_c][idx_pos_time] == 0:
                        break
                    idx_e = int(obj_lidar[idx_l][idx_pos_time].item())
                    error_signal_idx, legend = calculate_error_signal_idx(obj_camera                = obj_camera,
                                                                          obj_lidar                 = obj_lidar,
                                                                          obj_ego                   = obj_ego,
                                                                          idx_c                     = idx_c,
                                                                          idx_l                     = idx_l,
                                                                          idx_e                     = idx_e,
                                                                          data_order_dict           = data_order_dict,
                                                                          included_features_types   = included_features_types)
                    break

            if (enable_padding and
                error_signal_idx == None and
                    not (obj_camera[idx_c][0] == 0 and obj_camera[idx_c][1] == 0 and obj_camera[idx_c][2] == 0)):
                # No match between lidar an camera found, if padding=enabled, before no value was found and its not a 0-batched value
                error_signal_idx = [torch.tensor(padding_value)] * len(included_features_types)

            if error_signal_idx != None:
                error_signal.append(torch.stack(error_signal_idx).to(device))
                idx_e = int(obj_camera[idx_c][idx_pos_time].item())
                overlap_idx_list.append(idx_e)

        if enable_padding:
            # case: obj_camera < obj_lidar
            for idx_l in range(obj_lidar.shape[0]):
                error_signal_idx = None

                if (obj_lidar[idx_l][0] == 0 and obj_lidar[idx_l][1] == 0 and obj_lidar[idx_l][2] == 0):
                    break

                matches = (obj_camera[:, idx_pos_time] ==
                           obj_lidar[idx_l][idx_pos_time]).nonzero()
                if len(matches) == 0:
                    error_signal_idx = torch.stack(
                        [torch.tensor(-1*padding_value)] * len(included_features_types))
                    idx_e = int(obj_lidar[idx_l][idx_pos_time].item())
                    if obj_lidar[idx_l][idx_pos_time] < obj_camera[0][idx_pos_time]:
                        # Insert at beginning
                        error_signal.insert(0, error_signal_idx.to(device))
                        overlap_idx_list.insert(0, idx_e)
                    else:
                        # Append to end
                        error_signal.append(error_signal_idx.to(device))
                        overlap_idx_list.append(idx_e)

        error_signal_batch.append(torch.stack(error_signal))
        overlap_idx_batch.append(sorted(overlap_idx_list))

    error_signal_batch_pad_seq = torch.nn.utils.rnn.pad_sequence(error_signal_batch, batch_first=True, padding_value=0)

    return error_signal_batch_pad_seq, [torch.tensor(len(entry)) for entry in error_signal_batch], overlap_idx_batch


def calculate_error_signal_original(batch_lidar, batch_camera, included_features_types=['diff_x', 'diff_y']):
    batch_cohesion_table, batch_control_list = [], []

    # TODO get this information from the 'general_info' - data_order
    idx_pos_time = 0
    batch_error_signal_dict = {}
    for key in included_features_types:
        batch_error_signal_dict[key] = []

    for obj_camera, obj_lidar in zip(batch_camera, batch_lidar):
        error_signal, cohesion_table, control_list = {}, [], [
            ['Time camera', 'Time Lidar', 'Idx Camera', 'Idx Lidar', 'diff']]
        for features in included_features_types:
            error_signal[features] = []

        for idx_c in range(obj_camera.shape[0]):
            for idx_l in range(obj_lidar.shape[0]):
                if obj_camera[idx_c][2] == obj_lidar[idx_l][2]:

                    if idx_c != 0 and obj_lidar[idx_l][idx_pos_time] == 0 and obj_camera[idx_c][idx_pos_time] == 0:
                        break
                    error_signal = calculate_error_signal(
                        error_signal, obj_camera, obj_lidar, idx_c, idx_l, included_features_types)
                    cohesion_table.append([idx_c, idx_l])
                    control_list.append(
                        [obj_camera[idx_c][idx_pos_time], obj_lidar[idx_l][idx_pos_time], idx_c, idx_l])
                    break

        for key in error_signal:
            batch_error_signal_dict[key].append(torch.stack(error_signal[key]))
        batch_cohesion_table.append(cohesion_table)
        batch_control_list.append(control_list)

    batch_error_signal_pad_sequence_dict = {key: torch.nn.utils.rnn.pad_sequence(
        batch_error_signal_dict[key], batch_first=True, padding_value=0) for key in batch_error_signal_dict}

    return batch_error_signal_pad_sequence_dict, batch_cohesion_table, batch_control_list


def calculate_error_signal_pred(batch_lidar, batch_camera, batch_cohesion_table, error_signal_type=error_signal_types[0], included_features_types=['diff_x', 'diff_y']):
    batch_error_signal_dict = {}
    for features_key in included_features_types:
        batch_error_signal_dict[features_key] = []

    for obj_camera, obj_lidar, cohesion_table in zip(batch_camera, batch_lidar, batch_cohesion_table):
        error_signal = {}
        for features_key in included_features_types:
            error_signal[features_key] = []

        for idx_pair in cohesion_table:
            idx_c = idx_pair[0]
            idx_l = idx_pair[1]
            error_signal = calculate_error_signal(
                error_signal, obj_camera, obj_lidar, idx_c, idx_l, included_features_types)

        for key in error_signal:
            batch_error_signal_dict[key].append(torch.stack(error_signal[key]))
    batch_error_signal_pad_sequence_dict = {key: torch.nn.utils.rnn.pad_sequence(
        batch_error_signal_dict[key], batch_first=True, padding_value=0) for key in batch_error_signal_dict}

    return batch_error_signal_pad_sequence_dict
