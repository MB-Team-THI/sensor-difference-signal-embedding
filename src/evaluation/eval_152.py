import os
import json
import wandb
import random
import numpy as np
import torch



from src.evaluation.eval import eval
from src.utils.difference_signal import calculate_error_signal_full_with_padding
from src.utils.distance_between_objects import get_distance_to_ego 
from src.utils.dimensionality_reduction import dim_reduction 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

architecture_types      = ["Encoder", "AE", "AE_advanced_phase1", "AE_advanced_phase2", "AE_error_signal"]
decoderT_types          = ['LSTM-Decoder', 'CNN-Decoder', 'MLP']
processed_objects_types = ['obj_pair', 'obj_lidar', 'obj_camera']


possible_classes        = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck', 'unknown']
possible_categories     = ["human.pedestrian.adult", "human.pedestrian.child", "human.pedestrian.wheelchair", "human.pedestrian.stroller", "human.pedestrian.personal_mobility", "human.pedestrian.police_officer", 
                           "human.pedestrian.construction_worker", "animal", "vehicle.car", "vehicle.motorcycle", "vehicle.bicycle", "vehicle.bus.bendy", "vehicle.bus.rigid", "vehicle.truck", "vehicle.construction", 
                           "vehicle.emergency.ambulance", "vehicle.emergency.police", "vehicle.trailer", "movable_object.barrier", "movable_object.trafficcone", "movable_object.pushable_pullable", 
                           "movable_object.debris",   "static_object.bicycle_rack", "not_available"]
possible_attributes     = ["vehicle.moving", "vehicle.stopped", "vehicle.parked", "cycle.with_rider", "cycle.without_rider",
                           "pedestrian.sitting_lying_down", "pedestrian.standing", "pedestrian.moving", "not_available"]
possible_weather_odd    = ['no_rain', 'rain']
possible_day_night_odd  = ['day', 'night']


counter_gt_na = 0

def get_emb_dimensions(model, dataset_local):
    input_data = dataset_local[0].__getitem__(1)
    traj_sample = []
    input_data['obj_camera'] = np.array([line[0:10] for line in input_data['obj_camera']])
    obj_sample =  torch.from_numpy(input_data['obj_camera']).to(torch.float).t()
    traj_sample = torch.cat((obj_sample, obj_sample), 1)
    traj_sample = traj_sample.flatten()
    z_sample = model.get_embeddings(traj_sample.cuda())
    return z_sample.shape[0]


def count_label_dist(labels, possible_labels):
    # More complicated count method in order to counts if a value has zero appearances

    uniques, counts = np.unique(labels, return_counts=True)
    counter_all = np.zeros(len(possible_labels))

    for idx, _ in enumerate(counter_all):
        if idx in uniques:
            counter_all[idx] = counts[uniques.tolist().index(idx)]

    counter_all = [int(x) for x in counter_all]

    return counter_all


def setup_description_dicts(description_dict, label_dict):
    ### Create description files    
    dummy_description = ['Dummy0', 'Dummy1', 'Dummy2', 'Dummy3', 'Dummy4']
    counts_dummys = count_label_dist(labels = label_dict['dummy'], possible_labels = dummy_description)
    description_dict['dummy'] = {'name':            'dummy',
                                 'ranges':          [[x, x] for x in range(len(dummy_description))],
                                 'description':     dummy_description,
                                 'clustercount':    counts_dummys
                                }
    
    # ranges just work for integers (classes) not float values
    description_dict['dummy_ranges'] = {'name':            'dummy_range',
                                        'ranges':          [[0, 4], [5, 9]],
                                        'description':     ['DummyRange1', 'DummyRange2'],
                                        'clustercount':    [1, 1]
                                      }

    # gt_pred_class
    counts_classes = count_label_dist(labels = label_dict['gt_pred_class'], possible_labels = possible_classes)
    description_dict['gt_pred_class'] = {'name':            'gt_pred_class',
                                         'ranges':          [[x, x] for x in range(len(possible_classes))],
                                         'description':     possible_classes,
                                         'clustercount':    counts_classes
                                        }
    
    # gt_visibility
    counts_vis = count_label_dist(labels = label_dict['gt_visibility'], possible_labels = range(0, 4))
    description_dict['gt_visibility'] = {'name':            'gt_visibility',
                                         'ranges':          [[0, 0], [1, 1], [2, 2], [3, 3]],
                                         'description':     ['v0-40', 'v40-60', 'v60-80', 'v80-100'],
                                         'clustercount':    counts_vis
                                        }
    
    # gt_category
    counts_category = count_label_dist(labels = label_dict['gt_category'], possible_labels = possible_categories)
    description_dict['gt_category'] = {'name':             'gt_category',
                                       'ranges':           [[x, x] for x in range(len(possible_categories))],
                                       'description':      possible_categories,
                                       'clustercount':     counts_category
                                      }

    # gt_attribute
    counts_attributes = count_label_dist(labels = label_dict['gt_attribute'], possible_labels = possible_attributes)
    description_dict['gt_attribute'] = {'name':            'gt_attribute',
                                        'ranges':          [[x, x] for x in range(len(possible_attributes))],
                                        'description':     possible_attributes,
                                        'clustercount':    counts_attributes
                                       }
     
    # ego_v_mean    
    label_ego_mean_kmh = [x*3.6 for x in label_dict['ego_v_mean']]
    # Max. ego- velocity is 43 km/h -> 9 ranges each 5 km/h, v_ego is in m/s
    description_v_ego = ['0-5 km/h',   '5-10 km/h',  '10-15 km/h', '15-20 km/h', '20-25 km/h', '25-30 km/h', 
                         '30-35 km/h', '35-40 km/h', '40-45 km/h', '45-50 km/h', '50-55 km/h', '55-60 km/h',
                         '60-65 km/h', '65-70 km/h', '70-75 km/h', '75-80 km/h']
    v_ego_limits = [x*5 for x in range(len(description_v_ego)+1)]
    v_ego_ranges = [[v_ego_limits[idx], v_ego_limits[idx+1]] for idx in range(len(v_ego_limits)-1)]

    new_v_ego_labels = []
    for val in label_ego_mean_kmh:
        for idx, v_range in enumerate(v_ego_ranges):
            if v_range[0] <= val < v_range[1]:
                new_v_ego_labels.append(idx)
    assert len(label_dict['ego_v_mean']) == len(new_v_ego_labels), "some values are lost here - the same length must be kept"
    label_dict['ego_v_mean'] = new_v_ego_labels

    counts_v_ego = count_label_dist(labels=label_dict['ego_v_mean'], possible_labels = description_v_ego)
    description_dict['ego_v_mean'] = {'name':           'ego_v_mean',
                                      'ranges':         [[x, x] for x in range(len(description_v_ego))],
                                      'description':    description_v_ego,
                                      'clustercount':   counts_v_ego
                                      }
    

    # gt_v_mean
    label_gt_v_mean_kmh = [x*3.6 for x in label_dict['gt_v_mean']]
    # Max gt-velocity is 60.7 km/h -> 12 ranges each 5 km/h (last 6 km/h), gt_v_mean is in m/s
    description_v_gt = ['0-5 km/h',   '5-10 km/h',  '10-15 km/h', '15-20 km/h', '20-25 km/h', '25-30 km/h', 
                        '30-35 km/h', '35-40 km/h', '40-45 km/h', '45-50 km/h', '50-55 km/h', '55-60 km/h',
                        '60-65 km/h', '65-70 km/h', '70-75 km/h', '75-80 km/h']
    v_gt_limits      = [x*5 for x in range(len(description_v_gt)+1)]
    v_gt_limits[-1]  = v_gt_limits [-1]+(1) # last range is up to 6 km/h    
    v_gt_ranges      = [[v_gt_limits[idx], v_gt_limits[idx+1]] for idx in range(len(v_gt_limits)-1)]

    new_v_gt_labels = []
    for val in label_gt_v_mean_kmh:
        for idx, v_range in enumerate(v_gt_ranges):
            if v_range[0] <= val < v_range[1]:
                new_v_gt_labels.append(idx)  
    assert len(label_dict['gt_v_mean']) == len(new_v_gt_labels), "some values are lost here - the same length must be kept"              
    label_dict['gt_v_mean'] = new_v_gt_labels

    counts_v_gt = count_label_dist(labels=label_dict['gt_v_mean'], possible_labels = description_v_gt)
    description_dict['gt_v_mean'] = {'name':           'gt_v_mean',
                                     'ranges':         [[x, x] for x in range(len(description_v_gt))],
                                     'description':    description_v_gt,
                                     'clustercount':   counts_v_gt
                                    }

    # gt_num_lidar_pts_mean
    num_lidar_ranges = [[-1,0], [0,2], [2,4], [4,7], [7,10], [10,25], [25,50], [50,100], [100,200], [200, 500], [500, 1000], [1000,3000], [3000, 11000]]
    new_num_lidar_labels = []
    for val in label_dict['gt_num_lidar_pts_mean']:
        for idx, num_range in enumerate(num_lidar_ranges):
            if num_range[0] < val <= num_range[1]:
                new_num_lidar_labels.append(idx)
    assert len(label_dict['gt_num_lidar_pts_mean']) == len(new_num_lidar_labels), "some values are lost here - the same length must be kept"   
    label_dict['gt_num_lidar_pts_mean'] = new_num_lidar_labels


    description_num_lidar = [str(x[0])+'-'+str(x[1])+' lidar hits' for x in num_lidar_ranges]
    description_num_lidar[0] = '0 lidar hits'
    counts_num_lidar = count_label_dist(labels=label_dict['gt_num_lidar_pts_mean'], possible_labels=description_num_lidar)
    description_dict['gt_num_lidar_pts_mean'] = {'name':           'gt_num_lidar_pts_mean',
                                                 'ranges':         [[x, x] for x in range(len(description_num_lidar))],
                                                 'description':    description_num_lidar,
                                                 'clustercount':   counts_num_lidar
                                                }

    ### Tracking scores
    tracking_score_ranges = [[0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1]]
    # Tracking Scores Camera
    new_tracking_scores_camera_labels = []
    for val in label_dict['tracking_scr_camera']:
        for idx, score_range in enumerate(tracking_score_ranges):
            if score_range[0] < val <= score_range[1]:
                new_tracking_scores_camera_labels.append(idx)
    assert len(label_dict['tracking_scr_camera']) == len(new_tracking_scores_camera_labels), "some values are lost here - the same length must be kept"   
    label_dict['tracking_scr_camera'] = new_tracking_scores_camera_labels

    description_tracking_scr_camera = [str(x[0]) + ' - ' + str(x[1]) for x in tracking_score_ranges]
    counts_tracking_scr_camera = count_label_dist(labels=label_dict['tracking_scr_camera'], possible_labels=tracking_score_ranges)
    description_dict['tracking_scr_camera'] = {'name':           'tracking_scores_camera',
                                               'ranges':         [[x, x] for x in range(len(description_tracking_scr_camera))],
                                               'description':    description_tracking_scr_camera,
                                               'clustercount':   counts_tracking_scr_camera
                                               }

    # Tracking Scores LiDAR
    new_tracking_scores_lidar_labels = []
    for val in label_dict['tracking_scr_lidar']:
        for idx, score_range in enumerate(tracking_score_ranges):
            if score_range[0] < val <= score_range[1]:
                new_tracking_scores_lidar_labels.append(idx)
    assert len(label_dict['tracking_scr_lidar']) == len(new_tracking_scores_lidar_labels), "some values are lost here - the same length must be kept"   
    label_dict['tracking_scr_lidar'] = new_tracking_scores_lidar_labels

    description_tracking_scr_lidar = [
        str(x[0]) + ' - ' + str(x[1]) for x in tracking_score_ranges]
    counts_tracking_scr_lidar = count_label_dist(labels=label_dict['tracking_scr_lidar'], possible_labels=tracking_score_ranges)
    description_dict['tracking_scr_lidar'] = {'name':            'tracking_scores_lidar',
                                               'ranges':         [[x, x] for x in range(len(description_tracking_scr_lidar))],
                                               'description':    description_tracking_scr_lidar,
                                               'clustercount':   counts_tracking_scr_lidar
                                             }

    # Tracking Scores Mean (of lidar and camera)
    new_tracking_scores_mean_labels = []
    for val in label_dict['tracking_scr_mean']:
        for idx, score_range in enumerate(tracking_score_ranges):
            if score_range[0] < val <= score_range[1]:
                new_tracking_scores_mean_labels.append(idx)
    assert len(label_dict['tracking_scr_mean']) == len(new_tracking_scores_mean_labels), "some values are lost here - the same length must be kept"   
    label_dict['tracking_scr_mean'] = new_tracking_scores_mean_labels

    description_tracking_scr_mean =  [str(x[0]) + ' - ' + str(x[1]) for x in tracking_score_ranges]
    counts_tracking_scr_mean = count_label_dist(labels=label_dict['tracking_scr_mean'], possible_labels=tracking_score_ranges)
    description_dict['tracking_scr_mean'] = {'name':           'tracking_scores_mean',
                                             'ranges':         [[x, x] for x in range(len(description_tracking_scr_mean))],
                                             'description':    description_tracking_scr_mean,
                                             'clustercount':   counts_tracking_scr_mean
                                            }


    ### ODD description and labels    
    # odd_weather
    counts_category = count_label_dist(labels=label_dict['odd_weather'], possible_labels = possible_weather_odd)
    description_dict['odd_weather'] = {'name':             'odd_weather',
                                       'ranges':           [[x, x] for x in range(len(possible_weather_odd))],
                                       'description':      possible_weather_odd,
                                       'clustercount':     counts_category
                                      }

    # odd_day_night
    counts_attributes = count_label_dist(labels=label_dict['odd_day_night'], possible_labels = possible_day_night_odd)
    description_dict['odd_day_night'] = {'name':            'odd_day_night',
                                         'ranges':          [[x, x] for x in range(len(possible_day_night_odd))],
                                         'description':     possible_day_night_odd,
                                         'clustercount':    counts_attributes
                                        }

    # odd_distance_ego_obj
    # Max odd_distance_ego_obj is 49 m
    description_dist =  ['0-5 m',   '5-10m',  '10-15m', '15-20 m', '20-25 m', '25-30 m', 
                         '30-35 m', '35-40 m', '40-45 m', '45-50 m', '50-55 m', '55-60 m',
                         '60-65 m', '65-70 m', '70-75 m', '75-80 m']
    dist_limits      = [x*5 for x in range(len(description_dist)+1)]
    dist_ranges      = [[dist_limits[idx], dist_limits[idx+1]] for idx in range(len(dist_limits)-1)]

    new_distance_obj_ego_labels = []
    for val in label_dict['odd_distance_ego_obj']:
        for idx, dist_range in enumerate(dist_ranges):
            if dist_range[0] <= val < dist_range[1]:
                new_distance_obj_ego_labels.append(idx)
    # TODO
    # assert len(label_dict['odd_distance_ego_obj']) == len(new_distance_obj_ego_labels), "some values are lost here - the same length must be kept"              
    label_dict['odd_distance_ego_obj'] = new_distance_obj_ego_labels

    counts_dist = count_label_dist(labels=label_dict['odd_distance_ego_obj'], possible_labels = description_dist)
    description_dict['odd_distance_ego_obj'] = {'name':           'odd_distance_ego_obj',
                                                'ranges':         [[x, x] for x in range(len(description_dist))],
                                                'description':    description_dist,
                                                'clustercount':   counts_dist
                                                }
    

    ### gIoU scores
    gIoU_score_ranges = [[-1.0, -0.8], [-0.8, -0.6], [-0.6, -0.4], [-0.4, -0.2], [-0.2, 0.0], 
                         [0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]]
    # gIoU_mean
    new_gIoU_mean = []
    for val in label_dict['gIoU_mean']:
        for idx, score_range in enumerate(gIoU_score_ranges):
            if score_range[0] < val <= score_range[1]:
                new_gIoU_mean.append(idx)
    assert len(label_dict['gIoU_mean']) == len(new_gIoU_mean), "some values are lost here - the same length must be kept"   
    label_dict['gIoU_mean'] = new_gIoU_mean

    description_gIoU_scr_mean = [str(x[0]) + ' - ' + str(x[1]) for x in gIoU_score_ranges]
    counts_gIoU_scr_mean      = count_label_dist(labels=label_dict['gIoU_mean'], possible_labels=gIoU_score_ranges)
    description_dict['gIoU_mean'] = {'name':           'gIoU_c_l_mean',
                                     'ranges':         [[x, x] for x in range(len(description_gIoU_scr_mean))],
                                     'description':    description_gIoU_scr_mean,
                                     'clustercount':   counts_gIoU_scr_mean
                                    }


    return description_dict, label_dict


def add_current_sample_to_label_dict(label_dict, traj_pair, input_data, dummy_label=0):
    for idx in range(len(traj_pair['obj_camera'])):
        scene_info  = input_data['scene_info'][idx]
        obj_gt      = input_data['obj_gt_org'][idx]
        obj_ego     = input_data['obj_ego_org'][idx]
        

        label_dict['dummy'].append(dummy_label)
        label_dict['dummy_ranges'].append(random.randint(0, 9))


        label_dict['ego_v_mean'].append(np.round(np.mean(obj_ego['v']), 4))

        if scene_info['rain_no_rain'].strip() in possible_weather_odd:
            label_dict['odd_weather'].append(possible_weather_odd.index(scene_info['rain_no_rain'].strip()))
        else:
            print("odd_weather: check this")

        if scene_info['day_night'].strip() in possible_day_night_odd:
            label_dict['odd_day_night'].append(possible_day_night_odd.index(scene_info['day_night'].strip()))
        else:
            print("odd_day_night: check this")

        distance_ego_objects = get_distance_to_ego(obj_ego      = traj_pair['obj_ego'][idx], 
                                                   obj_camera   = traj_pair['obj_camera'][idx],
                                                   obj_lidar    = traj_pair['obj_lidar'][idx],
                                                   metric       = 'euclidean', 
                                                   calc         = 'mean',
                                                   data_order   = input_data['general_info'][idx])
        label_dict['odd_distance_ego_obj'].append(distance_ego_objects)

        if obj_gt != []:
            most_frequent_visibility_token = int(np.bincount(obj_gt['visibility_token'].tolist()).argmax() - 1)
            label_dict['gt_visibility'].append(most_frequent_visibility_token)
            label_dict['gt_v_mean'].append(np.round(np.mean(obj_gt['v']), 4))
            label_dict['gt_num_lidar_pts_mean'].append(np.mean(obj_gt['num_lidar_pts']))
            label_dict['tracking_scr_camera'].append(np.mean(obj_gt['tracking_score_camera']))
            label_dict['tracking_scr_lidar'].append(np.mean(obj_gt['tracking_score_lidar']))
            label_dict['tracking_scr_mean'].append(np.mean([obj_gt['tracking_score_lidar'], obj_gt['tracking_score_camera']]))
            # label_dict['gt_num_radar_pts_mean'].append(int(np.mean(obj_gt['num_radar_pts'])))

            if obj_gt['pred_class'].strip() in possible_classes:
                label_dict['gt_pred_class'].append(possible_classes.index(obj_gt['pred_class'].strip()))
            else:
                print("gt_pred_class: check this")                

            if obj_gt['gt_attribute'].strip() in possible_attributes:
                label_dict['gt_attribute'].append(possible_attributes.index(obj_gt['gt_attribute'].strip()))
            else:
                print("gt_attribute: check this") 

            if obj_gt['gt_category'].strip() in possible_categories:
                label_dict['gt_category'].append(possible_categories.index(obj_gt['gt_category'].strip()))
            else:
                print("gt_category: check this") 

        else:
            label_dict['gt_v_mean']                        .append(-1)
            label_dict['gt_visibility_token_most_frequent'].append(-1)
            label_dict['gt_num_lidar_pts_mean']            .append(-1)
            label_dict['gt_num_radar_pts_mean']            .append(-1) 
            label_dict['gt_pred_class']                    .append(-1)
            label_dict['gt_attribute']                     .append(-1)
            label_dict['gt_category']                      .append(-1)
            print("gt_na")

        return label_dict


def save_eval_results(output_dir, res_embeddings, embedding_order, label_dict, description_dict, 
                      res_diff_sgns, res_names, model, epoch, test_file_order):

    ### Save Model
    filename = output_dir + "model_ep" + str(epoch) + ".pt"
    torch.save(model.state_dict(), filename)

    ### Save embeddings and labels ###
    assert len(res_embeddings) == len(label_dict['ego_v_mean']), "Embeddings and labels are not of same length"
    output_emb ={"name":   "embeddings",
                 "points": res_embeddings,
                 "order":  embedding_order}

    # Embeddings
    filename = output_dir + "embeddings_original_dim.json"
    with open(filename, 'w') as f:
        json.dump(output_emb, f)

    # Labels
    for idx, key in enumerate(label_dict):
        current_labels = label_dict[key]
        if type(current_labels) != list:
            current_labels = current_labels.tolist()
        filename = output_dir + "labels_" + str(idx) + ".json"
        with open(filename, 'w') as f:
            json.dump(current_labels, f)

    # Difference Signals
    assert len(res_diff_sgns) == len(label_dict['ego_v_mean']), "Embeddings and labels are not of same length"
    output_diff_sgns ={"name":   "difference_signals",
                       "points": res_diff_sgns,
                       "order":  embedding_order}
    filename = output_dir + "difference_signals.json"
    with open(filename, 'w') as f:
        json.dump(output_diff_sgns, f)

    # Descriptions
    for idx, key in enumerate(description_dict):
        current_description = description_dict[key]
        filename = output_dir + "description_" + str(idx) + ".json"
        with open(filename, 'w') as f:
            json.dump(current_description, f)

    # Save embedding order
    filename = output_dir + "embedding_order.json"
    with open(filename, 'w') as f:
        json.dump(embedding_order, f)

    if test_file_order:
        # Save all scene_names in a json file and comparing the two files.
        filename = output_dir + "record_file_order_embedding_generator.json"
        with open(filename, 'w') as f:
            json.dump(res_names, f)



class eval_152(eval):
    def __init__(self,
                 idx                                = 152,
                 architecture_type                  = architecture_types[0],
                 decoderT_type                      = decoderT_types[0],
                 processed_objects                  = processed_objects_types[0],
                 perform_embedding_space_evaluation = False,
                 plot_and_save_images               = False,
                 dummy_example                      = False,
                 data_adaption                      = [],
                 norm_max_values                    = {},
                 error_signal_features              = [],
                 output_dir                         = '',
                 name                               = 'eval approach 152',
                 input_     	                    = 'dataset reference and trained model',
                 output                             = 'embeddings, labels and image (original and reconstructed trajectory)',
                 description                        = 'Apply the whole dataset to the model, save embeddings after the encoder (latent-space) for the UMAP visualization, create pseudo-labels for the UMAP visualization, and save images (of original and reconstructed trajectory) for visualization.',
                 pred_                              = 50
                 ):
        super().__init__(idx, name, input_, output, description)
        
        assert decoderT_type in decoderT_types, 'decoderT keyword unknown'
        assert processed_objects in processed_objects_types, 'processed_objects keyword unknown'

        self.architecture_type                  = architecture_type
        self.decoderT_type                      = decoderT_type
        self.processed_objects                  = processed_objects
        self.plot_and_save_images               = plot_and_save_images
        self.dummy_example                      = dummy_example
        self.dummy_rand_vals                    = []
        self.data_adaption                      = data_adaption
        self.norm_max_values                    = norm_max_values
        self.error_signal_features              = error_signal_features
        self.perform_embedding_space_evaluation = perform_embedding_space_evaluation
        self.output_dir                         = output_dir
        self.pred_frames                        = pred_
        self.device                             = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.test_file_order                    = True


    def _init_dummy_vals(self, input_vals):
        self.dummy_rand_vals = input_vals

    def __call__(self, model, dataset_test_dict, dataloader_test, run_name, epoch=0, dummy_rand_vals=[]):
        self._evaluate(model, dataset_test_dict, dataloader_test, run_name, epoch, dummy_rand_vals)

    
    def _evaluate(self, model, dataset_test_dict, dataloader_test, run_name, epoch=0, dummy_vals=[]):
        '''
        Two-fold evaluation:
            - Saving of the reconstructed occupancy grid
            - Saving of the embeddings for the umap visualization
        '''
        model.to(self.device)
        model.eval()

        batch_size = dataloader_test.batch_size
        res_embeddings  = []
        res_diff_sgns   = []
        embedding_order = []
        key_order       = ['dummy', 'dummy_ranges', 'gt_pred_class', 'gt_visibility', 'gt_category', 'gt_attribute', 'ego_v_mean',
                           'gt_num_lidar_pts_mean', 'gt_v_mean', 'tracking_scr_camera', 'tracking_scr_lidar', 'tracking_scr_mean',
                           'odd_day_night', 'odd_weather', 'odd_distance_ego_obj', 'gIoU_mean'] 

        label_dict       = {key: [] for key in key_order}
        description_dict = {key: {} for key in key_order}
        res_names = []

        if dummy_vals != []:
            self._init_dummy_vals(dummy_vals)

        output_dir = self.output_dir + "\\epoch_"+str(epoch) + "\\" + dataset_test_dict[0]['mode'] +"\\"
        output_dir_images = output_dir + "\\images\\"
        if not os.path.exists(output_dir_images):
            os.makedirs(output_dir_images)

        dataloader = dataloader_test(epoch, rank=0)
        for batch_idx, input_data in enumerate(dataloader):    
            # Data Preprocessing -------------------------------------------------  
            obj_gt      = input_data['obj_gt_org']            
            assert obj_gt != [], ("Unexpected, only for the test-set the gt_obj should not be available obj-id" + str(input_data['obj_pair_id']))

            traj_pair = {'obj_camera':       input_data['obj_camera'], 
                         'obj_lidar':        input_data['obj_lidar'],
                         'obj_ego':          input_data['obj_ego'],
                         'data_order_dict':  input_data['general_info'][0],}    
                  

            # Network execution -------------------------------------------------   
            [embeddings, x_pred] = model(traj_pair)
            
            # Difference Signal ------------------------------------------------------
            difference_signal, _, _ = calculate_error_signal_full_with_padding(batch_camera            = traj_pair['obj_camera'],
                                                                               batch_lidar             = traj_pair['obj_lidar'],
                                                                               batch_ego               = traj_pair['obj_ego'],
                                                                               data_order_dict         = traj_pair['data_order_dict'],
                                                                               included_features_types = self.error_signal_features)
            for item in difference_signal:
                difference_signal_dict = {}
                for idx, key in enumerate(self.error_signal_features):
                    difference_signal_dict[key] = [x[idx] for x in item.tolist()]
                res_diff_sgns.append(difference_signal_dict)
            
 
            for idx, item in enumerate(embeddings):
                sample_idx = batch_size*batch_idx + idx
                # Embeddings -------------------------------------------------------       
                res_embeddings.append(item.squeeze().cpu().detach().numpy().tolist())


                # Embeddings Order -------------------------------------------------
                # Save the order in which labels and embeddings are processed
                current_obj_pair = {'obj_pair_name':        input_data['obj_pair_name'][idx],
                                    'obj_pair_global_idx':  int(input_data['obj_pair_global_idx'][idx]),
                                    'processing_idx':       int(sample_idx) }
                embedding_order.append(current_obj_pair)
            
                if self.test_file_order:
                    # test if the index order is the same for main_image_generator.py and main_embedding_generator.py
                    res_names.append(str(input_data['obj_pair_name'][idx]))
            
                # Labels ----------------------------------------------------------
                # Add GT and meta-information to label dict
                label_dict = add_current_sample_to_label_dict(label_dict, traj_pair, input_data)


        # Description ------------------------------------------------------
        # Save matching description for labels
        # especially for pred_class, gt_category, gt_attribute
        description_dict, label_dict = setup_description_dicts(description_dict, label_dict)

        # Perform Dimensionality Reduction
        dim_reduction(embeddings_original = res_embeddings, output_dir=output_dir)



        ####################################################################
        ########--------- SAVE RESULTS OF THIS MODEL STAGE ---------########
        ####################################################################
        save_eval_results(output_dir        = output_dir, 
                          res_embeddings    = res_embeddings, 
                          embedding_order   = embedding_order, 
                          label_dict        = label_dict, 
                          description_dict  = description_dict, 
                          res_diff_sgns     = res_diff_sgns, 
                          res_names         = res_names,
                          model             = model, 
                          epoch             = epoch, 
                          test_file_order   = self.test_file_order)    

        model.train()  
