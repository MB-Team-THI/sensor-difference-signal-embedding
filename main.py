import os
import json
import math
import wandb
from datetime import datetime
from fileinput import filename

from experiment import Experiment
from src.utils.json_files import check_dir_and_save_json

train_mode = True    # False -> eval-mode
dummy_example_enabled = False


save_config = False
meta_info = {"name":"advanced-autoencoder phase-1", 
             "description": "advanced-autoencoder phase 1: two encoders (LSTM or Transformer) that separately encodes the camera and lidar object; \
              two decoders (LSTM) separately reconstructing the trajectories; MSE-loss; save plot of original and reconstructed trajectory"}
base_dir  = ''
out_dir   = 'output_files\\models\\'

wandb_project_name = "sdse"

if not train_mode:
    # Load trained model for evaluation
    filename ='FILENAME'
    model_to_load_folder = filename.split('\\')[0]
    model_to_load_filename = base_dir + out_dir + filename


# Dataloader settings
features_to_load        = ['time_idx_in_scenario_frame', 'x', 'y', 'z', 'heading', 'v', 'size_x', 'size_y', 'size_z', 'radius', 'azimuth', 'radius_c', 'azimuth_c']
only_overlaying         = True
create_subdivisions     = False
representation_types    = ['object_pair_lut', 'object_pair']
representation_type     = representation_types[1]

# train number
# train hypers
# train_name = 'train_' + idx
training = {'idx': 0, 'traintest': [70, 30], 'num_gpus': 1, 'eval_epochs': [1,2, 100, 150, 200, 300],
            'enable_grad_clip': True, 'clip_value': 5, 'dummy_example': dummy_example_enabled, 
            'save_embeddings': ['train-set', 'test-set']}
# dataset number
# dataset hypers
dataset_train = [{'name':                   'nuscenes',
                  'augmentation_type':      ["no"],
                  'mode':                   'train',
                  'bbox_meter':             [200, 200],
                  'bbox_pixel':             [100,100],
                  'center_meter':           [100.0,100.0],
                  'hist_seq_first':         0,
                  'hist_seq_last':          49,
                  'features_to_load':       features_to_load,
                  'representation_type':    representation_type,
                  'only_overlaying':        only_overlaying,
                  'create_subdivisions':    create_subdivisions,
                  'subdivisions_params' :   {'distance_threshold': 30},
                  'rotation_type':          'ego',
                  'orientation':            'north'}]

dataset_test = [{'name':                    'nuscenes', 
                  'augmentation_type':      ["no"], 
                  'mode':                   'test', 
                  'bbox_meter':             [200, 200], 
                  'bbox_pixel':             [100, 100], 
                  'center_meter':           [100.0, 100.0],
                  'hist_seq_first':         0, 
                  'hist_seq_last':          49,
                  'features_to_load':       features_to_load,
                  'representation_type':    representation_type,
                  'only_overlaying':        only_overlaying,
                  'create_subdivisions':    create_subdivisions,
                  'subdivisions_params':    {'distance_threshold': 30},
                  'rotation_type':          'ego',
                  'orientation':            'north'}]


# dataloader number
# dataloader hypers

# For VICRec a batch_size >= 256 is probably required.
train_dataloader = {'idx':0, 'batch_size': 32, 'epochs': 400, 
                    'num_workers':8, 'shuffle':True, 'representation':'trajectory'}

test_dataloader  = {'idx':0, 'batch_size': 32, 'num_workers':8, 
                    'shuffle':False, 'representation':'trajectory'}

# model number
# model hypers

#model = {'idx':0,'model_depth':18, 'projector_dim': [1024, 2048]}
model = {'idx':                     0, 
         # ['Encoder', 'AE', 'AE_advanced_phase1' AE_advanced_phase2', 'AE_error_signal']
         'architecture_type':       "AE_error_signal",
         # processed_objects ['obj_pair', 'obj_lidar', 'obj_camera', 'recon_error_sgn', 'error_signal']
         'architecture_args':       {'processed_objects':      'obj_pair',
                                     'error_signal_decoder':   True},
         'encoderI_type':           "ResNet-18",
         # ["LSTM-Encoder", "Transformer-Encoder", "CNN-Encoder"]
         'encoderT_type':           "Transformer-Encoder",
         'merge_type':              "FC",
         'z_dim_m':                 8,
         'projection_head_args':    {'type':     'MLP',
                                     'n_layers': 3},
         'decoderT_type':           "LSTM-Decoder",
         'encoderT_args':           {'depth':       6,
                                     'heads':       8,
                                     'dim_trans':   128, 
                                     'dim_mlp':     128,
                                     'single_feature_encoding': True},
         'merge_args':              {'n_layers':    4},
         'decoderT_args':           {'image_width':         256,
                                     'image_height':        128,
                                     'n_image_channels':    2,
                                     'bbox_pixel':          [100, 100],
                                     'num_layers':          4},
         }


data_adaption_types = ['normalization', 'normalization_log']
data_adaption_type  = data_adaption_types[0]
# max values for train+val set
norm_max_values      = {'diff_x':                   32.3455,
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

if train_mode:
    run_name = (datetime.today().strftime('%Y%m%d') + '_' + 
                model['architecture_type'] 
                + '_encoder_' + model['encoderT_type'] 
                + '_decoder_' + model['decoderT_type'] 
                + '_eps_' + str(train_dataloader['epochs']) + '_' 
                + model['architecture_args']['processed_objects'] 
                + '_z' + str(model['z_dim_m'])
                )
else:
    # eval mode
    run_name = model_to_load_folder

# eval number
# eval hypers
# TODO futher adapt eval_0.py for the "AE_advanced_phase2" as well

output_dir = base_dir + out_dir + run_name
evaluation = {'idx':                                0,
              'architecture_type':                  model['architecture_type'],
              'decoderT_type':                      model['decoderT_type'],
              'processed_objects':                  model['architecture_args']['processed_objects'],
              'output_dir':                         output_dir,
              'perform_embedding_space_evaluation': True,
              'plot_and_save_images':               True,
              'dummy_example':                      dummy_example_enabled,
              'data_adaption':                      data_adaption_type,
              'norm_max_values':                    norm_max_values}

# optimizer number
# optimizer hypers
optimizer = {'idx':          1, 
             'lr':           2e-4, 
             'weight_decay': 0, 
             'betas':        (0.9, 0.999)}

scheduler = None

# loss number
# loss hypers
loss = {'idx':                   0,
        'architecture_type':     model['architecture_type'],
        'decoderT_type':         model['decoderT_type'],
        'processed_objects':     model['architecture_args']['processed_objects'],
        # ['reconstruction_loss', 'VICReg', 'SiamSim']
        'loss_type':             'reconstruction_loss',      
        # ['obj_pair', 'obj_lidar', 'obj_camera', 'calc_error_signal', 'recon_error_signal']
        'loss_target':           ['recon_error_signal'],
        #  ['diff_x', 'diff_y', 'directed_distance', 'euclidean_distance', 'mean_distance_to_ego_x', 'mean_distance_to_ego_y', 'diff_heading', 'diff_v', 'diff_dist_obj_ego', 'mean_dist_obj_ego']
        # Only use the features 'mean_distance_to_ego_x', 'mean_distance_to_ego_y' for the 'AE_error_signal'
        'error_signal_features': ['diff_x', 'diff_y'], # ,'diff_v', 'mean_eucl_dist_obj_ego', 'diff_x', 'diff_y', 'diff_radius_c', 'diff_azimuth_c']
        'enable_padding':        False,
        'padding_value':         15,
        'data_adaption':         data_adaption_type,
        'norm_max_values':       norm_max_values,
		'enable_kmeans_loss':    True,
        'learn_cluster_rep':     True,
        'kmeans_loss_pretrain_ep': 25,
		'z_dim_m':               model['z_dim_m'],
		'n_clusters':            22,
        'val_lambda':            1,
		'alpha':                 100,
		'h_init_range':          [-1.0, 1.0],
        }      
                             
model['error_signal_args'] = {'error_signal_features':    loss['error_signal_features'],
                              'enable_padding':           loss['enable_padding'],
                              'padding_value':            loss['padding_value'],
                              'data_adaption':            loss['data_adaption'],
                              'norm_max_values':          loss['norm_max_values'],
                             } 

model['decoderT_args']['error_signal_features'] = loss['error_signal_features']
evaluation['error_signal_features'] = loss['error_signal_features']


if __name__ == '__main__':

    run = wandb.init(
        project = wandb_project_name,
        notes   = "",
        tags    = ["baseline", "paper1"],
        config  = {
            "run_name":         run_name,
            "train_dataloader": train_dataloader,
            "test_dataloader":  test_dataloader,
            "dataset_train":    dataset_train,
            "model":            model,
            "training":         training,
            "optimizer":        optimizer,
            "loss":             loss,
            "meta_info":        meta_info,
        }
    )


    load_experiment_settings = False
    if load_experiment_settings:
        experiment = Experiment.from_config_file("./saved_configs/name123_09-04-2021_13-52-33.pkl")
    
    else:
        experiment = Experiment(meta_info=meta_info,
                                dataset_param_train=dataset_train,
                                dataset_param_test=dataset_test,
                                trdataloader_param=train_dataloader,
                                tedataloader_param=test_dataloader,
                                model_param=model,
                                training_param=training,
                                evaluation_param=evaluation,
                                optimizer_param=optimizer,
                                scheduler_param=scheduler,
                                loss_param=loss,
                                run_name=run_name)
        # TODO
        # experiment.save_experiment_config(add_timestamp=True)
    

    if save_config:
        # Get number of parameters
        num_params_total_model = sum(p.numel() for p in experiment.model.parameters() if p.requires_grad)
        num_params_encoder = sum(p.numel() for p in experiment.model.model.encoder.parameters() if p.requires_grad)
        num_params_decoder = sum(p.numel() for p in experiment.model.model.decoder.parameters() if p.requires_grad)
        model_parameter_size = {'num_params_total_model': num_params_total_model,
                                'num_params_encoder':     num_params_encoder,
                                'num_params_decoder':     num_params_decoder}
        wandb_settings = {'project_name':   wandb_project_name,
                          'run_name_orig':  run.name,
                          'run_id':         run.id,
                          'link':           'https://wandb.ai/' + run.path}

        parameter_settings = {
                'meta_info':                meta_info,
                'dataset_param_train':      dataset_train,
                'dataset_param_test':       dataset_test,
                'train_dataloader_param':   train_dataloader,
                'test_dataloader_param':    test_dataloader,
                'model_param':              model,
                'training_param':           training,
                'evaluation_param':         evaluation,
                'optimizer_param':          optimizer,
                'scheduler_param':          scheduler,
                'loss_param':               loss,
                'num_model_parameters':     model_parameter_size,
                'run_name':                 run_name,
                'wandb':                    wandb_settings,
        }


        
    if train_mode:
        if save_config:
            check_dir_and_save_json(output_dir  = output_dir, 
                                    filename    ='training_settings.json', 
                                    output_data = parameter_settings)

            print('Experiment config saved')

        experiment.train()

        # Save model after training
        filename = output_dir + '\\model_' + str(training['idx']) + '.pt'
        experiment.save_checkpoint(filename)
        print('Training complete - Model saved')

    else:
        experiment.load_checkpoint(model_to_load_filename)
        print('Model loaded')


    print('Evaluation started...')
    experiment.evaluate()
    print('Evaluation complete - Embeddings saved')

   
