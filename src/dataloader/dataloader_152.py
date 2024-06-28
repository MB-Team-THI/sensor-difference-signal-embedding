import random
from scipy.io import loadmat
from glob import glob
import os

import numpy as np
import torch
import torchnet as tnt
import math
import einops



class dataloader_152(object):
    def __init__(
        self,
        idx=152,
        dataset=None,
        batch_size=None,
        epochs=None,
        num_workers=0,
        num_gpus=1,
        shuffle=False,
        epoch_size=None,
        transformation=None,
        transformation3D=None,
        representation='TODO',
        test=False,
        grid_chosen=None,
        name='TODO',
        description='TODO'
    ):
        self.dataset = dataset[0]
        self.epoch_size = epoch_size if epoch_size is not None else len(self.dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers,
        self.epochs = epochs
        
        self.test = test
        self.num_gpus = num_gpus

    def _load_function(self, idx):
        idx     = idx % len(self.dataset)
        sample  = self.dataset[idx]

        if False:
            # Filtering for fixed length of autoencoder input 
            # TODO workaround for MLP Autoencoder of fixed length
            if len(sample['obj_lidar'][0]) > 9:
                sample['obj_camera'] = np.array([line[0:10] for line in sample['obj_camera']])
                sample['obj_lidar'] = np.array([line[0:10] for line in sample['obj_lidar']])
            else:
                new_idx = random.choice(range(len(self.dataset)))
                sample = self._load_function(new_idx)

        return sample

    def _collate_fun(self, batch):
        obj_pair_global_idx = []
        obj_pair_name       = []
        general_info        = []
        scene_info          = []
        map_info            = []
        obj_camera          = []
        obj_lidar           = []
        obj_ego             = []
        obj_gt              = []
        obj_ego_org         = []
        obj_gt_org          = []
        subdivision         = []
        
        for elems in batch:
            obj_pair_global_idx.append(elems['obj_pair_global_idx'])
            obj_pair_name.      append(elems['obj_pair_name'])
            general_info.       append(elems['general_info'])   
            scene_info.         append(elems['scene_info']) 
            map_info.           append(elems['map_info']) 
            obj_camera.         append(torch.from_numpy(elems['obj_camera']).to(torch.float).t())
            obj_lidar.          append(torch.from_numpy(elems['obj_lidar']).to(torch.float).t())
            obj_ego.            append(torch.from_numpy(elems['obj_ego']).to(torch.float).t())
            obj_gt.             append(elems['obj_gt'])
            obj_ego_org.        append(elems['obj_ego_org'])
            obj_gt_org.         append(elems['obj_gt_org'])
            subdivision.        append(elems['subdivision'])            
            # Dimensions of obj_camera, obj_lidar: torch.Size([N_time, N_features]) - per row the features of one timestamp are present
        
        obj_camera = torch.nn.utils.rnn.pad_sequence(obj_camera, batch_first=True)        
        obj_lidar  = torch.nn.utils.rnn.pad_sequence(obj_lidar,  batch_first=True)      
        obj_ego    = torch.nn.utils.rnn.pad_sequence(obj_ego,    batch_first=True)
        # Dimensions of obj_camera, obj_lidar: torch.Size([N_batch, N_time_max, N_features]) - padded objects with the maximum N_time
        
        out_dict = {'obj_pair_global_idx':  obj_pair_global_idx,
                    'obj_pair_name':        obj_pair_name,
                    'general_info':         general_info, 
                    'scene_info':           scene_info, 
                    'map_info':             map_info, 
                    'obj_camera':           obj_camera, 
                    'obj_lidar':            obj_lidar,
                    'obj_ego':              obj_ego, 
                    'obj_gt':               obj_gt,
                    'obj_ego_org':          obj_ego_org, 
                    'obj_gt_org':           obj_gt_org,
                    'subdivision':          subdivision}
        
        return  out_dict
        

    def get_iterator(self, epoch, gpu_idx):
        self.rand_seed1 = epoch

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
                                              load=self._load_function)

        sampler = torch.utils.data.distributed.DistributedSampler(
            tnt_dataset,
            num_replicas=self.num_gpus,
            shuffle=self.shuffle,
            rank=gpu_idx)
        sampler.set_epoch(epoch)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=self._collate_fun,
                                           num_workers=self.num_workers[0],
                                           sampler=sampler)
        return data_loader
    
    def __call__(self, epoch=0, rank=0):
        return self.get_iterator(epoch, rank)
    
    def __len__(self):
        return math.ceil((len(self.dataset) / self.batch_size) / self.num_gpus)