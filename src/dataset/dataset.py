from gettext import find
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat
from glob import glob
from pathlib import Path
import os

from src.utils.load_lut import load_lut_for_index_reference
from src.utils.load_lut import load_lut_to_get_dataset_size

'''
Input array structure
ego/obj[0,:] = X
ego/obj[1,:] = Y
ego/obj[2,:] = TIMESTAMP
ego/obj[3,:] = psi (angle velocity)
ego/obj[4,:] = v
ego/obj[5,:] = a_lon
ego/obj[6,:] = a_lat
ego/obj[7,:] = time_idx_in_scenario_frame !!starting with 1!!
ego/obj[8,:] = circle_or_rectangle
'''

augmentation_types = ["no","connectivity","fieldofview","range"]
representation_types = ["vector","image_vector","image","image_bound","graph","trajectory","image_vector_merge_recon",
                        "multiple_channels", "obj_lists_scene", "object_pair_lut", "object_pair", "image_generation"]
rotation_types = ['ego', 'first_obj']

filtered_indices = []




class dataset(Dataset):
    from .loaders._object_pair import _load_sample_object_pair


    
    def __init__(self, name='argoverse', augmentation_type=[augmentation_types[0]], augmentation_meta=None, representation_type='image', orientation='plain',
                 mode='train', only_edges=False, bbox_meter=[200.0, 200.0], bbox_pixel=[100, 100], center_meter=None, only_overlaying=False,  
                 create_subdivisions=False, subdivisions_params=None, features_to_load=None, rotation_type='ego', hist_seq_first=0, hist_seq_last=49, pred_seq_first=None,
                 pred_seq_last=None, filename_overall_lut=None, dir_association_luts=None):
        super().__init__()
        self.name = name
        path = (Path(__file__).parent.parent.parent).joinpath('data').joinpath(self.name)
        self.dir = str(path)
        assert representation_type in representation_types, 'representation keyword unknown'
        assert rotation_type in rotation_types, 'rotation keyword unknown'
        self.representation_type  = representation_type
        self.features_to_load     = features_to_load
        self.only_overlaying      = only_overlaying
        self.create_subdivisions  = create_subdivisions
        self.subdivisions_params  = subdivisions_params
        self.rotation_type        = rotation_type
        self.orientation          = orientation
        self.mode                 = mode
        self.dir_association_luts = dir_association_luts
        self.filename_overall_lut = filename_overall_lut
        #TODO
        #assert augmentation_type in augmentation_types, 'augmentation keyword unknown'
        self.augmentation_type    = augmentation_type
        self.augmentation_meta    = augmentation_meta

        self.bbox_meter     = bbox_meter
        self.bbox_pixel     = bbox_pixel
        self.center         = center_meter
        self.hist_seq_first = hist_seq_first
        self.hist_seq_last  = hist_seq_last
        self.hist_seq_len   = self.hist_seq_last - self.hist_seq_first + 1
        
        self.pred_seq_first = pred_seq_first
        if not pred_seq_first:
            self.pred_seq_first = self.hist_seq_last +1 
        self.pred_seq_last  = pred_seq_last
        if not pred_seq_last:
            self.pred_seq_len = None
        else:
            self.pred_seq_len = self.pred_seq_last - self.pred_seq_first + 1

        self.only_edges  = only_edges
        self.min_y_meter = - self.center[1]
        self.min_x_meter = - self.center[0]
        self.max_y_meter = self.min_y_meter + self.bbox_meter[1]
        self.max_x_meter = self.min_x_meter + self.bbox_meter[0]
        self.rect_width  = 2
        self.rect_length = 4.5
        self.square_size = 1.5

        # Check if there is any train/test/val split
        result = [f.name for f in path.rglob("*")]
        if 'train' in result and 'test' in result:
            train_test_split_valid = True
        else:
            print('Please generate train/test split beforehand')
            exit()
        
        self.files = self.augmentation_init()

        if self.representation_type in ["image_generation", "object_pair_lut"]:
            self.len = load_lut_to_get_dataset_size(self.filename_overall_lut)-1
        else:
            self.len = len(self.files)
        self.data = [None] * self.len

        # Load the data into a list of dictionaries
        if self.representation_type in ['vector', 'image_bound']:
            # Load the maps--------------------------
            map_files = glob(self.dir + "\\map\\*.mat")
            self.map  = {}
            ways_bbox = {}
            for map_file in map_files:
                map_temp = loadmat(map_file)
                map_ID   = os.path.basename(map_file)[:-4]

                way_ids     = map_temp['way_id'][0]
                way_inside  = map_temp['way_bound_in'][0]
                way_outside = map_temp['way_bound_out'][0]

                self.map[map_ID]  = (way_ids, way_inside, way_outside, map_temp['way_bbox'])
                ways_bbox[map_ID] = map_temp['way_bbox']


    def __getitem__(self, index):

        if self.representation_type == "image":
            return self._load_sample_image(self.files[index])
        if self.representation_type == "image_vector":
            return self._load_sample_image_vector(self.files[index])
        if self.representation_type == "vector":
            return self._load_sample_vector(self.files[index])
        if self.representation_type == "image_bound":
            return self._load_sample_image_bound(self.files[index])
        if self.representation_type == "graph":
            return self._load_sample_graph(self.files[index])
        if self.representation_type == "trajectory":
            return self._load_sample_trajectory(self.files[index])
        if self.representation_type == "image_vector_merge_recon":
            return self._load_sample_image_vector_merge_recon(self.files[index])
        if self.representation_type == "multiple_channels":
            return self._load_sample_multiple_channels(self.files[index])
        if self.representation_type == "obj_lists_scene":
            return self._load_sample_obj_lists_scene(self.files[index])
        if self.representation_type == "object_pair_lut":
            scene_idx, pair_ID_in_scenario, scene_name = load_lut_for_index_reference(filename           = self.filename_overall_lut, 
                                                                                      overall_sample_idx = index)
            return self._load_sample_object_pair_with_lut(filename_sample       = self.files[scene_idx],
                                                          scene_name            = scene_name,
                                                          pair_ID_in_scenario   = pair_ID_in_scenario,
                                                          dir_association_luts  = self.dir_association_luts,
                                                          only_overlaying       = self.only_overlaying,
                                                          create_subdivisions   = self.create_subdivisions,
                                                          subdivisions_params   = self.subdivisions_params,
                                                          rotation_type         = self.rotation_type)
        if self.representation_type == "object_pair":
            return self._load_sample_object_pair(filename_sample     = self.files[index], 
                                                 features_to_load    = self.features_to_load,
                                                 only_overlaying     = self.only_overlaying,
                                                 create_subdivisions = self.create_subdivisions,
                                                 subdivisions_params = self.subdivisions_params,
                                                 rotation_type       = self.rotation_type)
        if self.representation_type == "image_generation":
            # Not used anymore -> main_image_generator.py
            scene_idx, pair_ID_in_scenario, scene_name = load_lut_for_index_reference(filename           = self.filename_overall_lut, 
                                                                                      overall_sample_idx = index)
            return self._image_generation(filename_sample       = self.files[scene_idx], 
                                          scene_name            = scene_name, 
                                          pair_ID_in_scenario   = pair_ID_in_scenario, 
                                          dir_association_luts  = self.dir_association_luts, 
                                          only_overlaying       = self.only_overlaying, 
                                          rotation_type         = self.rotation_type)
    
    def __len__(self):
        return self.len

    def __str__(self):
        return "Loaded {} dataset: {} total samples: {}".format(self.mode,
                                                                self.name,
                                                                len(self))

    def _get_description(self):
        description = {'name': self.name}
        return description
    
    def augmentation_init(self):
        if "connectivity" in self.augmentation_type:
            files = sorted(glob(self.dir + "/" + self.mode + "/augmentation/*.mat"))
        else:
            files = sorted(glob(self.dir + "/" + self.mode + "/base/*.mat"))
            assert files != [], "No files found in {}".format(self.dir + "/" + self.mode + "/base/*.mat")
        return files
    
    def get_scene_and_pair_id(self, overall_sample_idx):
        pass
