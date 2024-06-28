from scipy.io import loadmat

def load_lut_for_index_reference(filename, overall_sample_idx):
    # This function loads a .mat-file, which functions as look-up-table (LUT) for the index-reference of the samples.
    # The overall_sample_idx is the index of the sampled association,
    # the corresponding scene_name, scene_idx and index in the scene are returned.
    mat_temp = loadmat(filename,
                       variable_names=["idx_overall", "scene_name",
                                       "idx_per_scene", "scene_idx"],
                       verify_compressed_data_integrity=False)

    target_idx          = list(mat_temp['idx_overall'][0]).index(overall_sample_idx)
    scene_idx           = mat_temp['scene_idx'][0][target_idx]
    pair_ID_in_scenario = mat_temp['idx_per_scene'][0][target_idx]
    scene_name          = mat_temp['scene_name'][target_idx]

    return scene_idx, pair_ID_in_scenario, scene_name


def load_lut_to_get_dataset_size(filename):
    # This function loads a .mat-file and returns the number of entries.
    mat_temp = loadmat(filename,
                       variable_names=["idx_overall"],
                       verify_compressed_data_integrity=False)
                       
    return len(mat_temp['idx_overall'][0])
