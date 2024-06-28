

import os
import json
import numpy as np

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def tsne_method(data, perplexity_val=3, output_dim=2):
    data_np = np.array(data)
    tsne_mapper = TSNE(n_components     = output_dim, 
                       learning_rate    = 'auto',
                       init             = 'random', 
                       perplexity       = perplexity_val)
    try:
        data_reduced = tsne_mapper.fit_transform(data_np)
    except:
        print("Execution of TSNE with metric does not work")
        data_reduced = []

    return data_reduced


def pca_method(data, output_dim=2):
    pca_mapper = PCA(n_components=output_dim)

    try:
        pca_mapper.fit(data)
        data_reduced = pca_mapper.transform(data)
    except:
        print("Execution of PCA is not possible")
        data_reduced = []

    return data_reduced


def umap_method(data, n_neighbors=15, min_dist=0.1, metric="euclidean", output_dim=2):

    umap_mapper = UMAP(n_neighbors  = n_neighbors,
                       min_dist     = min_dist,
                       metric       = metric,  # 'correlation' 'cosine', 'euclidean'
                       n_components = output_dim)
    try:
        data_reduced = umap_mapper.fit_transform(data)
    except:
        print("Execution of UMAP with metric " + metric + " does not work")
        data_reduced = []

    return data_reduced


def dim_red_selection(embeddings_original, output_dim, experiment):
        dim_red_method = experiment['dim_red_method']

        # UMAP
        if dim_red_method == 'umap':
            
            data_reduced = umap_method(data         = embeddings_original,
                                        n_neighbors  = experiment['n_neighbors'],
                                        min_dist     = experiment['min_dist'],
                                        metric       = experiment['metric'],
                                        output_dim   = output_dim)

            description_new = 'embeddings_UMAP_' + experiment['metric']
                    # str(experiment['n_neighbors']) + \
                    # '_dist=' + str(experiment['min_dist'])
            dim_reduction_params = {'dim_reduction_method': dim_red_method,
                                    'method_params_umap':   experiment['n_neighbors'],
                                    'min_dist':             experiment['min_dist'],
                                    'n_components':         output_dim}
            
        # TSNE    
        elif dim_red_method == 'tsne':
             
            data_reduced = tsne_method(data             = embeddings_original,
                                       perplexity_val   = experiment['perplexity'],
                                       output_dim       = output_dim)
            
            description_new = 'embeddings_TSNE_perpl' + str(experiment['perplexity'])
            dim_reduction_params = {'dim_reduction_method': dim_red_method,
                                    'perplexity':           experiment['perplexity'],
                                    'n_components':         output_dim}
        # PCA
        elif dim_red_method == 'pca':
            
            data_reduced = pca_method(data          = embeddings_original,
                                       output_dim   = output_dim)

            description_new = 'embeddings_PCA'
            dim_reduction_params = {'dim_reduction_method': dim_red_method,
                                    'n_components':         output_dim}

        # No dim reduction - saving of original embeddings
        elif dim_red_method == 'no_dim_red':
            
            data_reduced = np.array(embeddings_original)
            description_new = 'embeddings_original_dim' + str(len(embeddings_original[0]))
            dim_reduction_params = {'dim_reduction_method':          'not required',
                                    'number of original dimensions': len(embeddings_original[0])}

        else:
            # not implemented 
            assert False, 'TBD'

        return data_reduced, description_new, dim_reduction_params


def dim_reduction(embeddings_original, output_dir, output_dim=2, experiment_series=[]):

    if experiment_series == []:
        experiment_series = [
            {'dim_red_method': 'no_dim_red'},
            {'dim_red_method': 'pca'},
            {'dim_red_method': 'tsne',    'perplexity': 5},
            # {'dim_red_method': 'umap',    'n_neighbors': 15,    'min_dist': 0.0005,  'metric': 'correlation'},
            # {'dim_red_method': 'umap',    'n_neighbors': 15,    'min_dist': 0.0005,  'metric': 'cosine'},
            {'dim_red_method': 'umap',    'n_neighbors': 15,    'min_dist': 0.0005,  'metric': 'euclidean'},
            # {'dim_red_method': 'umap',    'n_neighbors': 15,    'min_dist': 0.0005,  'metric': 'manhattan'},
            # {'dim_red_method': 'umap',    'n_neighbors': 15,    'min_dist': 0.0005,  'metric': 'chebyshev'},
            # {'dim_red_method': 'umap',    'n_neighbors': 15,    'min_dist': 0.0005,  'metric': 'minkowski'},
            # {'dim_red_method': 'umap',    'n_neighbors': 15,    'min_dist': 0.0005,  'metric': 'canberra'},
            # {'dim_red_method': 'umap',    'n_neighbors': 15,    'min_dist': 0.0005,  'metric': 'braycurtis'},
            # {'dim_red_method': 'umap',    'n_neighbors': 15,    'min_dist': 0.0005,  'metric': 'mahalanobis'},
            # {'dim_red_method': 'umap',    'n_neighbors': 15,    'min_dist': 0.0005,  'metric': 'wminkowski'},
            # {'dim_red_method': 'umap',    'n_neighbors': 15,    'min_dist': 0.0005,  'metric': 'seuclidean'},    
        ]
    for idx, experiment in enumerate(experiment_series):

        data_reduced, desc_new, dim_red_params = dim_red_selection(embeddings_original = embeddings_original,
                                                                   output_dim          = output_dim,
                                                                   experiment          = experiment)
        if data_reduced == []:
            # Dimensionality reduction failed
            continue
        

        ### Save embeddings with settings-string
        dict_out = {'name':                  desc_new,
                    'points':                data_reduced.tolist(),
                    'dim_reduction_params':  dim_red_params}

        filename = output_dir + 'embeddings_' + str(idx) + '.json'
        with open(filename, 'w') as f:
            json.dump(dict_out, f)





        
