
import os
import json
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from datetime import date

from sklearn import metrics
from sklearn.cluster import KMeans
from pyod.models.abod import ABOD



normalize_methods  = ['min-max', 'np-normalize', 'no_normalization']

   

def load_json_file(filename):
    f = open(filename)
    loaded_json_file = json.load(f)
    f.close()

    return loaded_json_file


def check_dir_and_save_json(output_dir, filename, output_data):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename_dir = output_dir + '\\' + filename
    with open(filename_dir, 'w') as f:
        json.dump(output_data, f)


def plot_sc_scores(silhouette_scores, output_dir, data_split):
    # Convert data
    data_preproc = pd.DataFrame({
        'K Cluster':                            silhouette_scores['k_cluster'], 
        'Silhouette Score Mean':                silhouette_scores['sc_mean'],
        'Silhouette Score per Cluster Mean':    silhouette_scores['sc_mean_per_cluster']})
    
    # Plot data
    sns.set_theme()
    ax = sns.lineplot(x='K Cluster', y='value', hue='variable', 
                      data=pd.melt(data_preproc, ['K Cluster']))    
    ax.set(xlabel='K Cluster', ylabel='Silhouette Score')

    # Save plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = output_dir + 'silhouette_scores_' + data_split + '.png'
    plt.savefig(filename)
    plt.close()


def basic_cluster_eval(clustering_res, **kwargs):

    pred = clustering_res['pred']
    data = clustering_res['data']
    res_dict = {}

    # Number of data features
    res_dict['num_data_features'] = len(data[0])
    
    # Cluster distribution
    clusters, cluster_counts =  np.unique(pred, return_counts=True) 
    res_dict['cluster_distribution']  = {int(cluster_id): int(cnt) for cluster_id, cnt in zip(clusters, cluster_counts)}

    # N Cluster
    n_clusters_ = len(set(pred)) - (1 if -1 in pred else 0)
    res_dict['n_clusters'] = n_clusters_

    # Noise
    n_noise_ = list(pred).count(-1)
    res_dict['n_noise'] = n_noise_  

    # Centroids / Core samples
    if 'core_samples' in clustering_res:
        res_dict['core_sample_indices'] = clustering_res['core_samples'].tolist() 

    # Save algorithm specific parameters
    for key, item in kwargs.items():
        # Will this be valid for all cluster methods?
        res_dict[key] = {attribute: str(value) for attribute, value in item.__dict__.items()}

    return res_dict


def supervised_cluster_eval(res_dict):
    gt_clusters    = res_dict['gt_cluster']
    pred_clusters  = res_dict['pred_clusters']

    res_dict["Homogeneity"]                 = metrics.homogeneity_score(gt_clusters, pred_clusters)
    res_dict["Completeness"]                = metrics.completeness_score(gt_clusters, pred_clusters)
    res_dict["V-measure"]                   = metrics.v_measure_score(gt_clusters, pred_clusters)
    res_dict["Adjusted Rand Index"]         = metrics.adjusted_rand_score(gt_clusters, pred_clusters)
    res_dict["Adjusted Mutual Information"] = metrics.adjusted_mutual_info_score(gt_clusters, pred_clusters)

    return res_dict


def calinski_harabasz_index(pred_clusters, data):
    ### Calinski Harabasz Score aka. Variance Ratio Criterion --------------------------------------------------------------------------------
    # "The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster" - ScikitLearn
    if 1 < len(np.unique(pred_clusters)):
        ch_score = metrics.calinski_harabasz_score(data, pred_clusters)
        if math.isnan(ch_score): 
            ch_score['calinski_harabasz_index'] = None
    else:
        ch_score = None

    return ch_score


def davies_bouldin_index(pred_clusters, data):
    ### Davies-Bouldin Index ###
    # "A lower Davies-Bouldin index relates to a model with better separation between the clusters" - ScikitLearn
    if 1 < len(np.unique(pred_clusters)):
        db_score = metrics.davies_bouldin_score(data, pred_clusters)
        if math.isnan(db_score): db_score = None
    else:
        db_score = None

    return db_score


def silhouette_coefficient(pred_clusters, data, res_dict={}):
    ### Silhouette Coefficient - Intra-cluster distance - Inter-cluster distance ------------------------------------------------------------
    if 2 <= len(np.unique(pred_clusters)):
        # Attention: adaption of metrics.silhouette_score() is required to also return intra_clust_dists and inter_clust_dists
        silhouette_coefficient, intra_clust_dists, inter_clust_dists = metrics.silhouette_samples(data, pred_clusters)
        # result, intra_clust_dists, inter_clust_dists = silhouette_samples(X, labels, metric=metric, **kwds)

        unique_ids = np.unique(pred_clusters)
        silhouette_coefficient_per_cluster = {int(idx): 0 for idx in unique_ids}
        for cluster_id_1 in unique_ids:
            cluster_id_1_sum = []
            for idx, cluster_id_2 in enumerate(pred_clusters):
                if cluster_id_1 == cluster_id_2:
                    cluster_id_1_sum.append(silhouette_coefficient[idx])
            silhouette_coefficient_per_cluster[cluster_id_1] = float(np.mean(cluster_id_1_sum))

        res_dict['silhouette_coefficient_mean'] = float(np.mean(silhouette_coefficient))
        res_dict['silhouette_coefficient_per_cluster'] = silhouette_coefficient_per_cluster
        # Use Implementation from sklearn
        res_dict['intracluster_distance'] = float(np.mean(intra_clust_dists))
        res_dict['intercluster_distance'] = float(np.mean(inter_clust_dists))     
        
        if math.isnan(res_dict['silhouette_coefficient_mean']): res_dict['silhouette_coefficient_mean'] = None 
        if math.isnan(res_dict['intracluster_distance']):       res_dict['intracluster_distance'] = None       
        if math.isnan(res_dict['intercluster_distance']):       res_dict['intercluster_distance'] = None
    else:
        # No Silhouette Coefficient available for less than two clusters
        res_dict['silhouette_coefficient_mean'] = None
        res_dict['intracluster_distance']       = None
        res_dict['intercluster_distance']       = None


    return res_dict


def silhouette_analysis(pred_clusters, data, eval_res, output_dir):    
    ### Silhouette Analysis
    # based on https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    # Evaluation of silhouette analysis: a bad pick is characterized by:
    #               1. the presence of clusters with below average silhouette scores 
    #               2. a wide fluctuations in the size of the silhouette plots 

    n_cluster = len(np.unique(pred_clusters))
    if 2 <= n_cluster:
        silhouette_avg = eval_res['silhouette_coefficient_mean']
        # Create a subplot with 1 row and 2 columns
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 30)
        
        ax1.set_xlim([-0.1, 1])
        
        ax1.set_ylim([0, len(data) + (n_cluster + 1) * 10])
        
        sample_silhouette_values = metrics.silhouette_samples(data, pred_clusters)[0]
        max_silhouette_values_per_cluster = []

        y_lower = 10
        for i in range(n_cluster):
            #ith_cluster_silhouette_values = sample_silhouette_values[pred_clusters == i]
            ith_cluster_silhouette_values = np.array([x for idx, x in enumerate(sample_silhouette_values) if pred_clusters[idx] == i])
            ith_cluster_silhouette_values.sort()
            try:
                max_silhouette_values_per_cluster.append(max(ith_cluster_silhouette_values))
            except:
                max_silhouette_values_per_cluster.append(0.1)

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_cluster)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        max_delta = max(max_silhouette_values_per_cluster) - min(max_silhouette_values_per_cluster)

        deltas = []
        for item1 in max_silhouette_values_per_cluster:
            for item2 in max_silhouette_values_per_cluster:
                if item1 != item2:
                    deltas.append(abs(item1-item2))

        mean_delta = np.mean(deltas)
        title_str = ("The silhouette plot for the n_cluster = " + str(n_cluster) + 
                     "\n Average silhouette coefficient = " + "{0:.4f}".format(silhouette_avg) +
                     "\n Max. Delta = " + "{0:.4f}".format(max_delta) + 
                     "\n Average Delta = " + "{0:.4f}".format(mean_delta))
        ax1.set_title(title_str)
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        eval_res['silhouette_analysis_max_delta']  = max_delta
        eval_res['silhouette_analysis_mean_delta'] = mean_delta
        
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # plt.show()
        output_dir = output_dir + '\\silhouette_analysis\\'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = output_dir + 'n' + str(n_cluster) + '_cluster_silhouette_analysis.png'
        fig.savefig(filename)
        plt.close()

    else:
        eval_res['silhouette_analysis_max_delta']  = 100
        eval_res['silhouette_analysis_mean_delta'] = 100
    

    return eval_res


def evaluate_clustering(clustering_res, settings):

    pred_clusters = clustering_res['pred']
    data          = clustering_res['data']
    output_dir    = settings['output_dir']
    assert len(data) == len(pred_clusters), "They should have the same length"

    eval_res = {}    
    eval_res['basic_eval']              = basic_cluster_eval(clustering_res)

    ### Unsupervised Clustering Metrics ###
    eval_res['calinski_harabasz_index'] = calinski_harabasz_index(pred_clusters, data)
    eval_res['davies_bouldin_index']    = davies_bouldin_index(pred_clusters, data)
    eval_res                            = silhouette_coefficient(pred_clusters, data, eval_res)
    eval_res                            = silhouette_analysis(pred_clusters, data, eval_res, output_dir)
    

    return eval_res


def abod_method(cluster_id, X_cluster_train, X_cluster_test, cluster_idx_test, anomaly_labels_abod, anomaly_scores_abod):
    # Train data to fit model
    abod_model = ABOD(contamination = 0.03, 
                      n_neighbors   = 10,
                      method        = 'fast')
    ### Fit anomaly detection model on X_cluster_train
    abod_model.fit(np.array(X_cluster_train))

    abod_y_pred_train = abod_model.predict(np.array(X_cluster_train))
    # Test data to obtain predictions from model
    abod_y_pred_test  = abod_model.predict(np.array(X_cluster_test))

    abod_scores_test  = abod_model.decision_function(np.array(X_cluster_test))


    # Save results of the model and the prediction of the test-data       
    uniques_train, counts_train = np.unique(abod_y_pred_train, return_counts=True)
    uniques_test,  counts_test  = np.unique(abod_y_pred_test, return_counts=True)     
    res_abod = {'description':          'model fitted on train data, predictions made on test data',
                        'score':                'decision_scores_',
                        'decision_scores_':     abod_model.decision_scores_.tolist(),
                        'decision_scores_mean': np.mean(abod_model.decision_scores_),
                        'y_pred_train':         abod_y_pred_train.tolist(),
                        'y_distr_train':        {str(val): int(cnt) for val, cnt in zip(uniques_train.tolist(), counts_train.tolist())},
                        'n_outliers_train':     abod_y_pred_train.tolist().count(1),
                        'y_pred_test':          abod_y_pred_test.tolist(),
                        'y_distr_test':         {str(val): int(cnt) for val, cnt in zip(uniques_test.tolist(), counts_test.tolist())},
                        'n_outliers_test':      abod_y_pred_test.tolist().count(1),
                        'threshold_':           abod_model.threshold_,
                        'method':               abod_model.method,
                        'legend':               'decision_scores_: The higher, the more abnormal. Outliers tend to have higher scores.   \
                                                 y_pred: 0 stands for inliers and 1 for outliers/anomalies.'
                        }

    # Bring the predictions back in the original order of the val-set
    cluster_key = 'cluster_' + str(cluster_id)

    for idx in range(len(cluster_idx_test)):
        global_idx = cluster_idx_test[idx]

        ### Anomaly labels / prediction
        if abod_y_pred_test[idx] == 0:
            # Inlier
            anomaly_labels_abod[cluster_key][global_idx] = 1
        elif abod_y_pred_test[idx] == 1:
            # Outlier
            anomaly_labels_abod[cluster_key][global_idx] = 2
        else:
            # Other cluster = 0
            assert False, "All values in abod_y_pred_test should be 0 or 1"

        ### Anomaly Scores
        anomaly_scores_abod[global_idx] = abod_scores_test[idx]

    assert len(abod_y_pred_test) == len(np.nonzero(anomaly_labels_abod[cluster_key])[0])

    return res_abod, anomaly_labels_abod, anomaly_scores_abod


def get_n_extreme_values(order, values, n, small_scores_are_inlying):
    assert len(order) == len(values), "Length must be equal"
    # Create a list of tuple in order to associate the scores to an ID/name
    tuples = [(order[idx], values[idx]) for idx in range(len(order))]
    tuples.sort(key=lambda a: a[1])   # from small to big
    
    if small_scores_are_inlying:
        # Small scores = inlying, big scores = outlying
        n_inlying  = tuples[:n]       # n min scores
        n_outlying = tuples[-n:]      # n max scores
    else:
        # Small scores = outlying, big scores = inlying
        n_inlying  = tuples[-n:]     # n max scores 
        n_outlying = tuples[:n]      # n min scores


    return {'n_inlying':  n_inlying, 
            'n_outlying': n_outlying}


def main():
    # -------------------------------------------------------------------------------------------------
    # SETTINGS ========================================================================================
    # -------------------------------------------------------------------------------------------------
    base_dir          = ''
    folder_in         = 'output_files\\models\\ ' 
    folder_out        = 'clustering\\'

    base_dir          = 'C:\\Users\\fertig\\Documents\\FILES\\data\\datasets\\nuscenes\\'
    folder_in         = '20240122_train_set_diff_x_y\\' 
    folder_out        = '9_clustering_results\\'
    str_today         = str(date.today()).replace('-','')

    # TODO adapt MODEL_NAME
    train_folder  = ['\\MODEL_NAME\\epoch_100\\train\\'] 
    test_folder   = ['\\MODEL_NAME\\epoch_100\\test\\'] 

    folder_input_dir_train = base_dir + folder_in + train_folder[0]
    folder_input_dir_test  = base_dir + folder_in + test_folder[0]
   

    ### Load processed embeddings    
    filename_to_load_train  = folder_input_dir_train + 'embeddings_original_dim.json'
    filename_to_load_test   = folder_input_dir_test  + 'embeddings_original_dim.json'
    loaded_embeddings_train = load_json_file(filename_to_load_train)
    loaded_embeddings_test  = load_json_file(filename_to_load_test)
    X_train                 = loaded_embeddings_train['points']
    X_test                  = loaded_embeddings_test['points']
    X_train_order           = [x['obj_pair_name'] for x in loaded_embeddings_train['order']]
    X_test_order            = [x['obj_pair_name'] for x in loaded_embeddings_test['order']]


    min_n_clusters = 22
    max_n_clusters_plus_one = 23
    cluster_algorithms = ['kmeans_clustering']
    experiment_series  = [{'cluster_method': cluster_algorithms[0], 'n': idx} for idx in range(min_n_clusters, max_n_clusters_plus_one)]

    silhouette_scores_train = {'k_cluster':   [],
                               'sc_mean' : [],
                               'sc_mean_per_cluster': []}
    silhouette_scores_test = {'k_cluster':   [],
                               'sc_mean' : [],
                               'sc_mean_per_cluster': []}

    experiment_name = (str_today + '_' + experiment_series[0]['cluster_method'])
    output_dir      = base_dir + folder_out + experiment_name + '\\'


    for idx, current_experiment in enumerate(experiment_series):
        print('Current experiment '+ str(idx+1)+'/'+str(len(experiment_series)+1))

        ##################################
        ### Perform K-means Clustering ###
        ##################################
        # Create model
        n_clusters      = current_experiment['n']        
        cluster_model   = KMeans(n_clusters=n_clusters, n_init='auto').fit(X_train)
        
        # TRAIN Data --------------------------------------------------------------------------------------
        cluster_pred_train = cluster_model.predict(X_train)
        assert len(X_train) == len(cluster_pred_train), "They should have the same length"
        clustering_res_train = {'data':                  list(X_train),
                                'pred':                  cluster_pred_train.tolist(),
                                'inertia_':              cluster_model.inertia_,
                                'inertia_desc':          "Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.",
                                'cluster_centers_':      cluster_model.cluster_centers_.tolist(),
                                'cluster_centers_desc':  "Coordinates of cluster centers",
                                'n_features_in_':        cluster_model.n_features_in_}

        # TEST-data ------------------------------------------------------------------------------------
        cluster_pred_test = cluster_model.predict(X_test)
        assert len(X_test) == len(cluster_pred_test), "They should have the same length"
        clustering_res_test  = {'data':                  list(X_test),
                                'pred':                  cluster_pred_test.tolist(),
                                'inertia_':              cluster_model.inertia_,
                                'inertia_desc':          "Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.",
                                'cluster_centers_':      cluster_model.cluster_centers_.tolist(),
                                'cluster_centers_desc':  "Coordinates of cluster centers",
                                'n_features_in_':        cluster_model.n_features_in_}

        # Cluster Evaluation --------------------------------------------------------------------------
        eval_settings           = {'output_dir': output_dir}
        res_cluster_eval_train  = evaluate_clustering(clustering_res = clustering_res_train, 
                                                      settings       = eval_settings)
        res_cluster_eval_test   = evaluate_clustering(clustering_res = clustering_res_test, 
                                                      settings       = eval_settings)
        
        silhouette_scores_train['k_cluster']          .append(current_experiment['n'])
        silhouette_scores_train['sc_mean']            .append(res_cluster_eval_train['silhouette_coefficient_mean'])
        silhouette_scores_train['sc_mean_per_cluster'].append(np.mean(list(res_cluster_eval_train['silhouette_coefficient_per_cluster'].values())))
        
        silhouette_scores_test['k_cluster']          .append(current_experiment['n'])
        silhouette_scores_test['sc_mean']            .append(res_cluster_eval_test['silhouette_coefficient_mean'])
        silhouette_scores_test['sc_mean_per_cluster'].append(np.mean(list(res_cluster_eval_test['silhouette_coefficient_per_cluster'].values())))
        

        ##################################
        ### Cluster - Examination ###
        ##################################
        ad_methods = ['abod']
        cluster_ids_train, num_counts_train = np.unique(clustering_res_train['pred'], return_counts=True)
        cluster_ids_test,  num_counts_test  = np.unique(clustering_res_test['pred'], return_counts=True)

        anomaly_labels_abod     = {('cluster_'+str(id)): np.zeros(len(clustering_res_test['pred'])) for id in cluster_ids_test}
        anomaly_scores_abod     = np.zeros(len(clustering_res_test['pred']))


        res_anomaly_scores_per_cluster = {}
        for cluster_id, num_count_train in zip(cluster_ids_train, num_counts_train):            
            print('K=' + str(n_clusters) + '  Cluster: ' + str(cluster_id+1) + '/' + str(len(cluster_ids_train)))
            cluster_key = 'cluster_' + str(cluster_id)
            res_anomaly_scores_per_cluster[cluster_key] = {}

            if cluster_id in clustering_res_test['pred']:
                # Only perform if the cluster also appears in the test dataset

                ### Evaluation per Cluster -----------------------------------------------------------------------------
                # Get Cluster Samples of Train Set
                cluster_idx_train = np.where(np.array(clustering_res_train['pred']) == cluster_id)[0]
                assert len(cluster_idx_train) == num_count_train, "Should be of same length"
                X_cluster_train = [X_train[i] for i in cluster_idx_train]
                # Get Cluster Samples of Test Set
                cluster_idx_test = np.where(np.array(clustering_res_test['pred']) == cluster_id)[0]
                assert len(cluster_idx_test) == num_counts_test[ list(cluster_ids_test).index(cluster_id) ]
                X_cluster_test = [X_test[i] for i in cluster_idx_test]

                # ABOD 
                if 'abod' in ad_methods:
                    res_abod, anomaly_labels_abod, anomaly_scores_abod = abod_method(cluster_id                     = cluster_id, 
                                                                                    X_cluster_train                = X_cluster_train, 
                                                                                    X_cluster_test                 = X_cluster_test, 
                                                                                    cluster_idx_test               = cluster_idx_test, 
                                                                                    anomaly_labels_abod            = anomaly_labels_abod,
                                                                                    anomaly_scores_abod            = anomaly_scores_abod)
                    res_anomaly_scores_per_cluster[cluster_key]['abod'] = res_abod

            else:                
                if 'abod' in ad_methods:
                    res_anomaly_scores_per_cluster[cluster_key]['abod'] = {'y_pred_train': [-1],
                                                                           'y_pred_test':  [-1]}

        ### Postprocess anomaly scores
        if 'abod' in ad_methods:
            # ABOD
            assert len(X_test) == len(np.nonzero(anomaly_scores_abod)[0]), "No element in anomaly_scores_abod should be empty"
            order = [i for i in range(len(anomaly_scores_abod))]
            label_abod_extreme_points_test = get_n_extreme_values(order, anomaly_scores_abod, n=30, small_scores_are_inlying=True)


        # Get overall num outlier
        outliers_per_cluster_total = {}
        for method in ad_methods:
            outliers_per_cluster = {'sum': {'train_set': {'inlying':   sum([res_anomaly_scores_per_cluster[key][method]['y_pred_train'].count(0) for key in res_anomaly_scores_per_cluster]),
                                                          'outlying':  sum([res_anomaly_scores_per_cluster[key][method]['y_pred_train'].count(1) for key in res_anomaly_scores_per_cluster]),
                                                          'n_samples': sum([len(res_anomaly_scores_per_cluster[key][method]['y_pred_train']) for key in res_anomaly_scores_per_cluster])},
                                            'test_set':  {'inlying':   sum([res_anomaly_scores_per_cluster[key][method]['y_pred_test'].count(0) for key in res_anomaly_scores_per_cluster]),
                                                          'outlying':  sum([res_anomaly_scores_per_cluster[key][method]['y_pred_test'].count(1) for key in res_anomaly_scores_per_cluster]),
                                                          'n_samples': sum([len(res_anomaly_scores_per_cluster[key][method]['y_pred_test']) for key in res_anomaly_scores_per_cluster])},
                                            }}
            for key in res_anomaly_scores_per_cluster:
                    outliers_per_cluster[key] = {'train_set': {'inlying':  res_anomaly_scores_per_cluster[key][method]['y_pred_train'].count(0), 
                                                               'outlying': res_anomaly_scores_per_cluster[key][method]['y_pred_train'].count(1)},
                                                 'test_set':  {'inlying':  res_anomaly_scores_per_cluster[key][method]['y_pred_test'].count(0),
                                                               'outlying': res_anomaly_scores_per_cluster[key][method]['y_pred_test'].count(1),}}
            outliers_per_cluster_total[method] = outliers_per_cluster
        
        ### Save Results ###
        res_experiment = {'date':                   str_today,
                          'clustering_settings':    current_experiment,
                          'eval_settings':          eval_settings,
                          'clustering_train':       {'filename_to_load':       filename_to_load_train,
                                                     'clustering_results':     clustering_res_train,
                                                     'eval_results':           res_cluster_eval_train},
                          'clustering_test':        {'filename_to_load':       filename_to_load_test,
                                                     'clustering_results':     clustering_res_test,
                                                     'eval_results':           res_cluster_eval_test},
                          'anomaly_scores':         {'res':                    res_anomaly_scores_per_cluster,
                                                     'anomaly_numbers':        outliers_per_cluster_total}      
        }

        filename = 'clustering_res_n' + str(n_clusters) + '.json'
        check_dir_and_save_json(output_dir  = output_dir,
                                filename    = filename,
                                output_data = res_experiment)

        check_dir_and_save_json(output_dir  = output_dir,
                                filename    = 'record_file_order_train_set_clustering.json',
                                output_data = X_train_order)
        
        check_dir_and_save_json(output_dir  = output_dir,
                                filename    = 'record_file_order_test_set_clustering.json',
                                output_data = X_test_order)   


    # Silhouette Coefficient 
    plot_sc_scores(silhouette_scores_train, output_dir, data_split="train")
    plot_sc_scores(silhouette_scores_test,  output_dir, data_split="test")




if __name__ == '__main__':
    main()