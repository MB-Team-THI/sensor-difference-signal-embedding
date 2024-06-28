
import torch

import math
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans

from src.train.loss.loss import loss
from src.utils.difference_signal import calculate_error_signal_full_with_padding
from src.utils.difference_signal import calculate_error_signal_original
from src.utils.difference_signal import calculate_error_signal_pred
from src.utils.data_adaption import norm_error_signal
from src.utils.data_adaption import norm_error_signal_logarithmic



architecture_types          = ["Encoder", "AE", "AE_advanced_phase1", "AE_advanced_phase2", "AE_error_signal"]
decoderT_types              = ['LSTM-Decoder', 'CNN-Decoder', 'MLP']
loss_types                  = ['reconstruction_loss', 'VICReg', 'SiamSim']
processed_objects_types     = ['obj_pair', 'obj_lidar', 'obj_camera']
loss_target_recon_types     = ['obj_pair', 'obj_lidar', 'obj_camera', 'calc_error_signal', 'recon_error_signal']
error_signal_types          = ['euclidean', 'difference_for_each_dim']
error_signal_feature_types  = ['diff_x', 'diff_y', 'directed_distance', 'euclidean_distance', 'mean_distance_to_ego_x', 'mean_distance_to_ego_y']



def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class loss_152(loss):
    def __init__(self,
                idx                     = 152,
                architecture_type       = architecture_types[0],
                decoderT_type           = decoderT_types[0],
                processed_objects       = processed_objects_types[0],
                loss_type               = loss_types[0],
                loss_target             = loss_target_recon_types[0],
                error_signal_features   = error_signal_feature_types[0],
                enable_padding          = True,
                padding_value           = 15,
                data_adaption           = [],
                norm_max_values         = {},
                enable_kmeans_loss      = False,
                learn_cluster_rep       = False,
                kmeans_loss_pretrain_ep = 10,
                z_dim_m                 = 2,
                n_clusters              = -1,
                val_lambda              = 0.5,
                alpha                   = 1,
                h_init_range            = [-1.0, 1.0],
                name                    = 'Single Trajectory Reconstruction MSE Loss',
                description             = 'Advanced Autoencoder phase-1, the loss is applied on each reconstructed trajectory (lidar and camera) separately.',
                input_                  = 'Original and reconstructed trajectories of lidar and camera objects',
                output                  = 'loss-object'
                ) -> None:
        super().__init__(idx,name, description, input_,output)
  
        assert decoderT_type in decoderT_types, 'decoderT keyword unknown'
        assert processed_objects in processed_objects_types, 'processed_objects keyword unknown'
        assert loss_type in loss_types, 'loss_type keyword unknown'   
        for item in loss_target:
            assert item in loss_target_recon_types, 'loss_target_recon keyword unknown'

        self.architecture_type       = architecture_type
        self.decoderT_type           = decoderT_type
        self.processed_objects       = processed_objects
        self.loss_type               = loss_type
        self.loss_target_recon       = loss_target
        self.error_signal_features   = error_signal_features
        self.bn                      = nn.BatchNorm1d(2, affine=False)
        self.temperature             = 0.2
        self.verbose                 = False
        self.margin                  = 1.0
        self.enable_padding          = enable_padding
        self.padding_value           = padding_value
        self.data_adaption           = data_adaption        
        self.norm_max_values         = norm_max_values
        self.enable_kmeans_loss      = enable_kmeans_loss
        self.learn_cluster_rep       = learn_cluster_rep
        self.kmeans_loss_pretrain_ep = kmeans_loss_pretrain_ep
        self.z_dim_m                 = z_dim_m
        self.n_clusters              = n_clusters
        self.val_lambda              = val_lambda
        self.alpha                   = alpha
        self.pi_selection_vector     = np.zeros(self.n_clusters)
        self.kmeans_loss_initialized = False
        self.cluster_rep_learn       = torch.empty((self.n_clusters, self.z_dim_m), requires_grad=True)
        self.cluster_rep_comp        = torch.distributions.uniform.Uniform(-1.0, 1.0).sample([self.n_clusters, self.z_dim_m]) 
        self.batch_count             = 0

    
    def distance_measures(self, z_1, z_2, distance_type="neg_cosine_similarity"):
        distance = 0.0

        if distance_type == "neg_cosine_similarity":
            z_2 = z_2.detach()  # stop gradient

            z_1 = F.normalize(z_1, dim=1)  # l2-normalize
            z_2 = F.normalize(z_2, dim=1)  # l2-normalize

        else:
            raise ValueError("Unknown distance type", distance_type)

        return distance

    

    def init_kmeans_loss(self, z_set):
        # Init cluster centers based on k-means clustering on pretrained embeddings
        # Perform kmeans clustering on z_set
        kmeans_model = KMeans(n_clusters=self.n_clusters, init="k-means++").fit(z_set)
        self.cluster_rep_learn = torch.tensor(kmeans_model.cluster_centers_, requires_grad=True)
        self.cluster_rep_comp  = torch.tensor(kmeans_model.cluster_centers_, requires_grad=True)

        self.kmeans_loss_initialized = True


    def kmeans_loss_compute_r(self, z_batch):
        ### "Towards K-means-friendly Spaces: Simultaneous Deep Learning and Clustering", Yang 2017 
        #    similar to De Candido 2023

        ### Simple approach to perform k-means clustering loss
        # Distance to the cluster centers is added to the loss term
        # Cluster centers / representatives are just updated (not learned)
        cluster_rep     = self.cluster_rep_comp.to(z_batch.device)
        # 1. Compute distances 
        distances       = torch.cdist(z_batch, cluster_rep, p=2)

        # 2. Get min distance -> idx of next cluster_rep [n_batch x 1]
        cluster_assign  = torch.argmin(distances, dim=1)

        # 3. Sum up the differences of the samples to the selected representative
        batch_cluster_rep = []
        for j in cluster_assign:
            batch_cluster_rep.append(cluster_rep[j])
        stack_batch_clust_rep = torch.stack(batch_cluster_rep)

        # 4. Compute distance between samples and associated cluster_rep
        loss = torch.mean(torch.cdist(z_batch, stack_batch_clust_rep))

        # Perform the optimization step and afterward update cluster_rep
        return loss


    def kmeans_loss_learn_r(self, z_batch):
        ### "Deep K-Means", Fard 2020 

        ### Advanced approach to perform k-means clustering loss
        # Distance to the cluster centers is weighted and added to the loss term
        # Cluster centers / representatives are really learned (not just updated)
        
        if True:
            # Cluster based
            
            # 1. Compute distances per cluster (cluster-embeddings)
            distances = torch.cdist(self.cluster_rep_learn.to(z_batch.device).float(), z_batch.float(), p=2)
        
            # 2. Get min distance (= closest embedding to cluster)
            min_dist  = torch.min(distances, 1).values

            # 3. Compute exp shifts
            list_exp  = []
            for i in range(self.n_clusters):
                exp = torch.exp(-self.alpha * (distances[i] - min_dist[i]))
                list_exp.append(exp)
            stack_exp  = torch.stack(list_exp)
            sum_exp    = torch.sum(stack_exp, dim=1)

            # 4. Compute Softmax
            list_softmax        = []
            list_weighted_dist  = []
            for j in range(self.n_clusters):
                softmax         = stack_exp[j] / sum_exp[j]
                weighted_dist   = distances[j] * softmax
                list_softmax.append(softmax)
                list_weighted_dist.append(weighted_dist)
            stack_weighted_dist = torch.stack(list_weighted_dist)

            loss = torch.mean(torch.sum(stack_weighted_dist, dim=1))
        
        return loss
    

    def get_cluster_rep_learn(self):
        return self.cluster_rep_learn

    def update_cluster_rep(self, z_batch):
        cluster_rep = self.cluster_rep_comp.to(z_batch.device)

        # get the minimum distance for each z in z_batch to H
        distances       = torch.cdist(z_batch, cluster_rep, p=2)
        cluster_assign  = torch.argmin(distances, dim=1)

        # Update cluster representatives based on Fard2020
        count = torch.ones(self.n_clusters).to(z_batch.device) * 100
        for j in range(len(z_batch)):
            k = cluster_assign[j]  # cluster assignment k for sample j
            count[k] = count[k] + 1
            new_rep = (cluster_rep[k] - ((1 / count[k]) * (cluster_rep[k] - z_batch[j]))).detach()

            cluster_rep[k] = new_rep

        self.cluster_rep_comp = cluster_rep
    
    
    def MSE(self, x, x_pred):
        loss_function = nn.MSELoss(reduction="mean")
        return loss_function(x, x_pred)
    

    def recon_loss_image(self, x_img, x_img_pred, beta_traj=1.0, beta_background=6.0):

        l_background_red=0
        l_traj_red=0

        ### LiDAR-Loss ### --------------------------------------------------------------------------------------------------
        # Trajectory-loss of x_img: loss of non-zero pixels in x_img --------------------------------------------------------
        x_img_green = x_img[:,:,1]
        mask_traj = x_img_green > 0.0
        pred_masked_traj = torch.mul(x_img_pred[:,:,1], mask_traj) # all pixel that have a value in x_img
        # MSE loss on non-zero pixels (aka. trajectories)
        if mask_traj.sum() == 0:
            # prevent l_traj_green=nan
            l_traj_green = 0
        else:
            l_traj_green = ((x_img_green - pred_masked_traj) ** 2).sum()/mask_traj.sum()
        
        # Background-loss of x_img: loss of zero-pixels in x_img ------------------------------------------------------------
        mask_background = x_img_green == 0.0
        pred_masked_background = torch.mul(x_img_pred[:,:,1], mask_background) # all pixel that have NO value in x_img
        # MSE loss on zero pixels (aka. background)
        l_background_green = ((pred_masked_background) ** 2).sum()/mask_background.sum()


        # Weighted combination of both loss terms ------------------------------------------------------------------------
        loss = (beta_traj * l_traj_red   + beta_background * l_background_red +       
                beta_traj * l_traj_green + beta_background * l_background_green) #+beta_traj * l_traj_blue + beta_background * l_background_blue

        return loss
    

    def MSE_batch(self, batch_x, batch_x_pred):
        batch_x = batch_x.to(batch_x_pred.device)

        # Reduce x and x_pred to the original length to not falsely influence the loss by masked elements
        if batch_x.dim() == 2:
            sample_lengths = [torch.count_nonzero(x) for x in batch_x]
        elif batch_x.dim() == 3:
            sample_lengths = [int(torch.count_nonzero(x) / len(x[0])) for x in batch_x]

        loss = 0.0
        for x, x_pred, mask_length in zip(batch_x, batch_x_pred, sample_lengths):
            x_masked      = x[:mask_length]
            x_pred_masked = x_pred[:mask_length]
            # Apply the MSE loss
            loss += self.MSE(x_masked, x_pred_masked)
        return loss


    def cross_entropy(self, x, x_pred):

        loss = -1 * (x * math.log(x_pred) + (1 - x) * math.log(1-x))
        return loss


    def forward(self, x, x_pred, z=None, y=None):
        """
        Computes loss for each batch.
        """      
        loss = 0.0
        
        if self.architecture_type == architecture_types[2]:
            ### Loss for AE_advanced_phase1:
            if self.decoderT_type == decoderT_types[0]:
                #-----------------------------------------
                # Loss for sequence output of LSTM decoder
                #-----------------------------------------
                if self.loss_type == loss_types[0]:
                    ### Reconstruction Losses

                    if (loss_target_recon_types[0] in self.loss_target_recon or loss_target_recon_types[1] in self.loss_target_recon) and \
                       (self.processed_objects == processed_objects_types[0] or self.processed_objects == processed_objects_types[1]):
                        # Loss: reconstructed lidar-obj
                        loss += self.MSE_batch(batch_x      = x['obj_lidar'], 
                                               batch_x_pred = x_pred['decoded_lidar'])

                    if (loss_target_recon_types[0] in self.loss_target_recon or loss_target_recon_types[2] in self.loss_target_recon) and \
                       (self.processed_objects == processed_objects_types[0] or self.processed_objects == processed_objects_types[2]):
                        # Loss: reconstructed camera-obj
                        loss += self.MSE_batch(batch_x      = x['obj_camera'], 
                                               batch_x_pred = x_pred['decoded_camera'])

                    if loss_target_recon_types[3] in self.loss_target_recon and self.processed_objects == processed_objects_types[0]:
                        # Loss: error_signal(reconstructed_object_pair)
                        error_signal_dict, cohesion_table, _ = calculate_error_signal_original(batch_camera            = x['obj_camera'], 
                                                                                               batch_lidar             = x['obj_lidar'], 
                                                                                               included_features_types = self.error_signal_features)                       
    
                        error_signal_pred_dict = calculate_error_signal_pred(batch_camera            = x_pred['decoded_camera'], 
                                                                             batch_lidar             = x_pred['decoded_lidar'],
                                                                             batch_cohesion_table    = cohesion_table,
                                                                             included_features_types = self.error_signal_features)
                        # included_features_types
                        for key in error_signal_dict:
                            loss += self.MSE_batch(batch_x      = error_signal_dict[key], 
                                                   batch_x_pred = error_signal_pred_dict[key])

                    if loss_target_recon_types[4] in self.loss_target_recon:
                        # Loss: reconstructed error signal
                        error_signal_dict, _, _ = calculate_error_signal_original(batch_camera            = x['obj_camera'],
                                                                                  batch_lidar             = x['obj_lidar'],
                                                                                  included_features_types = self.error_signal_features)
                        # TODO convert this code part and put it in the function
                        error_feature_list = []
                        for key in error_signal_dict:
                            error_feature_list.append(error_signal_dict[key])
                        error_signal = torch.stack(error_feature_list).movedim(0, -1)

                        loss += self.MSE_batch(batch_x      = error_signal,
                                               batch_x_pred = x_pred['decoded_error_signal'])

                else:
                    raise ValueError('Other loss types still need to be defined.')


            elif self.decoderT_type == decoderT_types[2]:
                #-------------------------------------
                # Loss for "TBD" output of MLP decoder
                #-------------------------------------
                raise ValueError("The loss for the MLP decoder is undefined")


        elif self.architecture_type == architecture_types[4]:
            ### Loss for AE_error_signal:
            if self.decoderT_type == decoderT_types[0]:
                #-----------------------------------------
                # Loss for sequence output of LSTM decoder
                #-----------------------------------------
                if self.loss_type == loss_types[0]:
                    ### Reconstruction Loss
                    # Loss: reconstructed error signal
                    error_signal_batch, _, _ = calculate_error_signal_full_with_padding(batch_camera            = x['obj_camera'],
                                                                                        batch_lidar             = x['obj_lidar'],
                                                                                        batch_ego               = x['obj_ego'],
                                                                                        data_order_dict         = x['data_order_dict'],
                                                                                        included_features_types = self.error_signal_features,
                                                                                        enable_padding          = self.enable_padding,
                                                                                        padding_value           = self.padding_value)

                    if self.data_adaption == "normalization":
                        error_signal_batch_padded = norm_error_signal(error_signal_batch_padded = error_signal_batch, 
                                                                      error_signal_features     = self.error_signal_features, 
                                                                      norm_max_values           = self.norm_max_values)
                    elif self.data_adaption == "normalization_log":
                        error_signal_batch_padded = norm_error_signal_logarithmic(error_signal_batch_padded = error_signal_batch, 
                                                                                  error_signal_features     = self.error_signal_features, 
                                                                                  norm_max_values           = self.norm_max_values)

                    # Reconstruction loss
                    loss_recon = self.MSE_batch(batch_x      = error_signal_batch_padded,
                                                batch_x_pred = x_pred)
                    batch_size = len(x['obj_camera'])
                    wandb.log({"Train Reconstruction Loss":  (loss_recon  / batch_size)})     
                    
                    if self.enable_kmeans_loss and self.kmeans_loss_initialized:
                        # kmeans-Friendly-Loss
                        if self.learn_cluster_rep:
                            loss_kmeans = self.kmeans_loss_learn_r(z)
                        else:
                            loss_kmeans = self.kmeans_loss_compute_r(z)
                        wandb.log({"Train KMeans-friendly Loss": (loss_kmeans / batch_size)})
                        loss = loss_recon + self.val_lambda * loss_kmeans      
                    else:
                        loss = loss_recon



        assert not torch.isnan(loss).item(), "loss is NaN"
        return loss
    