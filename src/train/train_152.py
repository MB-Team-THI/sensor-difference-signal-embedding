
import math
import wandb
import random
import logging
import numpy as np
from tqdm import tqdm
from os import device_encoding

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from src.train.train import Training
from src.utils.average_meter import AverageMeter


def prep_data(input_data, cuda):
    """
    Takes a batch of tuplets and converts them into Pytorch variables 
    and puts them on GPU if available.
    """
    input_data_out = dict((k, Variable(v)) for k,v in input_data.items())
    input_data = input_data_out
    
    if cuda:
        input_data_out = dict((k, v.cuda()) for k,v in input_data.items())
    return input_data_out

class train_152(Training):
    def __init__(self,
                 idx                      = 152,
                 crops_for_assignment     = None,
                 nmb_crops                = None,
                 temperature              = 0.1,
                 freeze_prototypes_niters = 313,
                 epsilon                  = 0.05,
                 queue                    = None,
                 sinkhorn_iterations      = 3,
                 eval_epochs              = 10,
                 save_embeddings          = ['test-set'],
                 enable_grad_clip         = False,
                 clip_value               = None,
                 dummy_example            = False,  
                 **kwargs) -> None:
        super().__init__()
        if nmb_crops is None:
            nmb_crops = [2]
        if crops_for_assignment is None:
            crops_for_assignment = [0, 1]
        self.description                = "Advanced-autoencoder phase-1 training"
        self.temperature                = temperature
        self.freeze_prototypes_niters   = freeze_prototypes_niters
        self.epsilon                    = epsilon
        self.sinkhorn_iterations        = sinkhorn_iterations
        self.queue                      = queue
        self.eval_epochs                = eval_epochs
        self.save_embeddings            = save_embeddings
        self.enable_grad_clip           = enable_grad_clip
        self.clip_value                 = clip_value
        self.dummy_example              = dummy_example
        self.dummy_rand_vals            = [random.random() for idx in range(32)]
        torch.autograd.set_detect_anomaly(True)


    def run_training(self, model, dataloader_train, loss_fc, optimizer, _, dataloader_test, eval_fc, dataset_param_test, dataset_param_train, run_name):
        self._train(model, dataloader_train, loss_fc, optimizer, dataloader_test, eval_fc, dataset_param_test, dataset_param_train, run_name)

    def _train(self, model, dataloader_train, loss_fc, optimizer, dataloader_test, eval_fc, dataset_param_test, dataset_param_train, run_name):
        wandb.watch(model, log='all', log_freq=10)
        
        epochs = dataloader_train.epochs
        pbar   = tqdm(total=int(epochs * len(dataloader_train.dataset) /
                                dataloader_train.batch_size),
                      desc="init training...".center(50))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)

        model.train()
        writer = SummaryWriter()

        for epoch in range(epochs):
            loss_record = AverageMeter()
            batch_pass = 0
            if epoch== 1:
                d=2

            dataloader = dataloader_train(epoch, rank=0)
            for batch_idx, input_data in enumerate(dataloader, start=epoch * len(dataloader)):      
                # writer.add_embedding(z_test, metadata=labels_test,global_step=batch_idx)

                # ============ Forward pass and Loss ============
                batch_traj_pairs = {'obj_camera':       input_data['obj_camera'], 
                                    'obj_lidar':        input_data['obj_lidar'],
                                    'obj_ego':          input_data['obj_ego'],
                                    'data_order_dict':  input_data['general_info'][0],}


                [z, x_rec]       = model(batch_traj_pairs)
                batch_pass       += 1
                loss             = loss_fc(x=batch_traj_pairs, x_pred=x_rec, z=z)

                # ============ Backward pass and optim step  ============
                optimizer.zero_grad()
                loss.backward()                
                assert not torch.isnan(loss).item(), "loss is NaN"
                assert not torch.stack([torch.isnan(p).any() for p in model.parameters()]).any().item(), "model parameters are NaN"

                if self.enable_grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)

                optimizer.step()

                # ============ Progress bar and summary writer ============
                pbar.update(1)
                loss_record.update(loss.item(), loss)
                log_msg = "Epoch:{:2}/{}  Iter:{:3}/{} Avg Loss: {:6.3f}".format(
                        epoch + 1, epochs, 
                        batch_pass, len(dataloader),
                        round(loss_record.avg.item(), 3)).center(50)
                pbar.set_description(log_msg)
                if batch_idx%10==0:
                    writer.add_scalar("Train Loss", loss, batch_idx)
                    writer.add_scalar("Train Loss - AVG", loss_record.avg, batch_idx)
                logging.info(log_msg)
                # wandb
                wandb.log({"Train Loss":        (loss / len(z))})
                # TODO keep this / len(z) for the avg loss?
                wandb.log({"Train Loss - AVG":  (loss_record.avg / len(z))})
                wandb.log({"Epoch": epoch})                            

                if loss_fc.enable_kmeans_loss and loss_fc.kmeans_loss_initialized and not loss_fc.learn_cluster_rep:
                    [z_updated, x_rec] = model(batch_traj_pairs)
                    loss_fc.update_cluster_rep(z_updated)
            

            if loss_fc.enable_kmeans_loss:
                # Init cluster centers
                if (epoch+1) >= loss_fc.kmeans_loss_pretrain_ep and not loss_fc.kmeans_loss_initialized:
                    # Obtain the embeddings from the whole train-set 
                    # OR save and append them during the casual training over the epochs
                    dummy_label_set = []
                    z_train_set = []
                    # get all z of the training set
                    for idx, input_data in enumerate(dataloader, start=epoch * len(dataloader)):                    
                        # ============ Forward pass and Loss ============
                        batch_traj_pairs = {'obj_camera':       input_data['obj_camera'], 
                                            'obj_lidar':        input_data['obj_lidar'],
                                            'obj_ego':          input_data['obj_ego'],
                                            'data_order_dict':  input_data['general_info'][0],}                 
                        [z, _] = model(batch_traj_pairs)
                        z_train_set.append(z.cpu().detach())         
                    z_train_set = torch.cat(z_train_set)

                    # Init kmeans cluster rep based on z_train_set
                    loss_fc.init_kmeans_loss(z_train_set)

                # Save embeddings and corresponding clustering results every epoch
                cluster_rep = loss_fc.cluster_rep_comp
            else:
                cluster_rep = []
             
            
            # eval_epochs = [50, 100, 150, 200, 300, 400, 500, 600, 700]
            #if epoch % self.eval_freq == 0 and epoch != 0:
            if epoch in self.eval_epochs :

                if "test-set" in self.save_embeddings:
                    # Perform evaluation for TEST set and save embeddings
                    # save_clustering_embeddings(model, dataset_param_test, run_name=run_name, epoch=epoch,  
                    #                            cluster_rep=cluster_rep,   eval_152=eval_fc,  dummy_rand_vals=self.dummy_rand_vals)
                    eval_fc(model, dataset_param_test, dataloader_test = dataloader_test, 
                            run_name=run_name, epoch=epoch, dummy_rand_vals=self.dummy_rand_vals)

                if "train-set" in self.save_embeddings:
                    # Perform evaluation for TRAIN set and save embeddings                    
                    eval_fc(model, dataset_param_train, dataloader_test = dataloader_train, 
                            run_name=run_name, epoch=epoch, dummy_rand_vals=self.dummy_rand_vals)
                    
                    
