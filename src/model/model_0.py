from src.model.model import model 
import cv2
import math
import torch
import einops
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import torchvision.models as models
from x_transformers import ContinuousTransformerWrapper, Encoder

from src.utils.create_image_representation import trajectory_to_image
from src.train.loss.loss_0 import calculate_error_signal_original
from src.train.loss.loss_0 import calculate_error_signal_full_with_padding
from src.utils.data_adaption import norm_error_signal
from src.utils.data_adaption import norm_error_signal_logarithmic

architecture_types      = ["Encoder", "AE", "AE_advanced_phase1", "AE_advanced_phase2", "AE_error_signal"]
processed_objects_types = ['obj_pair', 'obj_lidar', 'obj_camera']
encoderI_types          = ["ViT", "ResNet-18"]
encoderT_types          = ["LSTM-Encoder", "Transformer-Encoder", "CNN-Encoder"]
merge_types             = ["FC"]
projector_types         = ["MLP"]
decoderT_types          = ['LSTM-Decoder', 'CNN-Decoder', 'MLP']



class distributer(nn.Module):
    def __init__(self, n_features, z_dim_m, encoderT_args, merge_args, dim_in = 1):
        super().__init__()
        self.n_features             = n_features
        self.dim_in                 = dim_in
        self.z_dim_m                = z_dim_m
        self.dim_out_feat_enc       = z_dim_m
        self.dim_encoder_mlp_head   = z_dim_m * 2
        self.encoderT_args          = encoderT_args
        self.merge_args             = merge_args

        # Feature Encoders
        self.feature_encoders       = nn.ModuleList([])
        for idx in range(self.n_features):
            self.feature_encoders.append(traj_encoder(dim_in     = self.dim_in, 
                                                      dim_out    = self.dim_out_feat_enc,
                                                      depth      = self.encoderT_args['depth'], 
                                                      heads      = self.encoderT_args['heads'],
                                                      dim_trans  = self.encoderT_args['dim_trans'], 
                                                      dim_mlp    = self.encoderT_args['dim_mlp'],
                                                      dim_emb_tokes = 2))
            
        # Merger MLP
        z_dim_m_in = self.dim_out_feat_enc * self.n_features
        dims = [[z_dim_m_in, z_dim_m_in]]
        if not self.merge_args:
            n_layers = 2
            for i in range(n_layers-2):
                dims.append([z_dim_m_in, z_dim_m_in])
            dims.append([z_dim_m_in, z_dim_m])
            self.encoder_mlp_head = FC(n_layers=n_layers, dims=dims)
        else:
            n_layers = self.merge_args['n_layers']
            for i in range(n_layers-2):
                dims.append([z_dim_m_in, z_dim_m_in])
            dims.append([z_dim_m_in, z_dim_m])
            self.encoder_mlp_head = FC(dims=dims,**self.merge_args) 

    
    def forward(self, x, x_lengths):
        # Split up the input and distribute the slices to the single encoders
        z_features = []
        for idx in range(self.n_features):
            x_feature = x[:, :, idx]
            x_feature_packed = torch.nn.utils.rnn.pack_padded_sequence(x_feature,
                                                                       x_lengths,
                                                                       batch_first=True,
                                                                       enforce_sorted=False)

            out_feat_encoder = self.feature_encoders[idx].forward(x_feature_packed)
            z_features.append(out_feat_encoder)

        # MLP to set it back together
        z_before_mlp = torch.cat((z_features), 1)
        z = self.encoder_mlp_head(z_before_mlp)
        return z



class traj_encoder(nn.Module):
    def __init__(self, *, dim_in, dim_out, depth, heads, dim_trans, dim_mlp, dim_emb_tokes=3):
        super().__init__()
        self.emb_token = nn.Parameter(torch.randn(1, dim_in))
        self.dim_emb_tokes = dim_emb_tokes

        self.model = ContinuousTransformerWrapper(
            dim_in      = dim_in,
            max_seq_len = 70,
            attn_layers = Encoder(dim   = dim_trans,
                                  depth = depth,
                                  heads = heads)
        )
        self.mlp_head = nn.Sequential(nn.Linear(dim_trans, dim_mlp),
                                      nn.ReLU(),
                                      nn.Linear(dim_mlp, dim_out))


    def forward(self, x, x_unpacked=None, x_lengths=None):
        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # target shape ob emb_tokens: [B, C, *], with B = batch size, C = number of channels, and * is not relevant for torch.cat()
        emb_tokens = self.emb_token.expand(seq_unpacked.shape[0], -1, -1)
        if self.dim_emb_tokes == 2:
            seq_unpacked = seq_unpacked.unsqueeze(2)

        emb_tokens = emb_tokens.to(seq_unpacked.device)
        # emb_tokens = einops.rearrange(emb_tokens, 'B T C -> B C T')
        seq_unpacked = torch.cat((emb_tokens, seq_unpacked), dim=1)
        mask = (torch.arange(seq_unpacked.shape[1])[None, :] < lens_unpacked[:, None]+1).to(seq_unpacked.get_device())
        self.model.to(mask.device)
        self.mlp_head.to(mask.device)
        z_intermediate = self.model.forward(seq_unpacked, mask=mask)
        return self.mlp_head(z_intermediate[:,0,:])



class lstm_decoder(nn.Module):    
    def __init__(self, input_size=None, hidden_size=None, output_size=3, num_layers=1): 
        super(lstm_decoder, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.vor_linear  = nn.Linear(hidden_size, hidden_size)
        
        self.lstm = nn.LSTM(input_size  = self.input_size, 
                            hidden_size = self.hidden_size,
                            num_layers  = self.num_layers, 
                            batch_first = True)
        self.nach_linear = nn.Linear(hidden_size, output_size)           


    def forward(self, x_input, encoder_hidden_states, t_steps):
        if t_steps==0:
            encoder_hidden_states = (encoder_hidden_states[0], encoder_hidden_states[0])
        output, self.hidden = self.lstm(x_input, encoder_hidden_states)

        output = self.nach_linear(output)        
        return output, self.hidden



class FC(nn.Module):
    def __init__(self, n_layers, dims):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(n_layers):
            self.layers.add_module("Lin_"+str(i), nn.Linear(dims[i][0], dims[i][1]))
            if i == n_layers-1:
                break
            # TODO append BN layer?? -> flag for BN?
            # self.layers.append(nn.BatchNorm1d(sizes[i + 1]))
            
            self.layers.add_module("Activation_"+str(i), nn.ReLU())


    def forward(self, x_in):    
        x_out = self.layers(x_in)
        return x_out
    


class DecoderFull(nn.Module):
    def __init__(self, *, architecture_args=None, decoderT_type=decoderT_types[0], z_dim_m=32, decoderT_args=None, lstm_input_size=3):
        super().__init__()
        
        if architecture_args == None:
            architecture_args={'processed_objects': processed_objects_types[0]}
        assert architecture_args['processed_objects'] in processed_objects_types, 'processed_objects keyword unknown'
        assert decoderT_type in decoderT_types, 'decoderT keyword unknown'

        self.processed_objects      = architecture_args['processed_objects']
        self.error_signal_decoder   = architecture_args['error_signal_decoder']
        self.decoderT_type          = decoderT_type
        self.z_dim_m                = z_dim_m
        self.image_width            = decoderT_args['image_width']
        self.image_height           = decoderT_args['image_height']
        self.n_image_channels       = decoderT_args['n_image_channels']
        self.bbox_pixel             = decoderT_args['bbox_pixel']
        self.error_signal_features  = decoderT_args['error_signal_features']
        self.lstm_input_size        = lstm_input_size

        # Decoder ====================================================================================================
        if decoderT_type == decoderT_types[0]:
            #-------------------------------------
            # LSTM-Decoder: trajectories as output
            #-------------------------------------
            self.decoder_lidar  = lstm_decoder(input_size=self.lstm_input_size, hidden_size=self.z_dim_m, output_size=3)   
            self.decoder_camera = lstm_decoder(input_size=self.lstm_input_size, hidden_size=self.z_dim_m, output_size=3)   

            if self.error_signal_decoder:
                self.decoder_error_signal = lstm_decoder(input_size=len(self.error_signal_features), hidden_size=self.z_dim_m, output_size=len(self.error_signal_features))
        
        if decoderT_type == decoderT_types[1]:
            #-------------------------------------
            # CNN-Decoders: images as output
            #-------------------------------------
            # Expected shape: BatchSize x FeaturesMaps x Height x Width
            if self.bbox_pixel == [100, 100]:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(self.z_dim_m, 64, 6, 2),                     # [B, z_dim_m, 1, 1] -> [B, 64, 6, 6]
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 128, 8, stride=2),                       # [B, 64, 6, 6] -> [B, 128, 18, 18]
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 128, 8, stride=2),                      # [B, 128, 18, 18] -> [B, 128, 42, 42]
                    nn.ReLU(),  
                    nn.ConvTranspose2d(128, 128, 8, stride=2),                      # [B, 128, 42, 42] -> [B, 128, 90, 90]
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 6, stride=1),                       # [B, 128, 90, 90] -> [B, 64, 95, 95]
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, self.n_image_channels, 6, stride=1),     # [B, 64, 95, 95] -> [B, 3, 100, 100]
                    nn.Sigmoid()    
                )

            elif self.bbox_pixel == [128, 256]:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(self.z_dim_m, 64, (8, 16)),     # [B, z_dim_m, 1, 1] -> [B, 32, 4, 4]
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 128, (8, 16) , stride=2),   # [B, 32, 4, 4] -> [B, 32, 8, 12]
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 128, (10, 20), stride=2),  # [B, 32, 8, 12] -> [B, 32, 8, 16]
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 128, (10, 20), stride=2),  # [B, 32, 8, 16] -> [B, 32, 16, 32]
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, (8, 10), stride=1),    # [B, 32, 16, 32] -> [B, 16, 32, 64]
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, (6, 6), stride=1),      # [B, 16, 32, 64] -> [B, 8, 64, 128]
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, self.n_image_channels, (5, 5), stride=1), # [B, 8, 64, 128] -> [B, 3, 128, 256]
                    nn.Sigmoid()                                                           # sigmoid for inputs between 0 and 1
                )

    def single_obj_processor_for_lstm(self, x, decoder, batch_size, prediction_length, sample_lengths, prev_pred_dims=3):
        prediction_list = []
        hidden_state, cell_state = None, None

        for prediction_idx in range(prediction_length):                        
            if prediction_idx == 0:
                # Transform into format [1, B, n_features]
                hidden_state = torch.moveaxis(torch.moveaxis(x, (1,3), (3,1)), (0,2), (2,0)).squeeze(0)
                cell_state = hidden_state
                prev_pred = torch.zeros((batch_size, 1, prev_pred_dims)).to(x.get_device())
            else:
                prev_pred = prediction_list[prediction_idx-1]
            # Apply to prediction_idx-th LSTM
            (pred, (hidden_state, cell_state)) = decoder(prev_pred, (hidden_state, cell_state), prediction_idx)
            prediction_list.append(pred)

        decoded_unmasked = torch.stack((prediction_list))
        decoded_unmasked = torch.moveaxis(decoded_unmasked, (0,2), (2,0)).squeeze(0)
        # mask the encodings based on the original length
        decoded = []
        for idx in range(batch_size):
            masked = torch.concat((decoded_unmasked[idx][0:sample_lengths[idx], :], 
                                   torch.zeros((prediction_length-sample_lengths[idx], prev_pred_dims)).to(x.device)))
            decoded.append(masked)
        decoded = torch.stack(decoded).to(x.device)

        return decoded

    def forward(self, x, decoder_args=None):
        if self.decoderT_type == decoderT_types[0]:
            #--------------------------------------
            # LSTM-Decoder
            #--------------------------------------
            if self.processed_objects in processed_objects_types[0] or \
               self.processed_objects in processed_objects_types[1]:
                # Lidar-object
                decoded_lidar = self.single_obj_processor_for_lstm(x, 
                                                                   self.decoder_lidar, 
                                                                   decoder_args['batch_size'], 
                                                                   decoder_args['max_length_lidar'], 
                                                                   decoder_args['sample_lengths_lidar'])
            else:
                decoded_lidar = None

            if self.processed_objects == processed_objects_types[0] or \
               self.processed_objects == processed_objects_types[2]:
                # Camera-object
                decoded_camera = self.single_obj_processor_for_lstm(x, 
                                                                    self.decoder_camera,
                                                                    decoder_args['batch_size'],
                                                                    decoder_args['max_length_camera'],
                                                                    decoder_args['sample_lengths_camera'])
            else:
                decoded_camera = None    

            if self.error_signal_decoder:
                # Error-Signal to be reconstructed by the decoder
                decoded_error_signal = self.single_obj_processor_for_lstm(x,
                                                                          self.decoder_error_signal,
                                                                          decoder_args['batch_size'],
                                                                          decoder_args['max_length_camera'],
                                                                          decoder_args['sample_lengths_camera'],
                                                                          prev_pred_dims=len(self.error_signal_features))
            else:
                decoded_error_signal = None
                
            decoded = {'decoded_lidar':         decoded_lidar,
                       'decoded_camera':        decoded_camera,
                       'decoded_error_signal':  decoded_error_signal}

        elif self.decoderT_type == decoderT_types[1]:
            #--------------------------------------
            # CNN-Decoder
            #--------------------------------------
            decoded = self.decoder(x)


        return decoded


class EncoderFull(nn.Module):
    def __init__(self, *, architecture_args=None, encoderI_type=encoderI_types[0], encoderT_type=encoderT_types[1], merge_type=merge_types[0], projector_type=projector_types[0],
                 encoderI_args=None, encoderT_args=None, merge_args=None, z_dim_t=32, z_dim_i=64, z_dim_m=64,
                 channels=1, traj_size=3, nmb_prototypes_infra=512, nmb_prototypes_merge=512, normalize_z_infra = True, normalize_z_merge = True, joint_encoding = True):
        super().__init__()
        if architecture_args == None:
            architecture_args={'processed_objects': processed_objects_types[0]}
        assert architecture_args['processed_objects'] in processed_objects_types, 'processed_objects keyword unknown'
        assert encoderI_type in encoderI_types, 'EncoderI keyword unknown'
        assert encoderT_type in encoderT_types, 'EncoderT keyword unknown'
        assert merge_type in merge_types, 'Merger keyword unknown'
        assert projector_type in projector_types, 'Projector keyword unknown'

        self.processed_objects        = architecture_args['processed_objects']
        self.encoderI_type            = encoderI_type
        self.encoderT_type            = encoderT_type
        self.merge_type               = merge_type
        self.projector_type           = projector_type
        self.normalize_z_infra        = normalize_z_infra
        self.normalize_z_merge        = normalize_z_merge
        self.projection_head_n_layers = 3
        self.joint_encoding           = joint_encoding

        self.infra_prototypes         = nn.Linear(z_dim_i, nmb_prototypes_infra, bias=False)
        self.merge_prototypes         = nn.Linear(z_dim_m, nmb_prototypes_merge, bias=False)
       
        # Trajectory Encoder==========================================================================================================================
        if self.encoderT_type == encoderT_types[0]:
            #---------------------------------------
            # LSTM-Encoder
            #---------------------------------------
            self.encoder_trajectory_camera = nn.LSTM(input_size=traj_size, hidden_size=z_dim_t, num_layers=3, batch_first=True)
            self.encoder_trajectory_lidar  = nn.LSTM(input_size=traj_size, hidden_size=z_dim_t, num_layers=3, batch_first=True)

        elif self.encoderT_type == encoderT_types[1]:
            #---------------------------------------
            # Transformer-Encoder
            #---------------------------------------
            if not encoderT_args:
                self.encoder_trajectory_camera = traj_encoder(dim_in=traj_size, dim_out=z_dim_t, depth=6, heads=8, dim_trans=128, dim_mlp=128)
                self.encoder_trajectory_lidar  = traj_encoder(dim_in=traj_size, dim_out=z_dim_t, depth=6, heads=8, dim_trans=128, dim_mlp=128)
            else:
                self.encoder_trajectory_camera = traj_encoder(dim_in=traj_size, dim_out=z_dim_t, **encoderT_args)
                self.encoder_trajectory_lidar  = traj_encoder(dim_in=traj_size, dim_out=z_dim_t, **encoderT_args)

        elif self.encoderT_type == encoderT_types[2]:
            #---------------------------------------
            # CNN-Encoder
            #---------------------------------------
            self.cnn_trajectory_encoder = models.resnet18()            
            self.cnn_trajectory_encoder.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.cnn_trajectory_encoder.fc = nn.Linear(512, z_dim_m, bias=True)

        # Projection Head==========================================================================================================================
        if self.projector_type == projector_types[0]:
            n_layers = self.projection_head_n_layers
            z_dim_ph = 64
            if n_layers == 2:
                dims = [[z_dim_t, 32], [32, z_dim_ph]]
            elif n_layers == 3:
                dims = [[z_dim_t, 32], [32, 64], [64, z_dim_ph]]
            else:
                assert 'Projection Head with n_layers = ' + str(n_layers) + ' not implemented'
            
            self.projection_head = FC(n_layers=n_layers, dims=dims)
                
        
        #  Encoder Merger==========================================================================================================================
        if self.merge_type == merge_types[0]:
            #---------------------------------------
            # FC      
            #---------------------------------------
            # z_dim_t * 2 = trajectory_camera + trajectory_lidar
            z_dim_m_in = z_dim_t * 2
            dims = [[z_dim_m_in, z_dim_m_in]]
            if not merge_args:
                n_layers = 2
                for i in range(n_layers-2):
                    dims.append([z_dim_m_in, z_dim_m_in])
                dims.append([z_dim_m_in, z_dim_m])
                self.encoder_merge = FC(n_layers=n_layers, dims=dims)
            else:
                n_layers = merge_args['n_layers']
                for i in range(n_layers-2):
                    dims.append([z_dim_m_in, z_dim_m_in])
                dims.append([z_dim_m_in, z_dim_m])
                self.encoder_merge = FC(dims=dims,**merge_args)      
              
    def forward(self, x_1, x_2=None):
        x_traj = x_1
        if self.encoderT_type == encoderT_types[2]:
            #----------------------------------------------------------
            # CNN-Encoder: joint transformation into image and encoding 
            #----------------------------------------------------------
            self.bbox_pixel = [100, 100]
            self.bbox_meter = [200, 200]
            self.min_x_meter = -50
            self.min_y_meter = -100


            x_traj_camera = x_traj['obj_camera']
            x_traj_lidar = x_traj['obj_lidar']
            if x_traj_camera.device == torch.device('cuda:0') or x_traj_camera.device == torch.device('cuda:0'):
                x_traj_camera = x_traj_camera.cpu()
                x_traj_lidar = x_traj_lidar.cpu()
            # convert trajectory to image
            x_batch = []
            for idx in range(len(x_traj_lidar)):

                if self.joint_encoding: 
                    x_traj_camera[idx] = None
                
                x_img_idx = trajectory_to_image(self,
                                                x_traj_camera[idx].numpy(),
                                                x_traj_lidar[idx].numpy())
                x_img_idx = cv2.normalize(x_img_idx, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)                             
                x_batch.append(torch.tensor(x_img_idx, dtype=torch.float64))
            x_img = torch.stack(x_batch).cuda()
            x_img = torch.moveaxis(x_img, 3, 1)
            x_img = x_img.type(torch.cuda.FloatTensor)
            
            # Remove last channel
            x_img=  x_img[:, 0:2, :, :]

            output = self.cnn_trajectory_encoder(x_img)

        else:
            #--------------------------------------------------------------------
            # LSTM- or Transformer-Encoder: separate encoding of the trajectories  
            #--------------------------------------------------------------------
            # Normalize trajectory
            # TODO put these params in proper settings
            for key in x_traj.keys():
                x_traj[key][:,:,0] = x_traj[key][:,:,0] / 200.0
                x_traj[key][:,:,1] = x_traj[key][:,:,1] / 200.0
                x_traj[key][:,:,2] = x_traj[key][:,:,2] / 40.0

            # Encoder camera trajectory
            x_traj_camera = x_traj['obj_camera'].cuda()
            traj_lengths_camera = [torch.count_nonzero(torch.tensor([x[2] for x in x_traj_camera[idx,1:]]))+1 for idx in range(len(x_traj_camera))]
            packed_trajectory_camera = torch.nn.utils.rnn.pack_padded_sequence(x_traj_camera,
                                                                        traj_lengths_camera, 
                                                                        batch_first=True, 
                                                                        enforce_sorted=False)
            z_traj_camera = self.encoder_trajectory_camera.forward(packed_trajectory_camera)

            # Encoder lidar trajectory
            x_traj_lidar = x_traj['obj_lidar'].cuda()
            traj_lengths_lidar = [torch.count_nonzero(torch.tensor([x[2] for x in x_traj_lidar[idx,1:]]))+1 for idx in range(len(x_traj_lidar))]
            packed_trajectory_lidar = torch.nn.utils.rnn.pack_padded_sequence(x_traj_lidar,
                                                                            traj_lengths_lidar,
                                                                            batch_first=True,
                                                                            enforce_sorted=False)
            z_traj_lidar  = self.encoder_trajectory_lidar.forward(packed_trajectory_lidar)

            if self.encoderT_type == encoderT_types[0]:
                # LSTM: (output, (hidden state, cell state)) -> take only hidden state valid_obj x embedding_dim
                # hidden state = more local LSTM state; cell state = more global state of the LSTMs
                z_traj_camera = z_traj_camera[1][0][0, :, :]
                z_traj_lidar = z_traj_lidar[1][0][0, :, :]

                
            if self.processed_objects == processed_objects_types[0]:
                # Merge camera and lidar trajectory
                # Process both objects (lidar and camera)
                output = self.encoder_merge(torch.cat((z_traj_camera, z_traj_lidar), 1))
            elif self.processed_objects == processed_objects_types[1]:
                # Process only lidar object
                output = z_traj_lidar
            elif self.processed_objects == processed_objects_types[2]:
                # Process only camera object
                output = z_traj_camera
            
        return output

    def normalize_prototypes(self):
        # TODO what does this function do?
        w = self.infra_prototypes.weight.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        self.infra_prototypes.weight.copy_(w)
        w = self.merge_prototypes.weight.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        self.merge_prototypes.weight.copy_(w)
        return True



class AutoencoderAdvancedPhase1(nn.Module):
    def __init__(self, *, architecture_args=None, encoderI_type=encoderI_types[0], encoderT_type=encoderT_types[1], merge_type=merge_types[0], projector_type=projector_types[0],
                 encoderI_args=None, encoderT_args=None, merge_args=None, z_dim_t=32, z_dim_i=64, z_dim_m=64,
                 channels=1, traj_size=3, normalize_z_infra=True, normalize_z_merge=True,
                 decoderT_type=decoderT_types[0], decoderT_args=None):
        super().__init__()
        if architecture_args == None:
            architecture_args = {'processed_objects': processed_objects_types[0]}

        assert architecture_args['processed_objects'] in processed_objects_types, 'processed_objects keyword unknown'
        assert encoderI_type in encoderI_types, 'EncoderI keyword unknown'
        assert encoderT_type in encoderT_types, 'EncoderT keyword unknown'
        assert merge_type in merge_types, 'Merger keyword unknown'
        assert projector_type in projector_types, 'Projector keyword unknown'
        assert decoderT_type in decoderT_types, 'decoderT keyword unknown'
        self.decoderT_type = decoderT_type

        # Encoder ========================================================================================================
        self.encoder = EncoderFull(
            architecture_args   = architecture_args,
            encoderI_type       = encoderI_type,
            encoderT_type       = encoderT_type,
            merge_type          = merge_type,
            encoderI_args       = encoderI_args,
            encoderT_args       = encoderT_args,
            merge_args          = merge_args,
            z_dim_t             = z_dim_t,
            z_dim_i             = z_dim_i,
            z_dim_m             = z_dim_m,
            channels            = channels,
            traj_size           = traj_size,
        )

        # Decoder ========================================================================================================
        self.decoder = DecoderFull(
            architecture_args=architecture_args,
            decoderT_type=decoderT_type,
            decoderT_args=decoderT_args,
            z_dim_m=z_dim_m,
        )

    def forward(self, x_1, x_2=None):
        x = x_1
        encoded = self.encoder(x)
        encoded_unsqueezed = encoded.unsqueeze(2).unsqueeze(3)

        if self.decoderT_type == decoderT_types[0]:
            #--------------------------------------
            # LSTM-Decoder
            #--------------------------------------
            sample_lengths_lidar  = [torch.count_nonzero(x['obj_lidar'][idx][1:, 2])+1  for idx in range(len(x['obj_lidar']))]
            sample_lengths_camera = [torch.count_nonzero(x['obj_camera'][idx][1:, 2])+1 for idx in range(len(x['obj_camera']))]

            decoder_args = {'decoder_type': self.decoderT_type,
                            'batch_size': x['obj_lidar'].shape[0],
                            'max_length_lidar': x['obj_lidar'].shape[1],
                            'max_length_camera': x['obj_camera'].shape[1],
                            'sample_lengths_lidar': sample_lengths_lidar,
                            'sample_lengths_camera': sample_lengths_camera, }
            # TODO sample_lengths_error_sgn ??
            decoded = self.decoder(encoded_unsqueezed, decoder_args)
        else:
            #--------------------------------------
            # Other-Decoder (CNN or MLP)
            #--------------------------------------
            decoded = self.decoder(encoded_unsqueezed)

        return encoded, decoded



class AutoencoderAdvancedPhase2(nn.Module):
    def __init__(self, *, encoderI_type=encoderI_types[0], encoderT_type=encoderT_types[1], merge_type=merge_types[0], projector_type=projector_types[0],
                 architecture_args=None, encoderI_args=None, encoderT_args=None, merge_args=None, projection_head_args=None, z_dim_t=32, z_dim_i=64, z_dim_m=64,
                 channels=1, traj_size=3, normalize_z_infra=True, normalize_z_merge=True,
                 decoderT_type=decoderT_types[0], decoderT_args=None):
        super().__init__()
        if architecture_args == None:
            architecture_args = {'processed_objects': processed_objects_types[0]}
        if projection_head_args == None:
            projection_head_args = {'type': "MLP", 
                                    'n_layers': 3}
        assert architecture_args['processed_objects'] in processed_objects_types, 'processed_objects keyword unknown'
        assert encoderI_type in encoderI_types, 'EncoderI keyword unknown'
        assert encoderT_type in encoderT_types, 'EncoderT keyword unknown'
        assert merge_type in merge_types, 'Merger keyword unknown'
        assert projector_type in projector_types, 'Projector keyword unknown'
        assert decoderT_type in decoderT_types, 'decoderT keyword unknown'
        self.decoderT_type = decoderT_type

        # Encoder ========================================================================================================
        self.encoder = EncoderFull(
            architecture_args   = architecture_args,
            encoderI_type       = encoderI_type,
            encoderT_type       = encoderT_type,
            merge_type          = merge_type,
            encoderI_args       = encoderI_args,
            encoderT_args       = encoderT_args,
            merge_args          = merge_args,
            z_dim_t             = z_dim_t,
            z_dim_i             = z_dim_i,
            z_dim_m             = z_dim_m,
            channels            = channels,
            traj_size           = traj_size,
        )
        
        # Projection Head==================================================================================================
        # TODO pass projection head args and check if input is valid with the list comprehension
        n_layers = projection_head_args['n_layers']
        self.projection_head = FC(n_layers=n_layers, dims=[(z_dim_m, z_dim_m) for idx in range(n_layers)])

        # Decoder ========================================================================================================
        self.decoder = DecoderFull(
            architecture_args   = architecture_args,
            decoderT_type       = decoderT_type,
            decoderT_args       = decoderT_args,
            z_dim_m             = z_dim_m,
        )

    def forward(self, x_1, x_2):
        h_1 = self.encoder(x_1)
        h_1_unsqueezed = h_1.unsqueeze(2).unsqueeze(3)

        h_2 = self.encoder(x_2)
        h_2_unsqueezed = h_2.unsqueeze(2).unsqueeze(3)

        # TODO 
        z_1 = self.projection_head(h_1)
        z_2 = self.projection_head(h_2)

        if self.decoderT_type == decoderT_types[0]:
            #--------------------------------------
            # LSTM-Decoder
            #--------------------------------------
            sample_1_lengths_lidar =  [torch.count_nonzero(x_1['obj_lidar'][idx][1:, 2]) +1 for idx in range(len(x_1['obj_lidar']))]
            sample_1_lengths_camera = [torch.count_nonzero(x_1['obj_camera'][idx][1:, 2])+1 for idx in range(len(x_1['obj_camera']))]
            decoder_args_1 = {'decoder_type':             self.decoderT_type,
                              'batch_size':               x_1['obj_lidar'].shape[0],
                              'max_length_lidar':         x_1['obj_lidar'].shape[1],
                              'max_length_camera':        x_1['obj_camera'].shape[1],
                              'sample_lengths_lidar':     sample_1_lengths_lidar,
                              'sample_lengths_camera':    sample_1_lengths_camera, }
            decoded_1 = self.decoder(h_1_unsqueezed, decoder_args_1)

            sample_2_lengths_lidar =  [torch.count_nonzero(x_2['obj_lidar'][idx][1:, 2]) +1 for idx in range(len(x_2['obj_lidar']))]
            sample_2_lengths_camera = [torch.count_nonzero(x_2['obj_camera'][idx][1:, 2])+1 for idx in range(len(x_2['obj_camera']))]
            decoder_args_2 = {'decoder_type':             self.decoderT_type,
                              'batch_size':               x_2['obj_lidar'].shape[0],
                              'max_length_lidar':         x_2['obj_lidar'].shape[1],
                              'max_length_camera':        x_2['obj_camera'].shape[1],
                              'sample_lengths_lidar':     sample_2_lengths_lidar,
                              'sample_lengths_camera':    sample_2_lengths_camera, }
            decoded_2 = self.decoder(h_2_unsqueezed, decoder_args_2)
        else:
            #--------------------------------------
            # Other-Decoder (CNN or MLP)
            #--------------------------------------
            decoded_1 = self.decoder(h_1_unsqueezed)
            decoded_2 = self.decoder(h_2_unsqueezed)

        return (h_1_unsqueezed, h_2_unsqueezed), (decoded_1, decoded_2)



class AE(nn.Module):
    def __init__(self, traj_num_features=6, traj_length=10):
        super().__init__()
        self.traj_num_features = traj_num_features
        self.traj_length = traj_length
        self.num_inputs = self.traj_num_features * self.traj_length
        self.encoder = nn.Sequential(
            nn.Linear(self.num_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_inputs),
        )

    def forward(self, x_1, x_2=None):
        x = x_1
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def normalize_prototypes(self):
        pass



class AE_error_signal(nn.Module):
    def __init__(self, encoderT_type, decoderT_type, architecture_args, encoderT_args, decoderT_args, merge_args, z_dim_m=16, error_signal_args={}):
        super().__init__()
        assert encoderT_type in encoderT_types, 'EncoderT keyword unknown'
        assert decoderT_type in decoderT_types, 'decoderT keyword unknown'
        assert error_signal_args != {}, "Empty dict error_signal_features"

        self.encoderT_type         = encoderT_type
        self.z_dim_m               = z_dim_m
        self.decoderT_type         = decoderT_type
        self.architecture_args     = architecture_args
        self.encoderT_args         = encoderT_args
        self.decoderT_args         = decoderT_args
        self.merge_args            = merge_args
        self.error_signal_features = error_signal_args['error_signal_features']
        self.enable_padding        = error_signal_args['enable_padding']
        self.padding_value         = error_signal_args['padding_value']
        self.data_adaption         = error_signal_args['data_adaption']
        self.norm_max_values       = error_signal_args['norm_max_values']

        self.n_error_signal_features = len(self.error_signal_features)


        self.norm_feature_range    = {x: -1 for x in self.error_signal_features}
       
        # Encoder ========================================================================================================
        if encoderT_type == encoderT_types[1]:
            # Transformer-Encoder

            if self.encoderT_args['single_feature_encoding'] == False:
                # Encode all n features together with one transformer encoder
                self.encoder = traj_encoder(dim_in     = self.n_error_signal_features, 
                                            dim_out    = z_dim_m,
                                            depth      = self.encoderT_args['depth'], 
                                            heads      = self.encoderT_args['heads'],
                                            dim_trans  = self.encoderT_args['dim_trans'], 
                                            dim_mlp    = self.encoderT_args['dim_mlp'])
                encoder_total_params_learnable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

            elif self.encoderT_args['single_feature_encoding'] == True:
                # Encode each feature separately with a transformer encoder (n features -> n transformer encoders)
                self.encoder = distributer(n_features       = self.n_error_signal_features, 
                                           z_dim_m          = z_dim_m,
                                           encoderT_args    = self.encoderT_args,
                                           merge_args       = self.merge_args)

        else:
            assert False, "Encoder undefined"

        # Decoder ========================================================================================================
        if decoderT_type == decoderT_types[0]:
            # LSTM-Decoder
            self.decoder = lstm_decoder(input_size  = self.n_error_signal_features, 
                                        hidden_size = z_dim_m, 
                                        output_size = self.n_error_signal_features,
                                        num_layers  = self.decoderT_args['num_layers'])
            decoder_total_params_learnable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        else:
            assert False, "Decoder undefined"

    
    def single_obj_processor_for_lstm(self, x, decoder, batch_size, prediction_length, sample_lengths, num_layers):
        # TODO make n_features a parameter 
        n_features = self.n_error_signal_features

        prediction_list = []
        hidden_state, cell_state = None, None

        for prediction_idx in range(prediction_length):                        
            if prediction_idx == 0:
                # Transform into format [1, B, n_features]
                hidden_state = torch.stack([x for i in range(num_layers)])
                # hidden_state = torch.moveaxis(torch.moveaxis(x, (1,3), (3,1)), (0,2), (2,0)).squeeze(0)
                cell_state = hidden_state
                prev_pred = torch.zeros((batch_size, 1, n_features)).to(x.get_device())
            else:
                prev_pred = prediction_list[prediction_idx-1]
            # Apply to prediction_idx-th LSTM
            (pred, (hidden_state, cell_state)) = decoder(prev_pred, (hidden_state, cell_state), prediction_idx)
            prediction_list.append(pred)

        decoded_unmasked = torch.stack((prediction_list))
        decoded_unmasked = torch.moveaxis(decoded_unmasked, (0,2), (2,0)).squeeze(0)
        # mask the encodings based on the original length
        decoded = []
        for idx in range(batch_size):
            masked = torch.concat((decoded_unmasked[idx][0:sample_lengths[idx], :], 
                                   torch.zeros((prediction_length-sample_lengths[idx], n_features)).to(x.device)))
            decoded.append(masked)
        decoded = torch.stack(decoded).to(x.device)

        return decoded


    def forward(self, x_1, x_2=None):
        # Get Error Signal ===============================================================================================
        error_signal_batch_padded, x_lengths, _ = calculate_error_signal_full_with_padding(batch_camera            = x_1['obj_camera'],
                                                                                           batch_lidar             = x_1['obj_lidar'],
                                                                                           batch_ego               = x_1['obj_ego'],
                                                                                           data_order_dict         = x_1['data_order_dict'],
                                                                                           included_features_types = self.error_signal_features,
                                                                                           enable_padding          = self.enable_padding,
                                                                                           padding_value           = self.padding_value)

        # Normalize the error signal
        if self.data_adaption == "normalization":
            error_signal_batch_padded = norm_error_signal(error_signal_batch_padded = error_signal_batch_padded, 
                                                          error_signal_features     = self.error_signal_features, 
                                                          norm_max_values           = self.norm_max_values)
        elif self.data_adaption == "normalization_log":
            error_signal_batch_padded = norm_error_signal_logarithmic(error_signal_batch_padded = error_signal_batch_padded, 
                                                                      error_signal_features     = self.error_signal_features, 
                                                                      norm_max_values           = self.norm_max_values)
        else:
            # Obtain the max values per feature for the train-set
            for error_signal in error_signal_batch_padded:
                for idx, key in enumerate(self.error_signal_features):
                    # let this run for one epoch and get the max values for each feature over the train-set
                    max_v = max(abs(error_signal[:, idx]))

                    if 'azimuth' in key:
                        if max_v > math.pi:
                            max_v = 2 * math.pi - max_v    
                            assert False, "This should never be here as this is already taken care of during creation of the error signal"

                    if max_v > self.norm_feature_range[key]:
                        self.norm_feature_range[key] = max_v


        
        # Encoder ========================================================================================================
        if self.encoderT_type == encoderT_types[1]:
            #--------------------------------------------------------------------
            # LSTM- or Transformer-Encoder: separate encoding of the trajectories
            #--------------------------------------------------------------------

            x = error_signal_batch_padded.cuda()
            packed_x = torch.nn.utils.rnn.pack_padded_sequence(x,
                                                               x_lengths,
                                                               batch_first=True,
                                                               enforce_sorted=False)
            
            if self.encoderT_args['single_feature_encoding'] == True:
                z = self.encoder.forward(x=x, x_lengths=x_lengths)
            else:
                z = self.encoder.forward(x=packed_x)

        else:
            assert False, "Encoder undefined"


        # Decoder ========================================================================================================
        if self.decoderT_type == decoderT_types[0]:
            #--------------------------------------
            # LSTM-Decoder
            #--------------------------------------
            x_lengths = [torch.count_nonzero(torch.tensor([item[0] for item in x[idx,1:]]))+1 for idx in range(len(x))]

            decoder_args = {'decoder_type': self.decoderT_type,
                            'batch_size':   x.shape[0],
                            'max_length':   x.shape[1],
                            'x_lengths':    x_lengths, }

            x_pred = self.single_obj_processor_for_lstm(x                   = z, 
                                                        decoder             = self.decoder,
                                                        batch_size          = decoder_args['batch_size'], 
                                                        prediction_length   = decoder_args['max_length'], 
                                                        sample_lengths      = decoder_args['x_lengths'],
                                                        num_layers=self.decoderT_args['num_layers'])
        else:
            assert False, "Decoder undefined"


        return z, x_pred



class model_0(model):
    def __init__(self, 
                architecture_type   = None,
                encoderI_type       = None, 
                encoderT_type       = None, 
                merge_type          = None, 
                decoderT_type       = None,
                architecture_args   = None,
                encoderI_args       = None,
                projection_head_args= None,
                encoderT_args       = None, 
                merge_args          = None,
                decoderT_args       = None,
                error_signal_args   = None,
                z_dim_t             = 64,
                z_dim_i             = 32,
                z_dim_m             = 128,
                channels            = 1,
                traj_size           = 3,
                idx                 = 0,
                name                = 'Advanced autoencoder approach',
                size                = None,
                n_params            = None,
                input_              = 'Trajectory of camera- and lidar-object', 
                output              = 'Reconstructed trajectory of camera- and lidar-object', 
                task                = 'Learn to separately encode the trajectories of camera and lidar and reconstruct them', 
                description         = 'advanced-autoencoder phase 1: two encoders (LSTM or Transformer) that separately encodes the camera and lidar object; \
                               two decoders (LSTM) separately reconstructing the trajectories; MSE-loss; save plot of original and reconstructed trajectory'
                ):        
        super().__init__(idx,name,size,n_params,input_,output,task,description)

        assert architecture_type in architecture_types, 'Architecture keyword unknown'
        self.architecture_type = architecture_type
        self.z_dim_m = z_dim_m

        if self.architecture_type == architecture_types[0]:
            # self.architecture_type: "Encoder"
            self.model = EncoderFull(
                encoderI_type   = encoderI_type,
                encoderT_type   = encoderT_type,
                merge_type      = merge_type,
                encoderI_args   = encoderI_args,
                encoderT_args   = encoderT_args,
                merge_args      = merge_args,
                z_dim_t         = z_dim_t,
                z_dim_i         = z_dim_i,
                z_dim_m         = z_dim_m,
                channels        = channels,
                traj_size       = traj_size,
            )

        elif self.architecture_type == architecture_types[1]:
            # self.architecture_type: "AE"
            self.model = AE(traj_num_features=6, traj_length=10)
        
        elif self.architecture_type == architecture_types[2]:
            # self.architecture_type: "AE_advanced_phase1"
            self.model = AutoencoderAdvancedPhase1(
                encoderI_type       = encoderI_type,
                encoderT_type       = encoderT_type,
                merge_type          = merge_type,
                decoderT_type       = decoderT_type,
                architecture_args   = architecture_args,
                encoderI_args       = encoderI_args,
                encoderT_args       = encoderT_args,
                merge_args          = merge_args,
                decoderT_args       = decoderT_args,
                z_dim_t             = z_dim_t,
                z_dim_i             = z_dim_i,
                channels            = channels,
                traj_size           = traj_size,
                z_dim_m             = z_dim_m,                   
            )
        elif self.architecture_type == architecture_types[3]:
            # self.architecture_type: "AE_advanced_phase2"
            self.model = AutoencoderAdvancedPhase2(
                encoderI_type       = encoderI_type,
                encoderT_type       = encoderT_type,
                merge_type          = merge_type,
                decoderT_type       = decoderT_type,
                architecture_args   = architecture_args,
                encoderI_args       = encoderI_args,
                encoderT_args       = encoderT_args,
                merge_args          = merge_args,
                projection_head_args= projection_head_args,
                decoderT_args       = decoderT_args,
                z_dim_t             = z_dim_t,
                z_dim_i             = z_dim_i,
                channels            = channels,
                traj_size           = traj_size,
                z_dim_m             = z_dim_m,
            )

        elif self.architecture_type == architecture_types[4]:
            # self.architecture_type: "AE_error_signal"
            self.model = AE_error_signal(
                encoderT_type       = encoderT_type,
                z_dim_m             = z_dim_m,
                decoderT_type       = decoderT_type,
                architecture_args   = architecture_args,
                encoderT_args       = encoderT_args,
                decoderT_args       = decoderT_args,
                merge_args          = merge_args,
                error_signal_args   = error_signal_args,
            )

    def encode(self, x_traj):
        output_encoder = self.model.encoder.forward(x_traj)
        return output_encoder
        
    def decode(self, z):      
        output_decoder = self.model.decoder.forward(z)
        return output_decoder

    def forward(self, x_traj):
        output_encoder, output_decoder = self.model.forward(x_1=x_traj, x_2=x_traj)
        return output_encoder, output_decoder
    
    def normalize_prototypes(self):
        if self.architecture_type == architecture_types[0]:
            self.model.normalize_prototypes()
        elif self.architecture_type == architecture_types[2]:
            self.model.encoder.normalize_prototypes()
        return True

def generate_model(**model_params):
    return model_0(**model_params)