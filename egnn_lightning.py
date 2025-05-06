'''
EGNN model 
https://github.com/lucidrains/egnn-pytorch
'''
import lightning as L 
import torch
from torch.profiler import profile, record_function, ProfilerActivity 
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F
import numpy as np 


from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from egnn_helper import exists, safe_div, batched_index_select, huber_reconstruction_loss
from egnn_pytorch import EGNN_Network
from egnn import Swish_, EGNN, PointNetDecoder
from utils import off_diagonal, LARS, get_feat_mask
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import math 
from data_code.data_utils import GraphScale
from data_code.smooth_utils import fast_batch_lowres
import time 

SiLU = nn.SiLU if hasattr(nn, "SiLU") else Swish_



class LitEGNNConsistent(L.LightningModule):
    '''
    Applies reconstruction and consistency loss. May need to refactor to apply to other, non EGNN models.
    TODO: handle the case where we need to mask input. Current convolution does NOT handle masking

    '''
    def __init__(self, cfg, profiler=None):
        super().__init__()
        self.cfg = cfg
        self.profiler = profiler 

        self.training_svm_outputs = [] # list of dictionaries containing svm training features
        self.test_svm_outputs = [] # list of ditionaries containing svm test features

        self.training_config = cfg.training
        self.model_config = cfg.model 
        self.optimizer_config = cfg.optimizer

        self.encoder_depth = self.model_config.encoder_depth
        self.encoder_init_feat_dim = self.model_config.encoder_init_feat_dim
        self.encoder_out_feat_dim = self.model_config.encoder_out_feat_dim 
        self.init_feat_dim = self.model_config.init_feat_dim
        self.num_nearest_neighbors = self.model_config.num_nearest_neighbors
        self.coor_weights_clamp_value = self.model_config.coor_weights_clamp_value
        self.norm_coors = self.model_config.norm_coors
        self.soft_edges = self.model_config.soft_edges

        self.recon_loss = self.training_config.recon_loss
        dropout = nn.Dropout(self.model_config.dropout) if self.model_config.dropout > 0 else nn.Identity()
        self.dropout = dropout          

        # convert data feat dim to encoder_init_feat_dim using mlp
        if self.init_feat_dim is not None and self.init_feat_dim != self.encoder_init_feat_dim:
            self.init_feat_embedder = nn.Sequential(
                nn.Linear(self.init_feat_dim, max(self.init_feat_dim*2, self.encoder_init_feat_dim // 2)),
                self.dropout,
                SiLU(),
                nn.Linear(max(self.init_feat_dim*2, self.encoder_init_feat_dim // 2), self.encoder_init_feat_dim),
                SiLU()
            )
        else:
            self.init_feat_embedder = nn.Identity()

        self.encoding_layers = nn.ModuleList([])

        # add encoding layers consisting of egnn's: all egnn's have same dim except possibly last 
        for i in range(0, self.encoder_depth - 1):
            self.encoding_layers.append(EGNN(self.encoder_init_feat_dim, self.encoder_init_feat_dim,\
            num_nearest_neighbors=self.num_nearest_neighbors,coor_weights_clamp_value=self.coor_weights_clamp_value,
            norm_coors=self.norm_coors, soft_edges=self.soft_edges))
        self.encoding_layers.append(EGNN(self.encoder_init_feat_dim, self.encoder_out_feat_dim,\
            num_nearest_neighbors=self.num_nearest_neighbors,coor_weights_clamp_value=self.coor_weights_clamp_value,
            norm_coors=self.norm_coors, soft_edges=self.soft_edges))

        # add pointnet segmentation network decoder 
        self.decoder_input_dim = self.encoder_out_feat_dim * 2
        if self.recon_loss:
            self.decoder = PointNetDecoder(self.decoder_input_dim)    

        self.save_hyperparameters()

    def encode(self, feats, coors, adj_mat=None, edges=None, mask=None):
        for egnn in self.encoding_layers:
            feats, coors = egnn(feats, coors, mask=mask)
        
        return feats, coors 

    def decode(self, feats):
        '''
        Returns reconstructed coordinates using encoded feature matrix
            Decoder takes in as input matrix that is concat [z, feature_matrix]

        Params:
            feats: (batch, N, encoder_out_feat_dim)


        Returns:
            x_hat (batch, N, 3)
        '''
        N = feats.shape[1]
        # calculate global representation of structure z as mean over point features
        z = torch.mean(feats, dim=1) 

        # decoder input is concat[feat, z]
        z_expanded = z.unsqueeze(1).repeat(1,N,1) #(B,N, z_dim)
        
        decode_input = torch.cat((feats, z_expanded), dim=-1) #(B, N, z_dim+out_dim,)
        
        concat_dim = 2*self.encoder_out_feat_dim
        assert decode_input.shape[2] == concat_dim

        x_hat = self.decoder(decode_input)
        return x_hat 

    def forward(self, feats, coors, adj_mat=None, edges=None, mask=None):
        '''
        Takes in feature vectors, coordinate vectors and passes to encoder with EGNN's 
            Uses the mean over the EGNN output_feat to get a global representation structure z
            concat[output_feat, z] is input to PointNet segmentation decoder, which reconsructs original coordinates
            If pairs supplied, also calculates projector(z) and projector(z') for BarlowTwins loss 

        Args:
            feats: (tensor) Initial feature vector (B, N, init_feat_dim)
            coors: (tensor) Initial coordinate vector (B, N, 3)
            mask: (tensor) Boolean mask of null padded nodes 

        Returns:
            Dictionary with keys
                z. Global representation
                x_hat (optional) Coordinate reconstruction 
                encoded_feats. Features of all nodes
        '''
        assert mask is not None, "Since we use padding we need mask"
       
        N = feats.shape[1]

        # apply MLP to transform data feature dimension to expected input feature dim if need be
        feats = self.init_feat_embedder(feats) #transform to (B,N,encoder_init_feat_dim)

        assert self.recon_loss, "If no pair loss there is no other loss currently"

        encoded_feats, encoded_coors = self.encode(feats, coors, mask=mask)

        # calculate global representation of structure z as mean over point features
        z = torch.mean(encoded_feats, dim=1) 
        assert z.shape == (feats.shape[0], self.encoder_out_feat_dim)

        # get reconstruction of coordinates with decoder
        # decoder input is concat[feat, z]
        x_hat = self.decode(encoded_feats)

        return_dict = {'z': z, 'x_hat': x_hat, 'encoded_feats': encoded_feats}
        return return_dict

    @torch.no_grad() 
    def get_svm_feat_labels(self, batch, is_test=False):
        '''
        Returns a list of features (representations from our base model) for SVM to train on and a list of target labels

        batch: (feats, coors, mask, label) where feats is (B,N,1); coors is (B,N,3); mask is (B,); label is (B,)

        Return:
            {'features': numpy.array (B,N,d), 'labels': List[np.float]} or {'test_features': numpy.array (B,N,d), 'test_labels': List[np.float]}

        '''
        was_training = self.training
        self.eval() 
        feats, coors, mask, label = batch  
        labels = label.detach().cpu().numpy()
        feats = feats.to(self.device, non_blocking=True)
        coors = coors.to(self.device, non_blocking=True)
        mask  = mask.to(self.device,  non_blocking=True)

        features = self(feats, coors, mask=mask)['encoded_feats'] 

        max_rep = features.max(dim=1)[0]
        mean_rep = features.mean(dim=1)
        sum_rep = max_rep + mean_rep      
        cat_rep = torch.cat([max_rep, mean_rep], dim=1)

        out = {
            "max_rep":  max_rep.detach().to("cpu").numpy(),
            "mean_rep": mean_rep.detach().to("cpu").numpy(),
            "sum_rep":  sum_rep.detach().to("cpu").numpy(),
            "cat_rep":  cat_rep.detach().to("cpu").numpy(),
            "labels":   labels,
        }

        if was_training:
            self.train()

        if is_test:
            out = {'test_' + k: v for k,v in out.items()}
        
        return out 

    def shared_step(self, batch, batch_idx, training=True, test=False, log_scale=True):
        '''
        Calculate reconstruction loss and consistency loss.

        batch: tuple of (feats, coors, masks)
        '''
        if not training or test: 
            self.eval()

        loss, tot_consistency_loss, tot_reconstruction_loss = 0., 0., 0. 
        loss_log_dict = {}
    
        feats, coors, masks = batch

        # first calculate z_1
        output_dict_orig = self(feats=feats, coors=coors, mask=masks)
        init_feature_matrices = output_dict_orig['encoded_feats'] #(B, N,d)

        # ensure that our z_1 does not collapse across different points 
        if self.training_config.feature_var_loss:
            var = torch.var(init_feature_matrices, dim=1)
            avg_var = torch.mean(var) #average across the point clouds and the feature dimensions
            
            #feature_var_loss = 1.0 / (avg_var + 1e-6)
            
            feature_var_loss = torch.clamp(-1 * torch.log((1/self.training_config.target_variance) * avg_var + math.exp(-1 * self.training_config.max_reg_loss) ),\
             min=0, max=self.training_config.max_reg_loss) # max(0, -ln(1/target * x + e^-max_loss) )
            
            loss += (self.training_config.feature_var_weight * feature_var_loss)
            loss_log_dict['feature_var_loss'] = feature_var_loss


        # add reconstruction loss for the original point clouds 
        if self.training_config.recon_loss:
            orig_x_hat = output_dict_orig['x_hat']
            orig_recon_loss = huber_reconstruction_loss(coors, orig_x_hat)
            tot_reconstruction_loss += orig_recon_loss
        
        if self.training_config.data_augment:
            # calculate low resolution versions based on various radius
            radii = np.random.uniform(low=self.training_config.min_radius, high=self.training_config.max_radius,\
            size=self.training_config.num_lowres_augmentations)

            batch_feats, batch_point_clouds, batch_mask, batch_features = fast_batch_lowres(coors, init_feature_matrices, radii) 

            output_dict = self(feats=batch_feats, coors=batch_point_clouds, mask=batch_mask)

            # consistency loss 
            if self.training_config.consistency_loss:
                z_t_hats = output_dict['encoded_feats']
                mse_loss = torch.nn.MSELoss()
                consistency_loss = mse_loss(z_t_hats, batch_features)
                tot_consistency_loss += consistency_loss
                loss += (self.training_config.consistency_weight * consistency_loss)
                loss_log_dict['consistency_loss'] = tot_consistency_loss

            # reconstruction loss
            if self.training_config.recon_loss:
                x_hat = output_dict['x_hat']
                # reconstruction loss should be huber loss on pairwise distance matrix 
                recon_loss = huber_reconstruction_loss(batch_point_clouds, x_hat)
                tot_reconstruction_loss += recon_loss
        
        if self.training_config.recon_loss:
            loss += (self.training_config.recon_weight * tot_reconstruction_loss)
            loss_log_dict['recon_loss'] = tot_reconstruction_loss

        loss_log_dict['loss'] = loss

        # label formatting
        return_log_dict = {}
        if training:
            for key, value in loss_log_dict.items():
                if log_scale:
                    if value >= 0: #don't log if loss is negative, if zero add a slight constant to prevent error
                        return_log_dict[f'log_train_{key}'] = math.log(value + 1e-20) 
                else:
                    return_log_dict[f'train_{key}'] = value
        elif test:
            for key, value in loss_log_dict.items():
                if log_scale:
                    if value >= 0:
                        return_log_dict[f'log_test_{key}'] = math.log(value + 1e-20) 
                else:
                    return_log_dict[f'test_{key}'] = value
        else:
            for key, value in loss_log_dict.items():
                if log_scale:
                    if value >= 0: 
                        return_log_dict[f'log_val_{key}'] = math.log(value + 1e-20) 
                else:
                    return_log_dict[f'val_{key}'] = value

        return loss, return_log_dict  

    '''
    def on_train_epoch_end(self):

        if not self.training_config.data_augment:
            return

        train_rep_dict, test_rep_dict = {}, {}
        for pool_type in ['max', 'mean', 'sum', 'cat']:
            train_rep_dict[pool_type] = np.concatenate([output_dict[f"{pool_type}_rep"] for output_dict in self.training_svm_outputs], axis=0) 
            test_rep_dict[pool_type] = np.concatenate([output_dict[f"test_{pool_type}_rep"] for output_dict in self.test_svm_outputs], axis=0)

        train_labels = np.concatenate([output_dict['labels'] for output_dict in self.training_svm_outputs], axis=0)
        test_labels = np.concatenate([output_dict['test_labels'] for output_dict in self.test_svm_outputs], axis=0)
        
        for pool_type in ['max', 'mean', 'sum', 'cat']:
            model = SVC(C = 0.01, kernel='linear')
            model.fit(train_rep_dict[pool_type], train_labels)
            test_accuracy = model.score(test_rep_dict[pool_type], test_labels)
            print(f"Pooling type {pool_type} model Linear accuracy: {test_accuracy}")
            self.log(f"{pool_type}_pool_test_accuracy", test_accuracy)
        
        # clear lists
        self.training_svm_outputs = []
        self.test_svm_outputs = []
    '''
        

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        # batch should be feats, coors, masks (but a small number, since we are going to augment the batch)
        loss, log_dict = self.shared_step(batch, batch_idx, training=True)
        
        if self.cfg.strategy != "auto":
            self.log_dict(log_dict, sync_dist=True) 
        else:
            self.log_dict(log_dict) 
        return {'loss': loss} 
    
    def validation_step(self, batch, batch_idx):
        # batch should be feats, coors, masks (but a small number, since we are going to augment the batch)
        loss, log_dict = self.shared_step(batch, batch_idx, training=False, test=False)
        if self.cfg.strategy != "auto":
            self.log_dict(log_dict, sync_dist=True) 
        else:
            self.log_dict(log_dict)
        return {'loss': loss} 
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, log_dict = self.shared_step(batch, batch_idx, training=False, test=True)
        if self.cfg.strategy != "auto":
            self.log_dict(log_dict, sync_dist=True) 
        else:
            self.log_dict(log_dict)
        return {'loss': loss} 
    
    def configure_optimizers(self):
        # Setup LARS optimizer if pair loss
        if self.optimizer_config.name == "lars":
            param_weights = []
            param_biases = [] 
            for param in self.parameters():
                if param.ndim == 1:
                    param_biases.append(param)
                else:
                    param_weights.append(param)
            parameters = [{'params': param_weights}, {'params': param_biases}]
            optim = LARS(parameters, lr=0, weight_decay=self.optimizer_config.weight_decay,\
                weight_decay_filter=True, lars_adaptation_filter=True)
        elif self.optimizer_config.name == "adamw":
            optim = torch.optim.AdamW(self.parameters(), lr=self.optimizer_config.learning_rate,\
            weight_decay=self.optimizer_config.weight_decay)
        else:
            raise ValueError("Optimizer either lars or adamW")
            
        scheduler = LinearWarmupCosineAnnealingLR(optim, warmup_epochs=self.optimizer_config.warmup_epochs, max_epochs=self.training_config.num_epochs)
        
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "monitor": "val_loss"
            }
        }

    
#### OLD CODE ######
#### OLD CODE ######
class LitEGNNPointNet(L.LightningModule):
    '''
    Torch lightning EGNN encoder with PointNet segmentation network (decoder).
        The EGNN encoder outputs features of points (N, f_dim) which is averaging to get a global representation z (length encoder_out_feat_dim);
        this is used as representation for BarlowTwin loss and moreover [concat((n, f_dim), z)] is used for reconstruction loss

        Args:
            ::cfg. Config 

            ::param encoder_depth: (int) Number of EGNN network layers used in encoder. MLP is used in between each EGNN layer to
                allow for dimension matching
            ::param encoder_init_feat_dim: (int) The initial dimension of the features that go into the encoder
            ::param encoder_out_feat_dim: (int) The output dimension of the features from the last EGNN in the encoder
            ::param init_feat_dim: (int | None) If supplied, adds MLP to match data feature dimension into the encoder_init_feat_dim
            ::param recon_loss: (Bool) Flag whether should use reconstruction loss (i.e add the decoder) 
    '''
    def __init__(self, cfg):
        super().__init__()
        self.training_config = cfg.training
        self.model_config = cfg.model 
        self.optimizer_config = cfg.optimizer

        self.encoder_depth = self.model_config.encoder_depth
        self.encoder_init_feat_dim = self.model_config.encoder_init_feat_dim
        self.encoder_out_feat_dim = self.model_config.encoder_out_feat_dim 
        self.init_feat_dim = self.model_config.init_feat_dim
        self.num_nearest_neighbors = self.model_config.num_nearest_neighbors
        self.coor_weights_clamp_value = self.model_config.coor_weights_clamp_value
        self.norm_coors = self.model_config.norm_coors
        self.soft_edges = self.model_config.soft_edges

        self.recon_loss = self.training_config.recon_loss
        dropout = nn.Dropout(self.model_config.dropout) if self.model_config.dropout > 0 else nn.Identity()
        self.dropout = dropout          

        # convert data feat dim to encoder_init_feat_dim using mlp
        if self.init_feat_dim is not None and self.init_feat_dim != self.encoder_init_feat_dim:
            self.init_feat_embedder = nn.Sequential(
                nn.Linear(self.init_feat_dim, max(self.init_feat_dim*2, self.encoder_init_feat_dim // 2)),
                self.dropout,
                SiLU(),
                nn.Linear(max(self.init_feat_dim*2, self.encoder_init_feat_dim // 2), self.encoder_init_feat_dim),
                SiLU()
            )
        else:
            self.init_feat_embedder = nn.Identity()

        self.encoding_layers = nn.ModuleList([])

        # add encoding layers consisting of egnn's: all egnn's have same dim except possibly last 
        for i in range(0, self.encoder_depth - 1):
            self.encoding_layers.append(EGNN(self.encoder_init_feat_dim, self.encoder_init_feat_dim,\
            num_nearest_neighbors=self.num_nearest_neighbors,coor_weights_clamp_value=self.coor_weights_clamp_value,
            norm_coors=self.norm_coors, soft_edges=self.soft_edges))
        self.encoding_layers.append(EGNN(self.encoder_init_feat_dim, self.encoder_out_feat_dim,\
            num_nearest_neighbors=self.num_nearest_neighbors,coor_weights_clamp_value=self.coor_weights_clamp_value,
            norm_coors=self.norm_coors, soft_edges=self.soft_edges))

        # add pointnet segmentation network decoder 
        self.decoder_input_dim = self.encoder_out_feat_dim * 2
        if self.recon_loss:
            self.decoder = PointNetDecoder(self.decoder_input_dim)    

        # add barlow projector
        self.projector_layers = nn.ModuleList([])

        projector_sizes = [self.encoder_out_feat_dim, *self.model_config.projector_sizes]
      
        for i in range(len(projector_sizes)-2):
            self.projector_layers.append(nn.Linear(projector_sizes[i], projector_sizes[i+1],bias=False))
            self.projector_layers.append(nn.BatchNorm1d(projector_sizes[i+1]))
            self.projector_layers.append(nn.ReLU(inplace=True))
        self.projector_layers.append(nn.Linear(projector_sizes[-2], projector_sizes[-1], bias=False))
        self.projector = nn.Sequential(*self.projector_layers)

        self.bn = nn.BatchNorm1d(projector_sizes[-1], affine=False)

        self.save_hyperparameters()

    def encode(self, feats, coors, adj_mat=None, edges=None, mask=None):
        for egnn in self.encoding_layers:
            feats, coors = egnn(feats, coors, mask=mask)
        
        return feats, coors 

    def decode(self, feats):
        '''
        Returns reconstructed coordinates using encoded feature matrix
            Decoder takes in as input matrix that is concat [z, feature_matrix]

        Params:
            feats: (batch, N, encoder_out_feat_dim)


        Returns:
            x_hat (batch, N, 3)
        '''
        N = feats.shape[1]
        # calculate global representation of structure z as mean over point features
        z = torch.mean(feats, dim=1) 

        # decoder input is concat[feat, z]
        z_expanded = z.unsqueeze(1).repeat(1,N,1) #(B,N, z_dim)
        
        decode_input = torch.cat((feats, z_expanded), dim=-1) #(B, N, z_dim+out_dim,)
        
        concat_dim = 2*self.encoder_out_feat_dim
        assert decode_input.shape[2] == concat_dim

        x_hat = self.decoder(decode_input)
        return x_hat 



    def forward(self, feats, coors, adj_mat=None, edges=None, mask=None, feats_prime=None, coors_prime=None, mask_prime=None):
        '''
        Takes in feature vectors, coordinate vectors and passes to encoder with EGNN's 
            Uses the mean over the EGNN output_feat to get a global representation structure z
            concat[output_feat, z] is input to PointNet segmentation decoder, which reconsructs original coordinates
            If pairs supplied, also calculates projector(z) and projector(z') for BarlowTwins loss 

        Args:
            feats: (tensor) Initial feature vector (B, N, init_feat_dim)
            coors: (tensor) Initial coordinate vector (B, N, 3)
            mask: (tensor) Boolean mask of null padded nodes 

            feats_prime: (tensor | None) Optional. If suppplied assume that we are adding barlowtwin loss 
            coors_prime: (tensor | None) Optional. If supplied assume that we are adding barlowtwin loss
            mask_prime: (tensor) Boolean mask of null padded nodes in twin. For rotation + noising, same as first mask

        Returns:
            Dictionary with keys
                z. Global representation
                z_prime. (optional) Global representation of pair 
                barlow_correlation_matrix (optional). Used for barlow loss  
                x_hat (optional) Coordinate reconstruction 
                x_hat_prime (optional) Coordinate reconstruction of pair 
                encoded_feats. Features of all nodes
                encoded_feats_prime. (optional) Features of all paired nodes   
        '''
        assert mask is not None, "Since we use padding we need mask"
        if mask_prime is None:
            mask_prime = mask 
        N = feats.shape[1]

        # apply MLP to transform data feature dimension to expected input feature dim if need be
        feats = self.init_feat_embedder(feats) #transform to (B,N,encoder_init_feat_dim)
        if feats_prime is not None:
            feats_prime = self.init_feat_embedder(feats_prime)

        if feats_prime is not None and coors_prime is not None: # pairs supplied
            encoded_feats, encoded_coors = self.encode(feats, coors, mask=mask)
            encoded_feats_prime, encoded_coors_prime = self.encode(feats_prime, coors_prime, mask=mask_prime)

            # calculate global representation of structure z as mean over point features
            z = torch.mean(encoded_feats, dim=1) 
            assert z.shape == (feats.shape[0], self.encoder_out_feat_dim)
            z_prime = torch.mean(encoded_feats_prime, dim=1)
            
            # calculate barlow projections and cross-correlation matrix 
            proj_z = self.projector(z)
            proj_z_prime = self.projector(z_prime)

            # cross correlation matrix 
            c = torch.mm(self.bn(proj_z).T, self.bn(proj_z_prime))
            c.div_(proj_z.shape[0]) 
            barlow_correlation_matrix = c 
            
            if self.recon_loss:
                # get reconstruction of coordinates with decoder; decoder input is concat[encoded_feat, z]
                
                x_hat = self.decode(encoded_feats)
                x_hat_prime = self.decode(encoded_feats_prime)

                return_dict = {'z': z, 'z_prime': z_prime, 'x_hat': x_hat, 'x_hat_prime': x_hat_prime, 'barlow_correlation_matrix': barlow_correlation_matrix,\
                    'encoded_feats': encoded_feats, 'encoded_feats_prime': encoded_feats_prime}
                return return_dict
            
            # no reconstruction loss
            return_dict = {'z': z, 'z_prime': z_prime, 'barlow_correlation_matrix': barlow_correlation_matrix,\
                    'encoded_feats': encoded_feats, 'encoded_feats_prime': encoded_feats_prime}
            return return_dict
        else: # pairs not supplied, no barlow calculation
            assert self.recon_loss, "If no pair loss there is no other loss currently"

            encoded_feats, encoded_coors = self.encode(feats, coors, mask=mask)

            # calculate global representation of structure z as mean over point features
            z = torch.mean(encoded_feats, dim=1) 
            assert z.shape == (feats.shape[0], self.encoder_out_feat_dim)

            # get reconstruction of coordinates with decoder
            # decoder input is concat[feat, z]
            x_hat = self.decode(encoded_feats)

            return_dict = {'z': z, 'x_hat': x_hat, 'encoded_feats': encoded_feats}
            return return_dict

    def shared_step(self, batch, batch_idx, training=True, test=False, log_scale=True):
        if not training: 
            self.eval()

        loss = 0. 
        loss_log_dict = {}
 
        if self.training_config.pair_loss:
            feats, feats_prime, coors, coors_prime, mask = batch
            output_dict = self(feats=feats, coors=coors,mask=mask,\
                feats_prime=feats_prime, coors_prime=coors_prime)
        else:
            feats, coors, mask = batch
            output_dict = self(feats=feats, coors=coors, mask=mask)
        
        if self.training_config.recon_loss:
            x_hat = output_dict['x_hat']
            # reconstruction loss should be huber loss on pairwise distance matrix 
            
            orig_pair_distance_matrices = torch.cdist(coors, coors)
            pred_pair_distance_matrices = torch.cdist(x_hat, x_hat)

            huber_loss = nn.HuberLoss()
            recon_loss = huber_loss(orig_pair_distance_matrices, pred_pair_distance_matrices)

            if self.training_config.pair_loss:
                x_hat_prime = output_dict['x_hat_prime']
                orig_pair_distance_matrices_prime = torch.cdist(coors_prime, coors_prime)
                pred_pair_distance_matrices_prime = torch.cdist(x_hat_prime, x_hat_prime)

                recon_loss_prime = huber_loss(orig_pair_distance_matrices_prime, pred_pair_distance_matrices_prime)
                loss += (self.training_config.recon_weight * recon_loss_prime)
                

            loss += (self.training_config.recon_weight * recon_loss)

            loss_log_dict['recon_loss'] = recon_loss

        if self.training_config.sparse_loss:
            # sparsity loss on encoded feats
            encoded_feats, encoded_feats_prime = output_dict['encoded_feats'], output_dict['encoded_feats_prime']
            sparse_loss = F.l1_loss(encoded_feats, torch.zeros_like(encoded_feats)) + \
                F.l1_loss(encoded_feats_prime, torch.zeros_like(encoded_feats_prime))
            loss += (self.training_config.sparsity_weight * sparse_loss)

            loss_log_dict['sparse_loss'] = sparse_loss
        
        if self.training_config.pair_loss:
            # barlow cross correlation loss 
            barlow_correlation_matrix = output_dict['barlow_correlation_matrix']
            feat_dim = barlow_correlation_matrix.shape[0]

            on_diag = torch.diagonal(barlow_correlation_matrix).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(barlow_correlation_matrix).pow_(2).sum() 
            barlow_loss = on_diag + self.training_config.barlow_lambda * off_diag

            loss += self.training_config.barlow_weight * barlow_loss

            # Important: need to log on_diag and off_diag normalized by D and D(D-1) respectively
            loss_log_dict['barlow_loss'] = barlow_loss 
            loss_log_dict['barlow_on_diag_loss_normed'] = on_diag / feat_dim
            loss_log_dict['barlow_off_diag_loss_normed'] = off_diag / ((feat_dim) * (feat_dim-1))
        
        loss_log_dict['loss'] = loss

        # label formatting
        return_log_dict = {}
        if training:
            for key, value in loss_log_dict.items():
                if log_scale:
                    return_log_dict[f'log_train_{key}'] = math.log(value) 
                else:
                    return_log_dict[f'train_{key}'] = value
        elif test:
            for key, value in loss_log_dict.items():
                if log_scale:
                    return_log_dict[f'log_test_{key}'] = math.log(value) 
                else:
                    return_log_dict[f'test_{key}'] = value
        else:
            for key, value in loss_log_dict.items():
                if log_scale:
                    return_log_dict[f'log_val_{key}'] = math.log(value) 
                else:
                    return_log_dict[f'val_{key}'] = value

        return loss, return_log_dict  

    def training_step(self, batch, batch_idx):
        loss, log_dict = self.shared_step(batch, batch_idx, True)
        self.log_dict(log_dict)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.shared_step(batch, batch_idx, False)
        self.log_dict(log_dict)
        return loss  
    
    def test_step(self, batch, batch_idx):
        loss, log_dict = self.shared_step(batch, batch_idx, training=False, test=True)
        self.log_dict(log_dict)
        return loss 
    
    def configure_optimizers(self):
        # Setup LARS optimizer if pair loss
        if self.optimizer_config.name == "lars":
            param_weights = []
            param_biases = [] 
            for param in self.parameters():
                if param.ndim == 1:
                    param_biases.append(param)
                else:
                    param_weights.append(param)
            parameters = [{'params': param_weights}, {'params': param_biases}]
            optim = LARS(parameters, lr=0, weight_decay=self.optimizer_config.weight_decay,\
                weight_decay_filter=True, lars_adaptation_filter=True)
        elif self.optimizer_config.name == "adamw":
            optim = torch.optim.AdamW(self.parameters(), lr=self.optimizer_config.learning_rate,\
            weight_decay=self.optimizer_config.weight_decay)
        else:
            raise ValueError("Optimizer either lars or adamW")
            
        scheduler = LinearWarmupCosineAnnealingLR(optim, warmup_epochs=self.optimizer_config.warmup_epochs, max_epochs=self.training_config.num_epochs)
        
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "monitor": "val_loss"
            }
        }

        