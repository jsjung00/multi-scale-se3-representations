'''
EGNN model 
https://github.com/lucidrains/egnn-pytorch
'''
import lightning as L 
import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from egnn_helper import exists, safe_div, batched_index_select
from egnn_pytorch import EGNN_Network

class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


SiLU = nn.SiLU if hasattr(nn, "SiLU") else Swish_

class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps 
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)
    
    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale 


class EGNN(nn.Module):
    def __init__(self, input_feat_dim, output_feat_dim, edge_dim=0, m_dim=16, fourier_features=0,
                 num_nearest_neighbors=0, dropout=0.01, init_eps=1e-3, norm_feats=True, norm_coors=False, norm_coors_scale_init=1e-2,
                 update_feats=True, update_coors=True, only_sparse_neighbors=False, valid_radius=float('inf'), m_pool_method='sum',
                 soft_edges=False, coor_weights_clamp_value=None):
        '''
        input_feat_dim: (int) initial dimension of the features of each point 
        output_feat_dim: (int) dimension of the returned output features of each point 
        fourier features seem like tricks don't need rn

        seems like both feats and coors should be updated
        '''
        super().__init__()
        assert m_pool_method in {'sum', 'mean'}, "pool is sum or mean"
        assert update_feats or update_coors, "must update either features or coordinates"
        self.fourier_features = fourier_features
        assert update_feats and update_coors, "for now update both as in egnn"
        
        self.residual = (input_feat_dim == output_feat_dim) # use residual connection for node MLP only if these equal 

        # custom assertions to turn off complex parameters
        assert fourier_features == 0 and edge_dim == 0 

        edge_input_dim = (fourier_features * 2) + (input_feat_dim * 2) + \
            edge_dim + 1  # phi_e(h_i, h_j, a_ij, distance)

        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # phi_e network to get message m_ij
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim*2),
            dropout,
            SiLU(),
            nn.Linear(edge_input_dim * 2, m_dim),
            SiLU()
        )

        # network to determine message weight
        self.edge_gate = nn.Sequential(
            nn.Linear(m_dim, 1),
            nn.Sigmoid()
        ) if soft_edges else None

        self.node_norm = nn.LayerNorm(input_feat_dim) if norm_feats else nn.Identity()
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()

        self.m_pool_method = m_pool_method

        # phi_h network that gets next layer node embedding
        self.node_mlp = nn.Sequential(
            nn.Linear(input_feat_dim + m_dim, input_feat_dim * 2),
            dropout,
            SiLU(),
            nn.Linear(input_feat_dim*2, output_feat_dim),
        ) if update_feats else None

        # phi_x network that weights difference in point vectors
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            dropout,
            SiLU(),
            nn.Linear(m_dim*4, 1)
        ) if update_coors else None

        self.num_nearest_neighbors = num_nearest_neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_radius = valid_radius

        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.init_eps = init_eps
        self.apply(self.init_)
        self.norm_coors = norm_coors

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.normal_(module.weight, std=self.init_eps)

    def forward(self, feats, coors, edges=None, mask=None, adj_mat=None):
        '''
        feats: (batch, num_points, input_feat_dim)
        coors: (batch, num_points, 3)
        edges: attribute of edge, should be None for our point cloud 
        mask: (batch, num_points) Mask has 0s where there are null points (i.e from padding)

        Returns: 
        node_out (batch, num_points, output_feat_dim) final feature vectors of points
        coors_out (batch, num_points, 3) final coordinate vectors of points 
        '''
        assert edges is None and mask is not None and adj_mat is None  #assume mask since have null padding 

    
        b, n, d, device, fourier_features, num_nearest, valid_radius, only_sparse_neighbors = * \
            feats.shape, feats.device, self.fourier_features, self.num_nearest_neighbors, self.valid_radius, self.only_sparse_neighbors

        if exists(mask):
            num_nodes = mask.sum(dim=-1)
        
        use_nearest = num_nearest > 0 or only_sparse_neighbors

        # calculate pairwise distance between points
        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = (rel_coors ** 2).sum(dim = -1, keepdim=True)

        i = j = n

        if use_nearest:
            ranking = rel_dist[..., 0].clone()

            if exists(mask):
                rank_mask = mask[:, :, None] * mask[:, None, :]
                ranking.masked_fill_(~rank_mask, 1e5)
            
            nbhd_ranking, nbhd_indices = ranking.topk(num_nearest, dim=-1, largest=False)
            nbhd_mask = nbhd_ranking <= valid_radius 

            rel_coors = batched_index_select(rel_coors, nbhd_indices, dim=2)
            rel_dist = batched_index_select(rel_dist, nbhd_indices, dim=2)

            j = num_nearest
        
        if use_nearest:
            feats_j = batched_index_select(feats, nbhd_indices, dim = 1)
        else:
            feats_j = rearrange(feats, 'b j d -> b () j d')
        
        # calculate the messages for each (i,j)
        feats_i = rearrange(feats, 'b i d -> b i () d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)
    
        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)
        m_ij = self.edge_mlp(edge_input)

        if exists(self.edge_gate):
            m_ij = m_ij * self.edge_gate(m_ij)

        if exists(mask):
            mask_i = rearrange(mask, 'b i -> b i ()')
            if use_nearest:
                mask_j = batched_index_select(mask, nbhd_indices, dim=1)
                mask = (mask_i * mask_j) & nbhd_mask
            else:
                mask_j = rearrange(mask, 'b j -> b () j')
                mask = mask_i * mask_j # (i,j) pair mask, both nodes i & j must be non null

        
        if exists(self.coors_mlp):
            # get weights for each point coordinate difference x_i - x_j
            coor_weights = self.coors_mlp(m_ij)
            coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

            rel_coors = self.coors_norm(rel_coors)

            # remove from sum of coordinate difference x_i - x_j those with a null node 
            if exists(mask):
                coor_weights.masked_fill_(~mask.bool(), 0.)
            
            if exists(self.coor_weights_clamp_value):
                clamp_value = self.coor_weights_clamp_value
                coor_weights.clamp_(min = -clamp_value, max=clamp_value)
            
            # update the coordinates by summing over all weighted coordinate differences (x_i - x_j)
            coors_out = einsum('b i j, b i j c -> b i c', coor_weights, rel_coors) + coors 
        
        if exists(self.node_mlp):
            # zero out all messages m_ij that has a null node 
            if exists(mask):
                m_ij_mask = rearrange(mask, '... -> ... ()')
                m_ij = m_ij.masked_fill(~m_ij_mask.bool(), 0.)

            # TODO: allow pool method be mean
            assert self.m_pool_method == "sum"
            m_i = m_ij.sum(dim=-2) #sum over j 

            normed_feats = self.node_norm(feats)
            node_mlp_input = torch.cat((normed_feats, m_i), dim=-1)
            node_out = self.node_mlp(node_mlp_input)
            if self.residual:
                node_out += feats 
        
        return node_out, coors_out 


class PointNetDecoder(nn.Module):
    def __init__(self, concat_dim):
        super().__init__()
        self.concat_dim = concat_dim 

        self.conv1 = torch.nn.Conv1d(self.concat_dim, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 3, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        '''
        x is concat[output_feats, z] of shape (B, N, concat_dim) where z is the global vector 

        Returns a prediction of point coordinates (B,N, 3)
        '''
        assert x.shape[-1] == self.concat_dim 
        # need to swap the 1st and 2nd dimension for the conv1d; expect (B, concat_dim, N)
        x = x.transpose(1,2).contiguous()
        assert x.shape[1] == self.concat_dim

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(1,2).contiguous()
        
        return x


class TestEGNN(EGNN_Network):
    '''
    Wraps around LucidRain EGNN Network. Adds MLP to handle initial feature matrix with wrong feature size

    pointnet_decoder: (bool) If true, then uses the output feature matrix, average feat as input to point net decoder to get the reconstruction x   
    '''
    def __init__(self, feat_init_dim, pointnet_decoder=True, **egnn_kwargs):
        super().__init__(**egnn_kwargs)
        dim = egnn_kwargs.get('dim', 32)
        self.pointnet_decoder = pointnet_decoder

        self.initial_feat_mlp = nn.Sequential(
            nn.Linear(feat_init_dim, dim * 2),
            SiLU(),
            nn.Linear(dim * 2, dim),
            SiLU()
        )
        
        decoder_input_dim = dim * 2
        self.decoder = PointNetDecoder(decoder_input_dim)    

    def forward(self, feats, *args, **kwargs):
        feats = self.initial_feat_mlp(feats) # make feats have the right dimension
        N = feats.shape[1]
        feats_out, coors_out = super().forward(feats, *args, **kwargs)

        if self.pointnet_decoder:
            z = torch.mean(feats_out, dim=1) 
            z_expanded = z.unsqueeze(1).repeat(1,N,1) #(B,N, z_dim)
            decode_input = torch.cat((feats_out, z_expanded), dim=-1)

            x_hat = self.decoder(decode_input)
            
            return z, x_hat, feats_out 
         
        return feats_out, coors_out 



class AE_EGNN(nn.Module):
    '''
    Autoencoder setup with EGNN layers. "Down/upsamples" by using MLP in between EGNN layers 
        NOTE: Do not need to use MLP! Fix because you can just change in input dimension and output dimension of EGNN!

    Supports both barlowtwins representation learning (i.e require pair of data) and normal learning
    '''
    def __init__(
        self,
        *,
        depth,
        dim,
        init_feat_dim=3, # dimension of the features that are supplied 
        edge_dim = 0,
        num_adj_degrees=None,
        adj_dim=0,
        dropout=0.01):
        super().__init__()
        
        self.encoding_layers = nn.ModuleList([])
        self.decoding_layers = nn.ModuleList([])
        self.init_feat_dim = init_feat_dim

        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout = dropout 
        self.projector_layers = nn.ModuleList([])
        for ind in range(depth):
            if ind == 0: prev_dim = init_feat_dim 
            current_dim = dim // (2 ** ind) if (dim //  (2 ** ind)) > 0 else 1 
            # MLP to embed features of prev_dim to current_dim
            self.encoding_layers.append(self.get_resizing_mlp(prev_dim, current_dim))
            # EGNN layer 
            self.encoding_layers.append(EGNN(dim = current_dim, edge_dim=0))
            prev_dim = current_dim

        projector_sizes = [current_dim, 1024, 1024, 1024]
        for i in range(len(projector_sizes)-2):
            self.projector_layers.append(nn.Linear(projector_sizes[i], projector_sizes[i+1],bias=False))
            self.projector_layers.append(nn.BatchNorm1d(projector_sizes[i+1]))
            self.projector_layers.append(nn.ReLU(inplace=True))
        self.projector_layers.append(nn.Linear(projector_sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*self.projector_layers)

        for ind in range(depth):
            cur_dim = dim // (2 ** (depth - 1 - ind)) if (dim // (2 ** (depth - 1 - ind))) > 0 else 1
            # MLP to embed features of prev_dim to current_dim
            self.decoding_layers.append(self.get_resizing_mlp(prev_dim, cur_dim))
            self.decoding_layers.append(EGNN(dim=cur_dim, edge_dim=0))
            prev_dim = cur_dim

    def get_resizing_mlp(self, input_dim, output_dim):
        # Returns a MLP to convert embedding of size input_dim to size output_dim
        return nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(output_dim*2, output_dim),
            SiLU()
        )

    def encode(self, feats, coors, adj_mat=None, edges=None, mask=None):
        for i, net in enumerate(self.encoding_layers):
            # first embed feats into right dim with MLP
            if i % 2 == 0:
                feats = net(feats) 
            else:
                feats, coors = net(feats, coors, adj_mat=adj_mat, edges=edges, mask=mask)
        return feats, coors 

    def decode(self, feats, coors, adj_mat=None, edges=None, mask=None):
        for i, net in enumerate(self.decoding_layers):
            # first embed feats into right dim with MLP
            if i % 2 == 0:
                feats = net(feats)
            else:
                feats, coors = net(feats, coors, adj_mat=adj_mat, edges=edges, mask=mask)
        return feats, coors 

    def forward(self, feats, coors, adj_mat=None, edges=None, mask = None, feats_prime=None, coors_prime=None):
        '''
        returns intermediate_feats, final coors (final coors for reconstruction loss, intermediate feats for representation)

        feats_prime: (tensor | None) Optional. If suppplied assume that we are adding barlowtwin loss 
        coors_prime: (tensor | None) Optional. If supplied assume that we are adding barlowtwin loss

        '''
        # TODO: confirm if a problem if reconstruction loss is on coordimates 
      
        if feats_prime is not None and coors_prime is not None: # add barlow twins loss
            # encoding 
            feats, coors = self.encode(feats, coors, adj_mat=adj_mat, edges=edges, mask=mask)
            feats_prime, coors_prime = self.encode(feats_prime, coors_prime, adj_mat=adj_mat, edges=edges, mask=mask)
            
            proj, proj_prime = self.projector()
            #Q: similarity on features or coors or both
            # TODO: add barlow projector and return the projected embedding
            # https://github.com/MaxLikesMath/Barlow-Twins-Pytorch
            
            # save the intermediate encoded features and encoded coordinates 
            inter_feats, inter_feats_prime =  feats.clone(), feats_prime.clone()
            inter_coors, inter_coors_prime = coors.clone(), coors_prime.clone()

             # decoding
            final_feats, final_coors = self.decode(feats, coors, adj_mat=adj_mat, edges=edges, mask=mask)
            final_feats_prime, final_coors_prime = self.decode(feats_prime, coors_prime, adj_mat=adj_mat, edges=edges, mask=mask)

            return inter_feats, inter_feats_prime, inter_coors, inter_coors_prime, final_feats, final_feats_prime,\
                final_coors, final_coors_prime
        else:
            feats, coors = self.encode(feats, coors, adj_mat=adj_mat, edges=edges, mask=mask)
            inter_feats =  feats.clone()
            inter_coors = coors.clone()
            final_feats, final_coors = self.decode(feats, coors, adj_mat=adj_mat, edges=edges, mask=mask)
        
            return inter_feats, inter_coors, final_feats, final_coors  



class EGNN_PointNet(nn.Module):
    '''
    EGNN encoder with PointNet segmentation network (decoder).
        The EGNN encoder outputs features of poitns (N, f_dim) which is averaging to get a global representation z (length encoder_out_feat_dim);
        this is used as representation for BarlowTwin loss and moreover [concat((n, f_dim), z)] is used for reconstruction loss

    Note: Unlike AE_EGNN, don't use resizing MLPs in between. Moreover, don't see a need to change the feat dim in between the EGNN layers,
        except for the last layer 

    encoder_depth: (int) Number of EGNN network layers used in encoder. 

    encoder_init_feat_dim: (int) The initial dimension of the features that go into the encoder
    encoder_out_feat_dim: (int) The output dimension of the features from the last EGNN in the encoder
    init_feat_dim: (int | None) If supplied, adds MLP to match data feature dimension into the encoder_init_feat_dim
    recon_loss: (Bool) Flag whether should use reconstruction loss (i.e add the decoder) 
    '''
    def __init__(self, encoder_depth, encoder_init_feat_dim, encoder_out_feat_dim,\
        num_nearest_neighbors, coor_weights_clamp_value, norm_coors, dropout=0.01, init_feat_dim=None, recon_loss=False):
        super().__init__()
        self.encoder_depth = encoder_depth 
        self.encoder_init_feat_dim = encoder_init_feat_dim
        self.encoder_out_feat_dim = encoder_out_feat_dim 
        self.init_feat_dim = init_feat_dim
        self.num_nearest_neighbors = num_nearest_neighbors
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.norm_coors = norm_coors
        self.recon_loss = recon_loss 

        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout = dropout 

        # convert data feat dim to encoder_init_feat_dim using mlp
        if self.init_feat_dim is not None and self.init_feat_dim != self.encoder_init_feat_dim:
            self.init_feat_embedder = nn.Sequential(
                nn.Linear(init_feat_dim, max(init_feat_dim*2, encoder_init_feat_dim // 2)),
                self.dropout,
                SiLU(),
                nn.Linear(max(init_feat_dim*2, encoder_init_feat_dim // 2), encoder_init_feat_dim),
                SiLU()
            )
        else:
            self.init_feat_embedder = nn.Identity()

        self.encoding_layers = nn.ModuleList([])

        # add encoding layers consisting of egnn's: all egnn's have same dim except possibly last 
        for i in range(0, self.encoder_depth - 1):
            self.encoding_layers.append(EGNN(encoder_init_feat_dim, encoder_init_feat_dim,\
            num_nearest_neighbors=self.num_nearest_neighbors,coor_weights_clamp_value=self.coor_weights_clamp_value,
            norm_coors=self.norm_coors))
        self.encoding_layers.append(EGNN(encoder_init_feat_dim, encoder_out_feat_dim,\
            num_nearest_neighbors=self.num_nearest_neighbors,coor_weights_clamp_value=self.coor_weights_clamp_value,
            norm_coors=self.norm_coors))

        # add pointnet segmentation network decoder 
        decoder_input_dim = encoder_out_feat_dim * 2
        if self.recon_loss:
            self.decoder = PointNetDecoder(decoder_input_dim)    

        # add barlow projector
        self.projector_layers = nn.ModuleList([])
        #TODO: add larger projector sizes
        projector_sizes = [self.encoder_out_feat_dim, 128]
        #projector_sizes = [self.encoder_out_feat_dim, 1024, 1024, 1024]
        for i in range(len(projector_sizes)-2):
            self.projector_layers.append(nn.Linear(projector_sizes[i], projector_sizes[i+1],bias=False))
            self.projector_layers.append(nn.BatchNorm1d(projector_sizes[i+1]))
            self.projector_layers.append(nn.ReLU(inplace=True))
        self.projector_layers.append(nn.Linear(projector_sizes[-2], projector_sizes[-1], bias=False))
        self.projector = nn.Sequential(*self.projector_layers)

        self.bn = nn.BatchNorm1d(projector_sizes[-1], affine=False)


    def encode(self, feats, coors, adj_mat=None, edges=None, mask=None):
        for egnn in self.encoding_layers:
            feats, coors = egnn(feats, coors, mask=mask)
        
        return feats, coors 

    def forward(self, feats, coors, adj_mat=None, edges=None, mask=None, feats_prime=None, coors_prime=None, mask_prime=None):
        '''
        Takes in feature vectors, coordinate vectors and passes to encoder with EGNN's 
            Uses the mean over the EGNN output_feat to get a global representation structure z
            concat[output_feat, z] is input to PointNet segmentation decoder, which reconsructs original coordinates

            If pairs supplied, also calculates projector(z) and projector(z') for BarlowTwins loss 

        Returns z (global representation), \hat{x} (coors reconstruction from decoder) 
            (if pair supplied) proj(z), proj(z')
        

        feats: (tensor) Initial feature vector (B, N, init_feat_dim)
        coors: (tensor) Initial coordinate vector (B, N, 3)
        mask: (tensor) Boolean mask of null padded nodes 

        feats_prime: (tensor | None) Optional. If suppplied assume that we are adding barlowtwin loss 
        coors_prime: (tensor | None) Optional. If supplied assume that we are adding barlowtwin loss
        mask_prime: (tensor) Boolean mask of null padded nodes in twin. For rotation + noising, same as first mask
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
                z_expanded = z.unsqueeze(1).repeat(1,N,1) #(B,N, z_dim)
                z_expanded_prime = z_prime.unsqueeze(1).repeat(1,N,1)

                decode_input = torch.cat((encoded_feats, z_expanded), dim=-1) #(B, N, z_dim+out_dim)
                decode_input_prime = torch.cat((encoded_feats_prime, z_expanded_prime), dim=-1) # (B, N, z_dim+out_dim)
                
                concat_dim = 2*self.encoder_out_feat_dim
                assert decode_input.shape[-1] == concat_dim

                x_hat = self.decoder(decode_input)
                x_hat_prime = self.decoder(decode_input_prime)
                
                return z, z_prime, x_hat, x_hat_prime, barlow_correlation_matrix, encoded_feats, encoded_feats_prime
            
            # no reconstruction loss
            return z, z_prime, barlow_correlation_matrix, encoded_feats, encoded_feats_prime  
        else: # pairs not supplied, no barlow calculation
            encoded_feats, encoded_coors = self.encode(feats, coors, mask=mask)

            # calculate global representation of structure z as mean over point features
            z = torch.mean(encoded_feats, dim=1) 
            assert z.shape == (feats.shape[0], self.encoder_out_feat_dim)

            # get reconstruction of coordinates with decoder
            # decoder input is concat[feat, z]
            z_expanded = z.unsqueeze(1).repeat(1,N,1) #(B,N, z_dim)
            
            decode_input = torch.cat((encoded_feats, z_expanded), dim=-1) #(B, N, z_dim+out_dim,)
            
            concat_dim = 2*self.encoder_out_feat_dim
            assert decode_input.shape[2] == concat_dim

            x_hat = self.decoder(decode_input)

            return z, x_hat, encoded_feats


