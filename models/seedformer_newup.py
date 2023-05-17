'''
==============================================================

SeedFormer: Point Cloud Completion
-> SeedFormer Models

==============================================================

Author: Haoran Zhou
Date: 2022-5-31

==============================================================
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from extensions.chamfer_dist import ChamferDistanceL2, ChamferDistanceL2_split, ChamferDistanceL1, ChamferDistanceL1_PM
from torch import einsum
from seed_utils.utils import vTransformer, PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, query_knn, grouping_operation, get_nearest_index, indexing_neighbor
from .build import MODELS
from seed_utils.utils import vTransformer_posenc, Knn_trans


class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024, n_knn=20):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = vTransformer(128, dim=64, n_knn=n_knn)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = vTransformer(256, dim=64, n_knn=n_knn)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, partial_cloud):
        """
        Args:
             partial_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = partial_cloud
        l0_points = partial_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 128)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points, l2_xyz, l2_points


class SeedGenerator(nn.Module):
    def __init__(self, feat_dim=512, seed_dim=128, n_knn=20, factor=2, attn_channel=True):
        super(SeedGenerator, self).__init__()
        self.uptrans = UpTransformer(256, 128, dim=64, n_knn=n_knn, use_upfeat=False, attn_channel=attn_channel, up_factor=factor, scale_layer=None)
        self.mlp_1 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=seed_dim)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat, patch_xyz, patch_feat):
        """
        Args:
            feat: Tensor (B, feat_dim, 1)
            patch_xyz: (B, 3, 128)
            patch_feat: (B, seed_dim, 128)
        """
        x1 = self.uptrans(patch_xyz, patch_feat, patch_feat, upfeat=None)  # (B, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (B, 128, 256)
        completion = self.mlp_4(x3)  # (B, 3, 256)
        return completion, x3

class UpTransformer(nn.Module):
    def __init__(self, in_channel, out_channel, dim, n_knn=20, up_factor=2, use_upfeat=True, 
                 pos_hidden_dim=64, attn_hidden_multiplier=4, scale_layer=nn.Softmax, attn_channel=True):
        super(UpTransformer, self).__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        self.use_upfeat = use_upfeat
        attn_out_channel = dim if attn_channel else 1

        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        if use_upfeat:
            self.conv_upfeat = nn.Conv1d(in_channel, dim, 1)

        self.scale = scale_layer(dim=-1) if scale_layer is not None else nn.Identity()

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        # attention layers
        self.attn_mlp = [nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
                         nn.BatchNorm2d(dim * attn_hidden_multiplier),
                         nn.ReLU()]
        if up_factor:
            self.attn_mlp.append(nn.ConvTranspose2d(dim * attn_hidden_multiplier, attn_out_channel, (up_factor,1), (up_factor,1)))
        else:
            self.attn_mlp.append(nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1))
        self.attn_mlp = nn.Sequential(*self.attn_mlp)

        # upsample previous feature
        self.upsample1 = nn.Upsample(scale_factor=(up_factor,1)) if up_factor else nn.Identity()
        self.upsample2 = nn.Upsample(scale_factor=up_factor) if up_factor else nn.Identity()

        # residual connection
        self.conv_end = nn.Conv1d(dim, out_channel, 1)
        if in_channel != out_channel:
            self.residual_layer = nn.Conv1d(in_channel, out_channel, 1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, pos, key, query, upfeat):
        """
        Inputs:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
        """

        value = self.mlp_v(torch.cat([key, query], 1)) # (B, dim, N)
        identity = value
        key = self.conv_key(key) # (B, dim, N)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)

        key = grouping_operation(key, idx_knn)  # (B, dim, N, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)

        # upfeat embedding
        if self.use_upfeat:
            upfeat = self.conv_upfeat(upfeat) # (B, dim, N)
            upfeat_rel = upfeat.reshape((b, -1, n, 1)) - grouping_operation(upfeat, idx_knn) # (B, dim, N, k)
        else:
            upfeat_rel = torch.zeros_like(qk_rel)

        # attention
        attention = self.attn_mlp(qk_rel + pos_embedding + upfeat_rel) # (B, dim, N*up_factor, k)

        # softmax function
        attention = self.scale(attention)

        # knn value is correct
        value = grouping_operation(value, idx_knn) + pos_embedding + upfeat_rel # (B, dim, N, k)
        value = self.upsample1(value) # (B, dim, N*up_factor, k)

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # (B, dim, N*up_factor)
        y = self.conv_end(agg) # (B, out_dim, N*up_factor)

        # shortcut
        identity = self.residual_layer(identity) # (B, out_dim, N)
        identity = self.upsample2(identity) # (B, out_dim, N*up_factor)

        return y+identity

class PosTransformer(nn.Module):
    def __init__(self, dim):
        super(PosTransformer, self).__init__()

        self.conv_query = nn.Conv1d(3, dim, 1, 1, 0)
        self.conv_key = nn.Conv1d(3, dim, 1, 1, 0)
        self.knn_trans = Knn_trans(dim=dim)

        self.scale = dim ** -0.5

        self.res_conv = nn.Sequential(
            nn.Conv1d(3, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1)
        )
 
    def forward(self, pos):
        """
        inputs:
            pos: (B, 3, N)
        outputs:
            out: (B, dim, N)
        """

        B, _, N = pos.shape

        query = self.conv_query(pos) # (B, dim, N)
        key = self.conv_key(pos) # (B, dim, N)
        value = self.knn_trans(pos)   # (B, dim, N)

        attn = torch.matmul(query.transpose(-2, -1), key) * self.scale  #(B, N, N)
        attn = attn.softmax(dim=-1) #(B, N, N)
        out = torch.matmul(attn, value.transpose(-2, -1)).transpose(-2, -1)  #(B, dim, N)

        res = self.res_conv(pos)

        return out+res
        

class UpLayer(nn.Module):
    """
    Upsample Layer with upsample transformers
    """
    def __init__(self, dim, seed_dim, up_factor=2, i=0, radius=1, n_knn=20, interpolate='three', attn_channel=True):
        super(UpLayer, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.n_knn = n_knn
        self.interpolate = interpolate

        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + seed_dim, layer_dims=[256, dim])

        self.uptrans1 = UpTransformer(dim, dim, dim=64, n_knn=self.n_knn, use_upfeat=True, up_factor=None)
        self.uptrans2 = UpTransformer(dim, dim, dim=64, n_knn=self.n_knn, use_upfeat=True, attn_channel=attn_channel, up_factor=self.up_factor)

        self.upsample = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=dim*2, hidden_dim=dim, out_dim=dim)

        self.mlp_delta = MLP_CONV(in_channel=dim, layer_dims=[64, 3])

    def forward(self, pcd_prev, seed, seed_feat, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            seed: (B, 256, 3)
            partial: (B, 2048, 3)

        Returns:
            pcd_new: Tensor, upsampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape

        # Collect seedfeature
        if self.interpolate == 'nearest':
            idx = get_nearest_index(pcd_prev, seed)
            feat_upsample = indexing_neighbor(seed_feat, idx).squeeze(3) # (B, seed_dim, N_prev)
        elif self.interpolate == 'three':
            # three interpolate
            idx, dis = get_nearest_index(pcd_prev, seed, k=3, return_dis=True) # (B, N_prev, 3), (B, N_prev, 3)
            dist_recip = 1.0 / (dis + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True) # (B, N_prev, 1)
            weight = dist_recip / norm # (B, N_prev, 3)
            feat_upsample = torch.sum(indexing_neighbor(seed_feat, idx) * weight.unsqueeze(1), dim=-1) # (B, seed_dim, N_prev)
        else:
            raise ValueError('Unknown Interpolation: {}'.format(self.interpolate))

        # Query mlps
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_upsample], 1)
        Q = self.mlp_2(feat_1)

        # Upsample Transformers
        H = self.uptrans1(pcd_prev, K_prev if K_prev is not None else Q, Q, upfeat=feat_upsample) # (B, 128, N_prev)
        feat_child = self.uptrans2(pcd_prev, K_prev if K_prev is not None else H, H, upfeat=feat_upsample) # (B, 128, N_prev*up_factor)

        # Get current features K
        H_up = self.upsample(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        # New point cloud
        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
        pcd_new = self.upsample(pcd_prev)
        pcd_new = pcd_new + delta

        return pcd_new, K_curr
   

@MODELS.register_module()
class SeedFormer_newup(nn.Module):
    """
    SeedFormer Point Cloud Completion with Patch Seeds and Upsample Transformer
    """
    def __init__(self, config, **kwargs):
    # def __init__(self, feat_dim=512, embed_dim=128, num_p0=512, n_knn=20, 
    # radius=1, up_factors=None, seed_factor=2, interpolate='three', attn_channel=True):
        """
        Args:
            feat_dim: dimension of global feature
            embed_dim: dimension of embedding feature
            num_p0: number of P0 coarse point cloud
            up_factors: upsampling factors
            seed_factor: seed generation factor
            interpolate: interpolate seed features (nearest/three)
            attn_channel: transformer self-attention dimension (channel/point)
        """
        super(SeedFormer_newup, self).__init__()
        self.num_p0 = config.num_p0
        self.num_p1 = 2048


        # Seed Generator
        self.feat_extractor = FeatureExtractor(out_dim=config.feat_dim, n_knn=config.n_knn)
        self.seed_generator = SeedGenerator(feat_dim=config.feat_dim, seed_dim=config.embed_dim, 
                                            n_knn=config.n_knn, factor=config.seed_factor, attn_channel=config.attn_channel)

        #get knn features
        self.postransformer1 = PosTransformer(64)
        self.postransformer2 = PosTransformer(64)
        self.postransformer3 = PosTransformer(128)

        #mlp
        self.mlp1 = nn.Sequential(
            nn.Conv1d(64, 3, 1),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            nn.Conv1d(3, 3, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 12, 1),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Conv1d(12, 12, 1)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 24, 1),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(24, 24, 1)
        )
        

        # Upsample layers
        # up_layers = []
        # for i, config.factor in enumerate(config.up_factors):
        #     up_layers.append(UpLayer(dim=config.embed_dim, seed_dim=config.embed_dim, up_factor=config.factor, 
        #                              i=i, n_knn=config.n_knn, radius=config.radius, 
        #                              interpolate=config.interpolate, attn_channel=config.attn_channel))
        # self.up_layers = nn.ModuleList(up_layers)

    def forward(self, partial_cloud):
        """
        Args:
            partial_cloud: (B, N, 3)
        output:
            List of predicted point clouds, order in [Pc, P1, P2, P3]
            [seed (B, 256, 3) 
            pcd_1 (B, 512, 3)
            pcd_2 (B, 2048, 3)
            pcd_3 (B, 16384, 3)]
        """
        # Encoder
        feat, patch_xyz, patch_feat = self.forward_encoder(partial_cloud)

        # Decoder
        pred_pcds = self.forward_decoder(feat, partial_cloud, patch_xyz, patch_feat)

        return pred_pcds

    def forward_encoder(self, partial_cloud):
        # feature extraction
        partial_cloud = partial_cloud.permute(0, 2, 1).contiguous()     #(B, 3, 2048)
        feat, patch_xyz, patch_feat = self.feat_extractor(partial_cloud) 
        #feat (B, 512, 1), patch_xyz (B, 3, 128), patch_feat (B, 256, 128)

        return feat, patch_xyz, patch_feat

    def forward_decoder(self, feat, partial_cloud, patch_xyz, patch_feat):
        """
        Args:
            feat: Tensor, (B, feat_dim, 1)
            partial_cloud: Tensor, (B, N, 3)
            patch_xyz: (B, 3, 128)
            patch_feat: (B, seed_dim, 128)
        """
        pred_pcds = []
        B, _, _ = feat.shape

        # Generate Seeds
        seed, seed_feat = self.seed_generator(feat, patch_xyz, patch_feat)
        seed = seed.permute(0, 2, 1).contiguous() # (B, num_pc, 3)
        pred_pcds.append(seed)

        # Upsample layers
        pcd = fps_subsample(torch.cat([seed, partial_cloud], 1), self.num_p0) # (B, 512, 3)
        pcd1_feat = self.postransformer1(pcd.transpose(-2, -1))   #(B, 64, 512)
        pcd1 = self.mlp1(pcd1_feat).permute(0, 2, 1).contiguous()  #(B, 512, 3)
        pred_pcds.append(pcd1)
        
        pcd2 = fps_subsample(torch.cat([pcd1, seed, partial_cloud], 1), self.num_p0) #(B, 512, 3)
        pcd2_feat = self.postransformer2(pcd2.transpose(-2, -1))   #(B, 64, 512)
        pcd2 = self.mlp2(pcd2_feat).reshape(B, 3, 2048).transpose(-2, -1)  #(B, 2048, 3)
        pred_pcds.append(pcd2)

        pcd3 = fps_subsample(torch.cat([pcd2, seed, partial_cloud], 1), self.num_p1) #(B, 2048, 3)
        pcd3_feat = self.postransformer3(pcd3.transpose(-2, -1))   #(B, 128, 2048)
        pcd3 = self.mlp3(pcd3_feat).reshape(B, 3, 16384).transpose(-2, -1)  #(B, 16384, 3)
        pred_pcds.append(pcd3)


        return pred_pcds
    
    def get_loss(self, recon, partial, gt):
        """loss function
        Args
            recon: 
                List of predicted point clouds, order in [Pc, P1, P2, P3]
                [seed (B, 256, 3) 
                pcd_1 (B, 512, 3)
                pcd_2 (B, 2048, 3)
                pcd_3 (B, 16384, 3)]
            partial: input partila point cloud (B, 2048, 3)
            gt: ground truth point cloud (B, 16384, 3)
            sqrt: CDL1 if true, else CDL2
        """
        CD = ChamferDistanceL1()
        PM = ChamferDistanceL1_PM()
        # if sqrt is False:
        #     CD = ChamferDistanceL2
        #     PM, _ = ChamferDistanceL2_split

        Pc, P1, P2, P3 = recon

        gt_2 = fps_subsample(gt, P2.shape[1])
        gt_1 = fps_subsample(gt_2, P1.shape[1])
        gt_c = fps_subsample(gt_1, Pc.shape[1])

        cdc = CD(Pc, gt_c)
        cd1 = CD(P1, gt_1)
        cd2 = CD(P2, gt_2)
        cd3 = CD(P3, gt)


        partial_matching = PM(partial, P3)

        loss_sum = (cdc + cd1 + cd2 + cd3 + partial_matching)
        loss_list = [cdc, cd1, cd2, cd3, partial_matching]
        return loss_sum, loss_list, [gt_c, gt_1, gt_2, gt]


###########################
# Recommended Architectures
###########################



# if __name__ == '__main__':

#     model = seedformer_dim128(up_factors=[1, 2, 2])
#     model = model.cuda()
#     print(model)

#     x = torch.rand(8, 2048, 3)
#     x = x.cuda()

#     y = model(x)
#     print([pc.size() for pc in y])


