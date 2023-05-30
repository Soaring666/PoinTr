import torch
import torch.nn as nn
from torch import einsum
from seed_utils.utils import query_knn, grouping_operation, PosEncode, PosEncode_3

class Self_postrans(nn.Module):
    def __init__(self, dim):
        super(Self_postrans, self).__init__()

        self.posencode = PosEncode_3(60)
        self.conv_query = nn.Conv1d(60, dim, 1, 1, 0)
        self.conv_key = nn.Conv1d(60, dim, 1, 1, 0)
        self.conv_value = nn.Conv1d(60, dim, 1, 1, 0)

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

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

        pos_emb = self.posencode(pos)  #(B, 60, N)
        query = self.conv_query(pos_emb) # (B, dim, N)
        key = self.conv_key(pos_emb) # (B, dim, N)
        value = self.conv_value(pos_emb)   # (B, dim, N)

        attn = (query.transpose(-2, -1) @ key) / self.scale  #(B, N, N)
        attn = torch.softmax(attn, -1)    #(B, N, N)

        out = attn @ value.transpose(-2, -1)  # (B, N, dim)
        out = out.transpose(-2, -1).contiguous()  # (B, dim, N)

        return out


class PosTransformer(nn.Module):
    def __init__(self):
        super(PosTransformer, self).__init__()

        self.pos_emb = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 1)
        )
 
        self.weight_mlp = nn.Sequential(
            nn.Conv2d(128, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 128, 1)
        )

        self.n_knn = 8
        self.posencode = PosEncode(60)

        self.feat_mlp = nn.Sequential(
            nn.Conv2d(60, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1)
        )
    def forward(self, pos, seed):
        """
        inputs:
            pos: (B, 3, N)
            seed: (B, 3, 256)
        outputs:
            out: (B, 128, N)
        """

        B, _, N = pos.shape
        pos_flipped = pos.permute(0, 2, 1).contiguous()
        seed_flipped = seed.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, seed_flipped, pos_flipped)

        knn_pos = grouping_operation(pos, idx_knn)  # (B, 3, N, k)
        rel_pos = pos.unsqueeze(-1) - knn_pos  # (B, 3, N, k)
        pos_emb = self.pos_emb(rel_pos)  # (B, 128, N, k)

        #get the weight of attn
        weight = self.weight_mlp(pos_emb)  # (B, 128, N, k)
        weight = torch.softmax(weight, -1)  # (B, 128, N, k)

        #use pos emb to get feats
        feat = self.posencode(knn_pos)  #(B, 60, N, K)
        feat = self.feat_mlp(feat)  #(B, 128, N, K)
        feat = feat + pos_emb

        grouped_feat = einsum('b c i j, b c i j -> b c i', weight, feat)    #(B, 128, N)

        return grouped_feat
 
class Label_emb(nn.Module):
    def __init__(self):
        super(Label_emb, self).__init__()
        self.embedding = nn.Embedding(8, 128)
        self.emb_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
    
    def forward(self, label):
        """
        inputs:
            label: (B)
        outputs:
            out: (B, 128)
        """
        B = label.shape
        label_emb = self.embedding(label)
        label_emb = self.emb_mlp(label_emb)

        return label_emb

class Label_mlp(nn.Module):
    def __init__(self, dim):
        super(Label_mlp, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(128, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(2*dim, 2*dim, 1),
            nn.GroupNorm(32, 2*dim),
            nn.LeakyReLU(0,2),
            nn.Conv1d(2*dim, dim, 1)
        )
        
    def forward(self, label, feat):
        #label: (B, 128), feat: (B, dim, N)
        label = self.layer1(label)
        label = label.unsqueeze(-1).repeat(1, 1, feat.shape[-1])
        feat = torch.cat([label, feat], dim=1)  #(B, 2*dim, N)
        feat = self.layer2(feat)    #(B, dim, N)
        
        return feat