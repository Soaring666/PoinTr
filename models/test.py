import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from seed_utils.utils import distance_knn

class Denoise(nn.Module):
    def __init__(self, nsample, k_denoise):
        super(Denoise, self).__init__()
        self.nsample = nsample
        self.k_denoise = k_denoise

    def forward(self, pcd):
        #pcd (B, N, 3)
        B, N, _ = pcd.shape
        dist_knn = distance_knn(self.nsample, pcd, pcd)   #(B, 512, nsample)
        dist_knn = dist_knn.mean(dim=-1)    #(B, 512)
        #找到离群点
        idx = torch.argsort(dist_knn, dim=1, descending=True)   #(B, 512)
        print(idx)
        new_pcd = gather_operation(pcd.transpose(-2, -1).contiguous(), idx.int())  #(B, 3, 512)
        new_pcd = new_pcd.transpose(-2, -1)  #(B, 512, 3)
        print(new_pcd)
        for i in range(B):
            for j in range(self.k_denoise):
                new_pcd[i, j, :] = new_pcd[i, -j-1, :]

        return new_pcd

denoise = Denoise(2, 3).cuda()
x = torch.randint(5, (2, 10, 3)).float().cuda()
print(x)
x = denoise(x)
print(x)