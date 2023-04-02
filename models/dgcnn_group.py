import torch
from torch import nn
from pointnet2_ops import pointnet2_utils
# from knn_cuda import KNN
# knn = KNN(k=16, transpose_mode=False)


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist  


class DGCNN_Grouper(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    
    @staticmethod
    def fps_downsample(coor, x, num_group):
        """
        input: 
            coor: b, 3, n
            x: b, f, n 
            num_group: the number of center points
        output:
            new_coor: b, 3, num_group(the coordinates of the center point)
            new_x: b, f, num_group(the features of the center point)
        """
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)  #b, num_group

        combined_x = torch.cat([coor, x], dim=1)  #b, 3+f, n

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )   #b, 3+f, num_group

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):
        """
        input:
            coor_q: the coordinate of center points 
            x_q: (bs, nq, c)the feature of center points
            coor_k: the coordinates of all points
            x_k: (bs, nk, c)the feature of all points

        output:
            features: bs, 2*c, nq, k(16)(nsamples)
            所得到的feature是原始的x_q的feature和knn的feature与x_q的相对差cat起来的
        """

        k = 16
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
#             _, idx = knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B nq k
            idx = idx.transpose(-1, -2).contiguous()    #(B, k(nsample), S)
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)  #(B*k*S)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()  #(bs, nk, c)
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]   #(B*k*S, c)
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k) #(bs, c, nq, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x):
        """

        input: 
            x(bs, 3, np)

        output: 
            coor: (B, 3, 128), the coordinate of 128 center points
            f: (B, 128, 128), the feature of 128 center points

        """
        coor = x  #(B, 3, N)
        f = self.input_trans(x) #the feature of points(B, 8, N)

        f = self.get_graph_feature(coor, f, coor, f)    #take all points as center points(B, 16, N, 16)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0] #(B, 32, N)

        coor_q, f_q = self.fps_downsample(coor, f, 512)  #get 512 center points
        f = self.get_graph_feature(coor_q, f_q, coor, f) #(B, 64, 512, 16)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0] #(B, 64, 512)
        coor = coor_q #(B, 3, 512)

        f = self.get_graph_feature(coor, f, coor, f) #(B, 128, 512, 16)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0] #(B, 64, 512)

        coor_q, f_q = self.fps_downsample(coor, f, 128) #get 128 center points
        f = self.get_graph_feature(coor_q, f_q, coor, f) #(B, 128, 128, 16)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0] #(B, 128, 128)
        coor = coor_q

        return coor, f
