
import torch
import torch.nn as nn
import numpy as np
from .DGCNN import DGCNN, DGCNNLayer
from .utils import getSkeletonOneRingIdx

class SkeletonConv(nn.Module):
    def __init__(self, inChannel= 512, layers = [256, 128, 64], jointNum = 24):
        super(SkeletonConv, self).__init__()
        self.skeletonConv1 = DGCNNLayer(intputNum = inChannel, mlp = [layers[0]], norm = "None", bias = True)
        self.skeletonConv2 = DGCNNLayer(intputNum = layers[0], mlp = [layers[1]], norm = "None", bias = True)
        self.skeletonConv3 = DGCNNLayer(intputNum = layers[1], mlp = [layers[2]], norm = "None", bias = True)

        self.skeletonOneRingIdx = getSkeletonOneRingIdx("human" if jointNum == 24 else "animal")
        self.skeletonOneRingIdx = torch.tensor(self.skeletonOneRingIdx).unsqueeze(0).long()
    def forward(self, x):
        batchNum = x.shape[0]

        if self.skeletonOneRingIdx.shape[0] != batchNum:
            self.skeletonOneRingIdx = self.skeletonOneRingIdx.repeat(batchNum, 1, 1).to(x.device)

        x1 = self.skeletonConv1(x, idx =  self.skeletonOneRingIdx)      
        x2 = self.skeletonConv2(x1, idx =  self.skeletonOneRingIdx)
        x3 = self.skeletonConv3(x2, idx =  self.skeletonOneRingIdx)

        return torch.cat((x1, x2, x3), dim = 1)

class SkeletonResNetNet(nn.Module):
    def __init__(self, inChannel= 512, res = 5, skeletonConvLayers = [256, 128, 64], jointNum = 24):
        super(SkeletonResNetNet, self).__init__()

        self.modelList = nn.ModuleList()

        for i in range(0, res - 1):
            self.modelList.append(
                nn.Sequential(SkeletonConv(inChannel = inChannel, layers = skeletonConvLayers, jointNum = jointNum),
                nn.Conv1d(sum(skeletonConvLayers), 1024, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(1024, 512, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(512, inChannel, 1)))

        self.lastMlp = nn.Sequential(
            SkeletonConv(inChannel = inChannel, layers = skeletonConvLayers, jointNum = jointNum),
            nn.Conv1d(sum(skeletonConvLayers), 512, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 3, 1)
        )

    def forward(self, x):
        for modle in self.modelList:
            x += modle(x)
        return self.lastMlp(x)

class TransformationNet(nn.Module):
    def __init__(self, jointNum = 24, dgcnnLayers = [64, 128, 256]):
        super(TransformationNet, self).__init__()
        self.geoNet = DGCNN(inChannel = 6, layers = dgcnnLayers, norm = "None", bias = True)

        self.skeletonConv = SkeletonConv(inChannel = 3 + 6 + sum(dgcnnLayers), jointNum = jointNum)
        self.jointMLP = nn.Sequential(
            nn.Conv1d((256+128+64) * 2, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 512, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 512, 1)
        )

        self.skeletonRes = SkeletonResNetNet()

    def forward(self, sV, sFacesOneRingIdx, sW, sJ,  rV, rFacesOneRingIdx, rW, rJ):

        batchNum = sV.shape[0]
        pointNum = sV.shape[1]

        sV = sV.permute(0, 2, 1)
        rV = rV.permute(0, 2, 1)
        sV = self.geoNet(sV, sFacesOneRingIdx).permute(0, 2, 1)         #[B, sum(layers), N]
        rV = self.geoNet(rV, rFacesOneRingIdx).permute(0, 2, 1)         #[B, sum(layers), N]
        sV = torch.bmm(sW, sV)                                          #[B, jointNum, 512]
        sV /= (sW.sum(-1).unsqueeze(-1) + 10e-6)                        #[B, jointNum, 512]

        rV = torch.bmm(rW, rV)                                          #[B, jointNum, 512]
        rV /= (rW.sum(-1).unsqueeze(-1) + 10e-6)                        #[B, jointNum, 512]

        sV = self.skeletonConv(torch.cat((sV, sJ), dim = -1).permute(0, 2, 1))
        rV = self.skeletonConv(torch.cat((rV, rJ), dim = -1).permute(0, 2, 1))
        resX = self.jointMLP(torch.cat((sV, rV), dim = 1))                          #[B, 512, jointNum]
        pose = self.skeletonRes(resX)
        return pose.permute(0, 2, 1)

class DeformationtionNet(nn.Module):
    def __init__(self, kintree_table):
        super(DeformationtionNet, self).__init__()

        self.kintree_table = kintree_table
        id_to_col = {self.kintree_table[1, i]: i
                for i in range(self.kintree_table.shape[1])}
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }
        
    def forward(self, V, J, pose, W):

        batch_num = V.shape[0]
        R = self.rodrigues(pose.reshape(-1, 1, 3)).reshape(batch_num, -1, 3, 3)

        results = []
        results.append(
            self.with_zeros(torch.cat((R[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
        )
        for i in range(1, self.kintree_table.shape[1]):
            results.append(
                torch.matmul(
                    results[self.parent[i]],
                    self.with_zeros(
                        torch.cat(
                            (R[:, i], torch.reshape(J[:, i, :] - J[:, self.parent[i], :], (-1, 3, 1))),
                            dim=2
                        )
                    )
                )
            )

        stacked = torch.stack(results, dim=1)
        results = stacked - \
                  self.pack(
                      torch.matmul(
                          stacked,
                          torch.reshape(
                              torch.cat((J, torch.zeros((batch_num, 24, 1), dtype=torch.float).to(V.device)),
                                        dim=2),
                              (batch_num, 24, 4, 1)
                          )
                      )
                  )

        # T = torch.tensordot(results, W, dims=([1], [1])).permute(0, 3, 1, 2)

        # rest_shape_h = torch.cat(
        #     (V, torch.ones((batch_num, V.shape[1], 1), dtype=torch.float).to(V.device)), dim=2
        # )
        
        # V = torch.matmul(T, torch.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
        # V = torch.reshape(V, (batch_num, -1, 4))[:, :, :3]

        return self.lbs(V, W, results)
    @staticmethod
    def lbs(V, W, T):
        """
        V:[B,N,3]模型顶点
        W:[B,N,M]权重
        T:[B,M,4,4]变换矩阵

        return: [B, N, 3]
        """
        batchSize = V.shape[0]
        pointNum = V.shape[1]
        nodeNum = W.shape[2]
        V=torch.cat((V, torch.ones((batchSize, pointNum, 1), device=V.device)), dim=-1) #[B,N,4]
        V = V.unsqueeze(2).repeat(1, 1, nodeNum, 1).unsqueeze(-1) #[B,N,M,4,1]
        W = W.unsqueeze(-1) #[B,N,M,1]
        T = T.unsqueeze(1).repeat(1, pointNum, 1, 1, 1) #[B,N,M,4,4]

        V = torch.matmul(T, V)      #[B,N,M,4,1]
        V = W * V.squeeze(-1)       #[B,N,M,4]
        V = V.sum(dim = 2,keepdim = False)#[B,N,4]
        return V[:,:,:3]#[B,N,3]

    @staticmethod
    def with_zeros(x):
        """
        Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

        Parameter:
        ---------
        x: Tensor to be appended.

        Return:
        ------
        Tensor after appending of shape [4,4]

        """
        ones = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float
        ).expand(x.shape[0], -1, -1).to(x.device)
        ret = torch.cat((x, ones), dim=1)
        return ret
    @staticmethod
    def rodrigues(r):
        """
        Rodrigues' rotation formula that turns axis-angle tensor into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size * angle_num, 3, 3].

        """
        eps = 1e-8
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
        theta_dim = theta.shape[0]
        r_hat = r / theta

        cos = torch.cos(theta * np.pi)
        z_stick = torch.zeros(theta_dim, dtype=torch.float).to(r.device)
        m = torch.stack(
            (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
                -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = (torch.eye(3, dtype=torch.float).unsqueeze(dim=0)
                    + torch.zeros((theta_dim, 3, 3), dtype=torch.float)).to(r.device)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    @staticmethod
    def pack(x):
        """
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

        Parameter:
        ----------
        x: A tensor of shape [batch_size, 4, 1]

        Return:
        ------
        A tensor of shape [batch_size, 4, 4] after appending.

        """
        zeros43 = torch.zeros(
            (x.shape[0], x.shape[1], 4, 3), dtype=torch.float).to(x.device)
        ret = torch.cat((zeros43, x), dim=3)
        return ret