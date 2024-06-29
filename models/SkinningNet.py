# In this document is the relevant code for the main network section
# 
# Includes Weight Module f_a（WeightBindingNet）, Joint Module f_c（JointNet）, Combined, this is SkinningNet
# "_noGIDXA" used in ablation experiment with 1-neighborhood removed

import torch
import torch.nn as nn
import numpy as np

from models.DGCNN import DGCNN, DGCNNLayer, DGCNN_origin

def getSkeletonOneRingIdx(skeletonType = "human"):
    if skeletonType == "human":
        order = np.array(
        [[1,0], [2,0], [3,0], [4, 1], [5, 2], [6, 3],[7, 4], [8, 5], [9, 6],[10,7], [11,8], [12,9], [13,9], [14,9],
        [15,12], [16,13], [17,14],[18,16], [19,17],[20,18], [21,19],[22,20], [23,21]])
        pointNum = 24
    elif skeletonType == "animal":
        order = np.array([[ 1.,  0.],[ 2.,  1.],[ 3.,  2.],[ 4.,  3.],[ 5.,  4.],[ 6.,  5.],[ 7.,  6.],[ 8.,  7.],[ 9.,  8.],[10.,  9.],[11.,  6.],[12., 11.],
       [13., 12.],[14., 13.],[15.,  6.],[16., 15.],[17.,  0.],[18., 17.],[19., 18.],[20., 19.],[21.,  0.],[22., 21.],[23., 22.],[24., 23.],[25.,  0.],
       [26., 25.],[27., 26.],[28., 27.],[29., 28.],[30., 29.],[31., 30.],[32., 16.]], dtype = np.int32)
        pointNum = 33

    e1, e2 = order[:, 0], order[:, 1]
    adjMatrix = np.zeros((pointNum, pointNum))
    adjMatrix[e1, e2] = True
    adjMatrix[e2, e1] = True
    maxNum = int(adjMatrix.sum(0).max() + 1)
    
    oneRingIdx = np.zeros((pointNum, maxNum), dtype = np.int32)
    oneRingIdx[:] = np.arange(pointNum).reshape((pointNum, 1))
    for i, adjIdx in enumerate(adjMatrix):
        oneRingIdx[i][-int(adjIdx.sum()):] = np.where(adjIdx == 1)[0]
    return oneRingIdx

class JointNet(nn.Module):
    def __init__(self, dgcnnLayers = [64, 128, 256]):
        super(JointNet, self).__init__()
        self.geoNet = DGCNN(layers = dgcnnLayers, norm = "None", bias = True)

        self.skeletonConv1 = DGCNNLayer(intputNum = 3 + sum(dgcnnLayers), mlp = [256], norm = "None", bias = True)
        self.skeletonConv2 = DGCNNLayer(intputNum = 256, mlp = [128], norm = "None", bias = True)
        self.skeletonConv3 = DGCNNLayer(intputNum = 128, mlp = [64], norm = "None", bias = True)
        self.jointMLP = nn.Sequential(
            nn.Conv1d(256+128+64, 512, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 3, 1)
        )

    def forward(self, V, W, facesOneRingIdx, skeletonOneRingIdx):

        batchNum = V.shape[0]
        pointNum = V.shape[1]
        V = V.permute(0, 2, 1)
        V = self.geoNet(V, facesOneRingIdx)   #
        
        V = V.permute(0, 2, 1)    
        V = torch.bmm(W, V)                                     #[B,M,N]*[B,N,C]
        V /= (W.sum(-1).unsqueeze(-1) + 10e-6)                  #
        V = V.permute(0, 2, 1)                                  #[B,C,M]
        
        V1 = self.skeletonConv1(V, idx = skeletonOneRingIdx)      
        V2 = self.skeletonConv2(V1, idx = skeletonOneRingIdx)
        V3 = self.skeletonConv3(V2, idx = skeletonOneRingIdx)
        joints = torch.cat((V1, V2, V3), dim = 1)                     
        joints = self.jointMLP(joints)                          #[B, 3, jointNum]

        return joints.permute(0, 2, 1)

class WeightBindingNet(nn.Module):
    def __init__(self, jointNum, dgcnnLayers = [64, 128, 256]):
        super(WeightBindingNet, self).__init__()
        self.geoNet = DGCNN(layers = dgcnnLayers)

        self.globleConv = nn.Sequential(nn.Conv1d(3+sum(dgcnnLayers), 512, 1, bias = False),
                            nn.BatchNorm1d(512),
                            nn.LeakyReLU(negative_slope=0.2))

        self.weightMlp = nn.Sequential(
                            nn.Conv1d(1024 + 3 + sum(dgcnnLayers), 1024, 1, bias = False),
                            nn.BatchNorm1d(1024),
                            nn.LeakyReLU(negative_slope=0.2),

                            nn.Conv1d(1024, 256, 1, bias = False),
                            nn.BatchNorm1d(256),
                            nn.LeakyReLU(negative_slope=0.2),

                            nn.Conv1d(256, 64, 1, bias = False),
                            nn.BatchNorm1d(64),
                            nn.LeakyReLU(negative_slope=0.2),

                            nn.Conv1d(64, jointNum, 1))

    def forward(self, V, oneRingIdx):

        batchNum = V.shape[0]
        pointNum = V.shape[1]
        V = V.permute(0, 2, 1)
        local = self.geoNet(V, oneRingIdx)                              #[B, sum(layers), N]

        V = self.globleConv(local)                                      #[B, 512, N]
        VMax = V.max(dim = -1, keepdim = True)[0].repeat(1, 1, pointNum)#[B, 512, 1]
        VMean = V.mean(dim = -1).unsqueeze(-1).repeat(1, 1, pointNum)   #[B, 512, 1]
        
        V = torch.cat((VMax, VMean, local), dim = 1)
        attention = self.weightMlp(V)                                   #[B, 24, N]

        return attention    

class SkinningNet(nn.Module):
    def __init__(self, jointNet = None, weightNet = None, jointNum = 24, dgcnnLayers = [64, 128, 256]):
        super(SkinningNet, self).__init__()
        if jointNet is not None:
            self.jointNet = jointNet
        else:
            self.jointNet = JointNet(dgcnnLayers = dgcnnLayers)
        if jointNet is not None:
            self.weightNet = weightNet
        else:
            self.weightNet = WeightBindingNet(jointNum, dgcnnLayers = dgcnnLayers)
            
        self.skeletonOneRingIdx = getSkeletonOneRingIdx("human" if jointNum == 24 else "animal")
        self.skeletonOneRingIdx = torch.tensor(self.skeletonOneRingIdx).unsqueeze(0).long()


    def forward(self, V, facesOneRingIdx, W = None):
        attentions = self.weightNet(V, facesOneRingIdx)

        if W is None:
            W = attentions.detach()
            W = (W == W.max(dim = 1, keepdim = True)[0]).float()

        skeletonOneRingIdx = self.skeletonOneRingIdx.to(V.device)
        skeletonOneRingIdx = skeletonOneRingIdx.repeat(V.shape[0], 1, 1)
        joints = self.jointNet(V, W, facesOneRingIdx, skeletonOneRingIdx)
        
        return joints, attentions                                         

class JointNet_noGIDX(nn.Module):
    def __init__(self, dgcnnLayers = [64, 128, 256]):
        super(JointNet_noGIDX, self).__init__()
        self.geoNet = DGCNN_origin(layers = dgcnnLayers, norm = "None", bias = True)

        self.skeletonConv1 = DGCNNLayer(intputNum = 3 + sum(dgcnnLayers), mlp = [256], norm = "None", bias = True)
        self.skeletonConv2 = DGCNNLayer(intputNum = 256, mlp = [128], norm = "None", bias = True)
        self.skeletonConv3 = DGCNNLayer(intputNum = 128, mlp = [64], norm = "None", bias = True)
        self.jointMLP = nn.Sequential(
            nn.Conv1d(256+128+64, 512, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 3, 1)
        )

    def forward(self, V, W, skeletonOneRingIdx):

        batchNum = V.shape[0]
        pointNum = V.shape[1]
        V = V.permute(0, 2, 1)
        V = self.geoNet(V).permute(0, 2, 1)    #[B, sum(layers), N]
        
        V = torch.bmm(W, V)                                     #[B, jointNum, 512]
        V /= (W.sum(-1).unsqueeze(-1) + 10e-6)                  #[B, jointNum, 512]
        
        V = V.permute(0, 2, 1)                                  #[B, 512, jointNum]
        V1 = self.skeletonConv1(V, idx = skeletonOneRingIdx)      
        V2 = self.skeletonConv2(V1, idx = skeletonOneRingIdx)
        V3 = self.skeletonConv3(V2, idx = skeletonOneRingIdx)
        joints = torch.cat((V1, V2, V3), dim = 1)                     
        joints = self.jointMLP(joints)                          #[B, 3, jointNum]

        return joints.permute(0, 2, 1)

class WeightBindingNet_noGIDX(nn.Module):
    def __init__(self, jointNum, dgcnnLayers = [64, 128, 256]):
        super(WeightBindingNet_noGIDX, self).__init__()
        self.geoNet = DGCNN_origin(layers = dgcnnLayers)

        self.globleConv = nn.Sequential(nn.Conv1d(3+sum(dgcnnLayers), 512, 1, bias = False),
                            nn.BatchNorm1d(512),
                            nn.LeakyReLU(negative_slope=0.2))

        self.weightMlp = nn.Sequential(
                            nn.Conv1d(1024 + 3 + sum(dgcnnLayers), 1024, 1, bias = False),
                            nn.BatchNorm1d(1024),
                            nn.LeakyReLU(negative_slope=0.2),

                            nn.Conv1d(1024, 256, 1, bias = False),
                            nn.BatchNorm1d(256),
                            nn.LeakyReLU(negative_slope=0.2),

                            nn.Conv1d(256, 64, 1, bias = False),
                            nn.BatchNorm1d(64),
                            nn.LeakyReLU(negative_slope=0.2),

                            nn.Conv1d(64, jointNum, 1))

    def forward(self, V):

        batchNum = V.shape[0]
        pointNum = V.shape[1]
        V = V.permute(0, 2, 1)
        local = self.geoNet(V)                              #[B, sum(layers), N]

        V = self.globleConv(local)                                      #[B, 512, N]
        VMax = V.max(dim = -1, keepdim = True)[0].repeat(1, 1, pointNum)#[B, 512, 1]
        VMean = V.mean(dim = -1).unsqueeze(-1).repeat(1, 1, pointNum)   #[B, 512, 1]
        
        V = torch.cat((VMax, VMean, local), dim = 1)
        attention = self.weightMlp(V)                                   #[B, 24, N]

        return attention    

class SkinningNet_noGIDX(nn.Module):
    def __init__(self, jointNet = None, weightNet = None, jointNum = 24, dgcnnLayers = [64, 128, 256]):
        super(SkinningNet_noGIDX, self).__init__()
        if jointNet is not None:
            self.jointNet = jointNet
        else:
            self.jointNet = JointNet_noGIDX(dgcnnLayers = dgcnnLayers)
        if jointNet is not None:
            self.weightNet = weightNet
        else:
            self.weightNet = WeightBindingNet_noGIDX(jointNum, dgcnnLayers = dgcnnLayers)
        self.skeletonOneRingIdx = getSkeletonOneRingIdx("human" if jointNum == 24 else "animal")
        self.skeletonOneRingIdx = torch.tensor(self.skeletonOneRingIdx).unsqueeze(0).long()


    def forward(self, V, W = None):
        attentions = self.weightNet(V)

        if W is None:
            W = attentions.detach()
            W = (W == W.max(dim = 1, keepdim = True)[0]).float()

        skeletonOneRingIdx = self.skeletonOneRingIdx.to(V.device)
        skeletonOneRingIdx = skeletonOneRingIdx.repeat(V.shape[0], 1, 1)
        joints = self.jointNet(V, W, skeletonOneRingIdx)
        
        return joints, attentions      