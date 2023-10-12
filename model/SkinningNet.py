from pyclbr import Function
import torch
import torch.nn as nn

from .DGCNN import DGCNN, DGCNNLayer, DGCNN_origin
from .utils import getSkeletonOneRingIdx

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
        V = self.geoNet(V, facesOneRingIdx).permute(0, 2, 1)    #[B, sum(layers), N]
        
        V = torch.bmm(W, V)                                     #[B, jointNum, 512]
        V /= (W.sum(-1).unsqueeze(-1) + 10e-6)                  #[B, jointNum, 512]
        
        V = V.permute(0, 2, 1)                                  #[B, 512, jointNum]
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