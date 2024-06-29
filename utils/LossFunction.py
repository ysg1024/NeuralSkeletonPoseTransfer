# This file contains mainly loss function related code for training
#
# PoseTransferLoss：Metrics for evaluating the methods in this paper
# 

import torch
import torch.nn as nn
import torch.nn.functional as F

def edgesLength(V, F):
    batchSize = V.shape[0]
    pointsNum = V.shape[1]
    device = V.device
    idxBase = torch.arange(0, batchSize, device=device).view(-1, 1)*pointsNum
    idx1 = F[:, :, 0] + idxBase   #[B, F]
    idx2 = F[:, :, 1] + idxBase
    idx3 = F[:, :, 2] + idxBase
    idx1 = idx1.view(-1)  #[B*F]
    idx2 = idx2.view(-1)  #[B*F]
    idx3 = idx3.view(-1)  #[B*F]
    V = V.contiguous()
    p1 = V.view(batchSize*pointsNum, -1)[idx1, :].view(batchSize, -1, 3)   #[B, F, 3]
    p2 = V.view(batchSize*pointsNum, -1)[idx2, :].view(batchSize, -1, 3)   #[B, F, 3]
    p3 = V.view(batchSize*pointsNum, -1)[idx3, :].view(batchSize, -1, 3)   #[B, F, 3]
    edges1 = torch.sqrt(torch.sum((p1 - p2)**2, dim = -1)) + 1e-5
    edges2 = torch.sqrt(torch.sum((p1 - p3)**2, dim = -1)) + 1e-5
    edges3 = torch.sqrt(torch.sum((p2 - p3)**2, dim = -1)) + 1e-5

    return edges1, edges2, edges3


def edgesLengthRadio(V1, V2, F):
    edges11, edges12, edges13 = edgesLength(V1, F)
    edges21, edges22, edges23 = edgesLength(V2, F)
    edgesLoss = torch.abs(edges11 / edges21 - 1)
    edgesLoss += torch.abs(edges12 / edges22 - 1)
    edgesLoss += torch.abs(edges13 / edges23 - 1)
    return torch.mean(edgesLoss)

class PoseTransferLoss(nn.Module):
    def __init__(self):
        super(PoseTransferLoss, self).__init__()
        self.mesLoss = nn.MSELoss()
    def __call__(self, preV, tV, F):
        recLoss = self.mesLoss(preV, tV)#PMD
        edgLoss = edgesLengthRadio(preV, tV, F)#ELR
        return recLoss, edgLoss

class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.mseLoss = nn.MSELoss()
    def forward(self, pred, target):
        batchSize = pred.shape[0]
        return self.mseLoss(pred, target)

class WeightBindingLoss(nn.Module):
    def __init__(self):
        super(WeightBindingLoss, self).__init__()
    def forward(self, pred, target):
        """
        pred:   [B, N, M], predict attentions
        target: [B, N, M], target smooth weights 
        """
        target = target.argmax(dim = 2)
        crossLoss = F.cross_entropy(pred, target, reduction='mean')
        with torch.no_grad():
            acc = (pred.argmax(dim = 1) == target).sum() / (pred.shape[0]*pred.shape[2])
        return crossLoss, acc

class SkinningLoss(nn.Module):
    def __init__(self):
        super(SkinningLoss, self).__init__()
        self.jointLoss = JointLoss()
        self.weightLoss = WeightBindingLoss()
        
    def __call__(self,  preJ, preAttention, joints, W):
        jLoss = self.jointLoss(preJ, joints)
        wLoss, acc = self.weightLoss(preAttention, W)
        return jLoss, wLoss, acc