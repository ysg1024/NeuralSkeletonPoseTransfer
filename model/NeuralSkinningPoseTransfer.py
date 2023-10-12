from ast import Not
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import lbs, dqs
from .utils import getLaplacianMatrix, getBetaMatrix
from .utils import getFacesOneRingIdx
from .SkeletonTransfer import AddVirtualJoints, skeletonTransferWithVirtualJoints, AddVirtualJoints_Animal, skeletonTransferWithVirtualJoints_Animal
import igl

def poseTransfer(net, sV, sFOneRingIdx, rV, rFOneRingIdx, laplacian, blendShape = "lbs", dLambda = 20, modelType = "human"):
    with torch.no_grad():
        preSJ, sAttention = net(sV, sFOneRingIdx)
        preRJ, rAttention = net(rV, rFOneRingIdx)

        sRigW = (sAttention == sAttention.max(dim = 1, keepdim = True)[0]).float().permute(0, 2, 1)
        rRigW = (rAttention == rAttention.max(dim = 1, keepdim = True)[0]).float().permute(0, 2, 1)

        if modelType == "human":
            sVJoints = AddVirtualJoints(preSJ, sV, sRigW)
            rVJoints = AddVirtualJoints(preRJ, rV, rRigW)
            M, preTJ = skeletonTransferWithVirtualJoints(sVJoints, rVJoints)
        else:
            sVJoints = AddVirtualJoints_Animal(preSJ, sV, sRigW)
            rVJoints = AddVirtualJoints_Animal(preRJ, rV, rRigW)
            M, preTJ = skeletonTransferWithVirtualJoints_Animal(sVJoints, rVJoints)

        Beta = getBetaMatrix(laplacian, dLambda)
        W = torch.bmm(Beta, sRigW)
        if blendShape == 'lbs':
            preV = lbs(sV, W, M)
        elif blendShape == 'dqs':
            preV = dqs(sV, W, M)
    return preV, preSJ, preRJ, preTJ, sRigW, rRigW

def poseTransferForOne(net, sV, sF, rV, rF, device, blendShape = 'lbs', dLambda = 20, modelType = "human"):
    
    sV = torch.tensor(sV).to(device).float().unsqueeze(0)
    rV = torch.tensor(rV).to(device).float().unsqueeze(0)
    sFOneRingIdx = torch.tensor(getFacesOneRingIdx(sF)).to(device).long().unsqueeze(0)
    rFOneRingIdx = torch.tensor(getFacesOneRingIdx(rF)).to(device).long().unsqueeze(0)
    sF = torch.tensor(sF).unsqueeze(0).to(device).long()
    rF = torch.tensor(rF).unsqueeze(0).to(device).long()

    laplacian = getLaplacianMatrix(sV, sF)
    preV, preSJ, preRJ, preTJ, sRigW, rRigW = poseTransfer(net, sV, sFOneRingIdx, rV, rFOneRingIdx, laplacian, blendShape, dLambda, modelType)
    preV = preV[0].cpu().numpy()
    preSJ = preSJ[0].cpu().numpy()
    preRJ = preRJ[0].cpu().numpy()
    preTJ = preTJ[0].cpu().numpy()
    sRigW = sRigW[0].cpu().numpy()
    rRigW = rRigW[0].cpu().numpy()
    return preV, preSJ, preRJ, preTJ, sRigW, rRigW