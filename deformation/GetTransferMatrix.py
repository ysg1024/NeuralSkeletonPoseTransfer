import torch
import numpy as np
from scipy.spatial.transform import Rotation

def getRotationByAxis(angle, axis):
    batchSize = angle.shape[0]
    cosA = torch.cos(angle).unsqueeze(-1)
    sinA = torch.sin(angle).unsqueeze(-1)
    eyes = torch.eye(3, device = axis.device).unsqueeze(0).repeat(batchSize, 1, 1)
    R = torch.zeros(batchSize, 3, 3, device = axis.device)
    R[:, 0, 1] = -axis[:, 2, 0]
    R[:, 0, 2] = axis[:, 1, 0]
    R[:, 1, 0] = -R[:, 0, 1]
    R[:, 1, 2] = -axis[:, 0, 0]
    R[:, 2, 0] = -R[:, 0, 2]
    R[:, 2, 1] = -R[:, 1, 2]

    R = cosA * eyes + (1 - cosA) * torch.bmm(axis, axis.permute(0, 2, 1)) + sinA * R
    return R.permute(0, 2, 1)

def getVectorTranform(rootV, V1, V2):
    """
    rootV:[B, 3, 1]
    V1:[B, 3, 1]
    V2:[B, 3, 1]
    """
    batchSize = rootV.shape[0]
    T = torch.zeros(batchSize, 4, 4, device = rootV.device)
    axis = torch.cross(V2, V1)
    axis /= torch.norm(axis, dim = 1).unsqueeze(-1)
    angle = torch.bmm(V1.permute(0, 2, 1), V2).squeeze(-1)/(torch.norm(V1, dim = 1) *  torch.norm(V2, dim = 1))
    angle = torch.acos(angle)

    R = getRotationByAxis(angle = angle, axis = axis)
    
    t = rootV - torch.bmm(R, rootV)
    T[:, 0:3, 0:3] = R
    T[:, 0:3, 3] = t.view(-1, 3)
    T[:, 3, 3] = 1
    return T

def skeletonTransferWithVirtualJoints(sourceJoints, targetJoints):
    """
    sourceJoints: [B, 29, 3]
    targetJoints: [B, 29, 3]
    return: [B, 24, 4, 4]
    """
    order = {
    1:0, 2:0, 3:0, 
    4:1, 5:2, 6:3,
    7:4, 8:5, 9:6,
    10:7, 11:8, 12:9, 13:9, 14:9,
    15:12, 16:13, 17:14,
    18:16, 19:17,
    20:18, 21:19,
    22:20, 23:21,
    24:10, 25:11, 26:15, 27:22, 28:23 }
    batchSize = sourceJoints.shape[0]
    jointNum = sourceJoints.shape[1]

    sourceJoints = torch.cat((sourceJoints, torch.ones(batchSize, jointNum, 1, device = sourceJoints.device)), dim = -1).unsqueeze(-1)
    targetJoints = torch.cat((targetJoints, torch.ones(batchSize, jointNum, 1, device = sourceJoints.device)), dim = -1).unsqueeze(-1)
    M = torch.eye(4, device = sourceJoints.device).view(1, 1, 4, 4).repeat(batchSize, 29, 1, 1)
    #Saving relative transformations
    M_local = torch.eye(4, device = sourceJoints.device).view(1, 1, 4, 4).repeat(batchSize, 29, 1, 1)
    #
    newJoint = sourceJoints.clone()
    for idx in order:
        pIdx = order[idx]
        newPJoint = torch.bmm(M[:, pIdx], sourceJoints[:, pIdx])
        newCJoint = torch.bmm(M[:, pIdx], sourceJoints[:, idx])
        V1 = (newCJoint - newPJoint)[:, 0:3]
        V2 = (targetJoints[:, idx] - targetJoints[:, pIdx])[:, 0:3]

        T = getVectorTranform(newPJoint[:, 0:3], V1, V2)
        #Saving relative transformations
        M_local[:,idx]=T
        #
        M[:, idx] = torch.bmm(T, M[:, pIdx])
        newJoint[:, idx] = torch.bmm(M[:, idx], newJoint[:, idx])

    ######
    reOrder={
        1:4, 2:5, 3:6, 4:7, 5:8, 6:9, 7:10, 8:11, 9:12,
        12:15, 13:16, 14:17,
        16:18, 17:19, 18:20, 19:21, 20:22, 21:23,
        10:24, 11:25, 15:26, 22:27, 23:28
    }
    trueM=torch.eye(4, device = M.device).view(1, 1, 4, 4).repeat(batchSize, 24, 1, 1)
    #Saving relative transformations
    trueM_Local=torch.eye(4, device = M.device).view(1, 1, 4, 4).repeat(batchSize, 24, 1, 1)
    #
    for i in range(24):
        if i == 0:
            continue
        elif i < 9:
            trueM[:,i]=M[:,reOrder[i]]
            trueM_Local[:,i]=M_local[:,reOrder[i]]
        # elif i < 10:#9Bind to parent node
        #     trueM[:,i]=M[:,order[i]]
        else:
            trueM[:,i]=M[:,reOrder[i]]
            trueM_Local[:,i]=M_local[:,reOrder[i]]

    return trueM, newJoint[:, :, 0:3].squeeze(-1), trueM_Local

def AddVirtualJoints(sourceJoints, sourceV, sourceW):
    """
    sourceJoints: [B, 24, 3]
    return: 
    sourceJoints: [B, 29, 3]
    """
    leaf_joints=[10, 11, 15, 22, 23]
    batch_size=sourceJoints.shape[0]

    add_joints=torch.zeros((batch_size, len(leaf_joints), 3), dtype=sourceJoints.dtype, device=sourceJoints.device)
    for i in range(len(leaf_joints)):
        for b in range(batch_size):
            index=torch.nonzero(sourceW[b,:,leaf_joints[i]]).reshape(-1)#
            v=sourceV[b,index]#[n,3]
            new_joint=torch.sum(v, dim=0,keepdim=True)/v.shape[0]
            # new_joint=torch.mean(v,dim=0)
            add_joints[b,i]=new_joint

    return torch.cat((sourceJoints,add_joints), dim=1)

def getVectorTranform_Q(rootV, V1, V2):
    """
    Compute the rotation of a quaternion representation from two vectors
    rootV:[B, 3, 1]
    V1:[B, 3, 1]source
    V2:[B, 3, 1]reference
    return:dQ[B, 4]
    """
    batchSize = rootV.shape[0]
    axis = torch.cross(V2, V1)
    axis /= torch.norm(axis, dim = 1)#[B,3,1]
    axis = axis.squeeze(-1)#[B,3]
    angle = torch.bmm(V1.permute(0, 2, 1), V2).squeeze(-1)/(torch.norm(V1, dim = 1) *  torch.norm(V2, dim = 1))
    angle = torch.acos(angle)#[B,1]

    R = Rotation.from_rotvec(np.array(axis * angle))
    dQ = R.as_quat()
    return torch.tensor(dQ, device=rootV.device)

def Q_mul(Q1, Q2):
    '''
    quadratic product
    Q1,Q2:[B,4]
    '''
    batch_size=Q1.shape[0]

    result = []
    for i in range(batch_size):
        a,b,c,d=Q1[i,3],Q1[i,0],Q1[i,1],Q1[i,2]
        e,f,g,h=Q2[i,3],Q2[i,0],Q2[i,1],Q2[i,2]
        w = a*e - b*f - c*g - d*h
        x = b*e + a*f - d*g + c*h
        y = c*e + d*f + a*g - b*h
        z = d*e - c*f + b*g + a*h
        result.append(np.array([x,y,z,w]))
    
    return torch.tensor(np.array(result), device=Q1.device)#[B,4]

def Q_length(Q):
    '''
    quadratic modulus
    Q:[B,4]
    '''
    return torch.norm(Q, p=2, dim=-1, keepdim=True)#[B,1]


def Q_Rot(Q, v):
    '''
    quaternion rotation
    Q:[B,4]
    v:[B,3]
    '''
    batch_size=Q.shape[0]

    v_Q = torch.cat((v, torch.zeros(batch_size,1,device=v.device)), dim=-1)#[B,4]Convert to quaternion
    Q_star = torch.cat((-Q[:,:3], Q[:,3].unsqueeze(-1)), dim=-1)#conjugate
    Q_inverse = Q_star / (Q_length(Q) * Q_length(Q))#inverse
    
    result = Q_mul(Q_mul(Q, v_Q), Q_inverse)
    return result[:,:3]#[B,3]


def skeletonTransferWithVirtualJoints_Q(sourceJoints, targetJoints):
    """
    Compute the absolute rotation of a quaternion
    sourceJoints: [B, 29, 3]
    targetJoints: [B, 29, 3]
    return: 
    trueM:[B, 24 ,4, 4] input of LBS
    newJoint:[B, 24, 3]
    vQ[B, 24, 4] input of DQS
    vT[B, 24, 3] input of DQS
    """
    order = {
    1:0, 2:0, 3:0, 
    4:1, 5:2, 6:3,
    7:4, 8:5, 9:6,
    10:7, 11:8, 12:9, 13:9, 14:9,
    15:12, 16:13, 17:14,
    18:16, 19:17,
    20:18, 21:19,
    22:20, 23:21,
    24:10, 25:11, 26:15, 27:22, 28:23 }
    batchSize = sourceJoints.shape[0]
    jointNum = sourceJoints.shape[1]

    sourceJoints = torch.cat((sourceJoints, torch.ones(batchSize, jointNum, 1, device = sourceJoints.device)), dim = -1).unsqueeze(-1)
    targetJoints = torch.cat((targetJoints, torch.ones(batchSize, jointNum, 1, device = sourceJoints.device)), dim = -1).unsqueeze(-1)
    M = torch.eye(4, device = sourceJoints.device).view(1, 1, 4, 4).repeat(batchSize, 29, 1, 1)
    #Quaternion rotation, translation
    vQ = torch.tensor([0., 0., 0., 1.], device = sourceJoints.device).view(1, 1, 4).repeat(batchSize, 29, 1)#[B, 29, 4]
    vT = torch.zeros(3, device = sourceJoints.device).view(1, 1, 3).repeat(batchSize, 29, 1)#[B, 29, 3]
    #####
    newJoint = sourceJoints.clone()
    for idx in order:
        pIdx = order[idx]
        newPJoint = torch.bmm(M[:, pIdx], sourceJoints[:, pIdx])
        newCJoint = torch.bmm(M[:, pIdx], sourceJoints[:, idx])
        V1 = (newCJoint - newPJoint)[:, 0:3]
        V2 = (targetJoints[:, idx] - targetJoints[:, pIdx])[:, 0:3]

        T = getVectorTranform(newPJoint[:, 0:3], V1, V2)

        M[:, idx] = torch.bmm(T, M[:, pIdx])
        newJoint[:, idx] = torch.bmm(M[:, idx], newJoint[:, idx])

        #Quaternion rotation, translation
        dQ = getVectorTranform_Q(newPJoint[:, 0:3], V1, V2)#[B,4]
        dT = torch.zeros((batchSize,3), device=sourceJoints.device)#[B,3]
        r = sourceJoints.squeeze(-1)[:,idx,:3]#[B, 3]
        vQ[:,idx] = Q_mul(vQ[:,pIdx], dQ)
        # Is vT wrong to count results with quaternions?
        # vT[:,idx] = vT[:,pIdx] - Q_Rot(vQ[:,idx], r) + Q_Rot(vQ[:,pIdx], (r+dT))
        vT[:,idx] = M[:,idx,:3,3]

    ######
    reOrder={
        1:4, 2:5, 3:6, 4:7, 5:8, 6:9, 7:10, 8:11, 9:12,
        12:15, 13:16, 14:17,
        16:18, 17:19, 18:20, 19:21, 20:22, 21:23,
        10:24, 11:25, 15:26, 22:27, 23:28
    }
    trueM=torch.eye(4, device = M.device).view(1, 1, 4, 4).repeat(batchSize, 24, 1, 1)
    #Quaternion rotation, translation
    true_vQ=torch.tensor([0., 0., 0., 1.], device = sourceJoints.device).view(1, 1, 4).repeat(batchSize, 24, 1)
    true_vT=torch.zeros(3, device = sourceJoints.device).view(1, 1, 3).repeat(batchSize, 24, 1)
    #
    for i in range(24):
        if i == 0:
            continue
        else:
            trueM[:,i]=M[:,reOrder[i]]
            true_vQ[:,i]=vQ[:,reOrder[i]]
            true_vT[:,i]=vT[:,reOrder[i]]
    
    #Where did the math go wrong? The imaginary part differs from the correct result by a plus or minus sign
    true_vQ=torch.cat((-true_vQ[:,:,:3],true_vQ[:,:,3].unsqueeze(-1)),dim=-1)

    return trueM, newJoint[:, :, 0:3].squeeze(-1), true_vQ, true_vT

def AddVirtualJoints_Animal(sourceJoints, sourceV, sourceW):
    """
    sourceJoints: [B, 33, 3]
    sourceV:vertice
    sourceW:rigid weight
    return: 
    sourceJoints: [B, 39, 3]
    """
    leaf_joints=[10, 14, 20, 24, 31, 32]
    batch_size=sourceJoints.shape[0]

    add_joints=torch.zeros((batch_size, len(leaf_joints), 3), dtype=sourceJoints.dtype, device=sourceJoints.device)
    for i in range(len(leaf_joints)):
        for b in range(batch_size):
            index=torch.nonzero(sourceW[b,:,leaf_joints[i]]).reshape(-1)#
            v=sourceV[b,index]#[n,3]
            new_joint=torch.sum(v, dim=0,keepdim=True)/v.shape[0]
            # new_joint=torch.mean(v,dim=0)
            add_joints[b,i]=new_joint

    return torch.cat((sourceJoints,add_joints), dim=1)

def skeletonTransferWithVirtualJoints_Animal(sourceJoints, targetJoints):
    """
    sourceJoints: [B, 39, 3]
    targetJoints: [B, 39, 3]
    return: [B, 33, 4, 4]
    """
    order = {
    1:0, 17:0, 21:0, 25:0, 
    2:1, 3:2, 4:3, 5:4, 6:5,
    7:6, 11:6, 15:6,
    8:7, 9:8, 10:9,
    12:11, 13:12, 14:13,
    16:15, 32:16,
    18:17, 19:18, 20:19,
    22:21, 23:22, 24:23,
    26:25, 27:26, 28:27, 29:28, 30:29, 31:30,
    33:10, 34:14, 35:20, 36:24, 37:31, 38:32 }
    batchSize = sourceJoints.shape[0]
    jointNum = sourceJoints.shape[1]

    sourceJoints = torch.cat((sourceJoints, torch.ones(batchSize, jointNum, 1, device = sourceJoints.device)), dim = -1).unsqueeze(-1)
    targetJoints = torch.cat((targetJoints, torch.ones(batchSize, jointNum, 1, device = sourceJoints.device)), dim = -1).unsqueeze(-1)
    M = torch.eye(4, device = sourceJoints.device).view(1, 1, 4, 4).repeat(batchSize, 39, 1, 1)

    newJoint = sourceJoints.clone()
    for idx in order:
        pIdx = order[idx]
        newPJoint = torch.bmm(M[:, pIdx], sourceJoints[:, pIdx])
        newCJoint = torch.bmm(M[:, pIdx], sourceJoints[:, idx])
        V1 = (newCJoint - newPJoint)[:, 0:3]
        V2 = (targetJoints[:, idx] - targetJoints[:, pIdx])[:, 0:3]

        T = getVectorTranform(newPJoint[:, 0:3], V1, V2)

        M[:, idx] = torch.bmm(T, M[:, pIdx])
        newJoint[:, idx] = torch.bmm(M[:, idx], newJoint[:, idx])

    ######
    reOrder={
        1:2, 2:3, 3:4, 4:5, 5:6, 6:15,
        7:8, 8:9, 9:10, 11:12, 12:13, 13:14, 15:16, 16:32,
        17:18, 18:19, 19:20, 21:22, 22:23, 23:24,
        25:26, 26:27, 27:28, 28:29, 29:30, 30:31,
        10:33, 14:34, 20:35, 24:36, 31:37, 32:38
    }
    trueM=torch.eye(4, device = M.device).view(1, 1, 4, 4).repeat(batchSize, 33, 1, 1)

    for i in range(33):
        if i == 0:
            continue
        else:
            trueM[:,i]=M[:,reOrder[i]]

    return trueM, newJoint[:, :, 0:3].squeeze(-1)