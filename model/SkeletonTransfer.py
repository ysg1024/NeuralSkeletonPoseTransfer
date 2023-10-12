
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def skeletonTransfer(sourceJoints, targetJoints):
    """
    sourceJoints: [B, 24, 3]
    targetJoints: [B, 24, 3]
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
    22:20, 23:21 }
    batchSize = sourceJoints.shape[0]
    jointNum = sourceJoints.shape[1]

    sourceJoints = torch.cat((sourceJoints, torch.ones(batchSize, jointNum, 1, device = sourceJoints.device)), dim = -1).unsqueeze(-1)
    targetJoints = torch.cat((targetJoints, torch.ones(batchSize, jointNum, 1, device = sourceJoints.device)), dim = -1).unsqueeze(-1)
    M = torch.eye(4, device = sourceJoints.device).view(1, 1, 4, 4).repeat(batchSize, 24, 1, 1)
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
    #前面的都没改，下面是我添加的部分，把错位的变换矩阵修正，具体来说就是父关节的变换矩阵应该是其子关节的变换矩阵
    ######
    reOrder={# 与前面order相反，根据父关节查找子关节
        1:4, 2:5, 3:6, 4:7, 5:8, 6:9, 7:10, 8:11,
        12:15, 13:16, 14:17,
        16:18, 17:19, 18:20, 19:21, 20:22, 21:23
    }
    trueM=torch.eye(4, device = M.device).view(1, 1, 4, 4).repeat(batchSize, 24, 1, 1)
    for i in range(24):
        if i == 0:
            continue
        elif i < 9:
            trueM[:,i]=M[:,reOrder[i]]
        elif i < 12:#手脚还有头5个末关节点的变换矩阵与他们的父关节保持一致
            trueM[:,i]=M[:,order[i]]
        elif i < 15:
            trueM[:,i]=M[:,reOrder[i]]
        elif i < 16:#手脚还有头5个末关节点的变换矩阵与他们的父关节保持一致
            trueM[:,i]=M[:,order[i]]
        elif i < 22:
            trueM[:,i]=M[:,reOrder[i]]
        else:#手脚还有头5个末关节点的变换矩阵与他们的父关节保持一致
            trueM[:,i]=M[:,order[i]]

    return trueM, newJoint[:, :, 0:3].squeeze(-1)

def AddVirtualJoints(sourceJoints, sourceV, sourceW):
    """
    sourceJoints: [B, 24, 3]，24个关节点
    sourceV: [B, v, 3]，模型顶点坐标
    sourceW: [B, v, 24]，刚性权重
    return: 
    sourceJoints: [B, 29, 3]，添加虚拟关节点后的29个关节点
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

def skeletonTransferWithVirtualJoints(sourceJoints, targetJoints):
    """
    这个是添加虚拟关节点后的骨骼姿态迁移，改善了末关节点的朝向问题
    注意输入的关节点需要加上5个我们添加的虚拟关节点，直接调用上面的AddVirtualJoints()就可以得到符合格式的输入
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
    #
    ######
    reOrder={
        1:4, 2:5, 3:6, 4:7, 5:8, 6:9, 7:10, 8:11,
        12:15, 13:16, 14:17,
        16:18, 17:19, 18:20, 19:21, 20:22, 21:23,
        10:24, 11:25, 15:26, 22:27, 23:28
    }
    trueM=torch.eye(4, device = M.device).view(1, 1, 4, 4).repeat(batchSize, 24, 1, 1)
    for i in range(24):
        if i == 0:
            continue
        elif i < 9:
            trueM[:,i]=M[:,reOrder[i]]
        elif i < 10:#9绑定到父节点
            trueM[:,i]=M[:,order[i]]
        else:
            trueM[:,i]=M[:,reOrder[i]]

    return trueM, newJoint[:, :, 0:3].squeeze(-1)

def AddVirtualJoints_Animal(sourceJoints, sourceV, sourceW):
    """
    sourceJoints: [B, 33, 3]
    sourceV:顶点
    sourceW:刚性权重
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