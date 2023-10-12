import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import igl
from scipy.spatial.transform import Rotation

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

    V = torch.matmul(T, V)    #[B,N,M,4,1]
    V = W * V.squeeze(-1) #[B,N,M,4]
    V = V.sum(dim = 2,keepdim = False)#[B,N,4]
    return V[:,:,:3]#[B,N,3]

def R2Q(R):
    '''
    R:[3,3]
    return:[4]
    '''
    r = Rotation.from_matrix(R)
    Q = r.as_quat()
    theta = 2 * np.arccos(Q[3])
    # print('theta:',theta)
    if theta < np.pi:
        return Q
    else :
        return -Q


def get_vQ_vT(M):
    '''
    M:[M,4,4]
    return: vQ(M,4)
            vT(M,3)
    '''
    Q = np.array(M[:,:3,:3])#[M,3,3]
    T = np.array(M[:,:3,3])#[M,3]

    vQ = []
    for i in range(Q.shape[0]):
        vQ.append(R2Q(Q[i]))
    vQ = np.array(vQ)

    vT = T
    return vQ,vT


def DQS_numpy(V, W, T):
    """
    V:ndarray[N,3]顶点
    W:ndarray[N,M]权重
    T:ndarray[M,4,4]变换矩阵

    return: ndarray[N, 3]
    """
    vQ, vT = get_vQ_vT(T)

    V = np.array(V, dtype = np.float64)
    W = np.array(W, dtype = np.float64)
    vQ = np.array(vQ, dtype = np.float64)
    vT = np.array(vT, dtype = np.float64)

    result = igl.dqs(V, W, vQ, vT)
    return result

def dqs(V, W, T):
    """
    V:tensor[B,N,3]顶点
    W:tensor[B,N,M]权重
    T:tensor[B,M,4,4]变换矩阵

    return: tensor[B, N, 3]
    """
    B = V.shape[0]
    result = V.clone()

    V = V.cpu().numpy()
    W = W.cpu().numpy()
    T = T.cpu().numpy()

    for i in range(B):
        transfer_matrix = T[i]
        vQ, vT = get_vQ_vT(transfer_matrix)

        vertex = np.array(V[i], dtype = np.float64)
        weight = np.array(W[i], dtype = np.float64)
        vQ = np.array(vQ, dtype = np.float64)
        vT = np.array(vT, dtype = np.float64)

        result[i] = torch.tensor(igl.dqs(vertex, weight, vQ, vT))
    
    return result


def getLaplacianMatrix(V, F, normalize = True, weight = "cotangent"):
    batchSize = V.shape[0]
    pointNum = V.shape[1]
    device = V.device
    V = V.detach().cpu().numpy()
    F = F.detach().cpu().numpy()
    L = np.zeros(shape = (batchSize, pointNum, pointNum), dtype = np.float32)
    
    for i in range(batchSize):
        if weight == "cotangent":
            L[i, :, :] = igl.cotmatrix(V[i], F[i]).todense()
        elif weight == "uniform":
            A = np.array(igl.adjacency_matrix(F[i]).todense())
            A[np.arange(A.shape[0]), np.arange(A.shape[0])] = -A.sum(axis = 1)
            L[i, :, :] = A
            
    L = torch.tensor(L).float().to(device)
    if normalize:
        L = L / L[:, np.arange(pointNum), np.arange(pointNum)].view(batchSize, pointNum, 1)
    return L

def getBetaMatrix(L, DLam = 20):
    """
    计算平滑矩阵
    L:拉普拉斯矩阵, [B, N, N]
    D:lambda矩阵, [B, N, N]
    p:平滑迭代次数
    return: Beta矩阵, [B, N, N]
    """

    batchSize = L.shape[0]
    pointsNum = L.shape[1]
    device = L.device
    eyes = torch.eye(pointsNum).unsqueeze(0).repeat(batchSize, 1, 1).to(device)
    B = eyes + DLam*L#torch.bmm(D, L)
    B = torch.linalg.inv(B) 
    return B

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

def getFacesOneRingIdx(F, maxNum = None):
    pointNum = F.max() + 1
    e1, e2, e3 = F[:, 0], F[:, 1], F[:, 2]
    adjMatrix = np.zeros((pointNum, pointNum))
    adjMatrix[e1, e2] = True
    adjMatrix[e1, e3] = True
    adjMatrix[e2, e3] = True
    adjMatrix[e2, e1] = True
    adjMatrix[e3, e1] = True
    adjMatrix[e3, e2] = True
    if maxNum == None:
        maxNum = int(adjMatrix.sum(0).max() + 1)
    
    oneRingIdx = np.zeros((pointNum, maxNum), dtype = np.int32)
    oneRingIdx[:] = np.arange(pointNum).reshape((pointNum, 1))
    for i, adjIdx in enumerate(adjMatrix):
        oneRingIdx[i][-int(adjIdx.sum()):] = np.where(adjIdx == 1)[0]
    return oneRingIdx

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
        recLoss = self.mesLoss(preV, tV)
        edgLoss = edgesLengthRadio(preV, tV, F)
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





