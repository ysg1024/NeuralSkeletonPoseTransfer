import torch
import numpy as np
import igl
from scipy.spatial.transform import Rotation

def LBS(V, W, T):
    """
    V:[B,N,3] vertice
    W:[B,N,M] weights
    T:[B,M,4,4] rotation matrix

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

# def R2Q(R):
#     '''
#     R:[3,3]
#     return:[4]
#     '''
#     r = Rotation.from_matrix(R)
#     Q = r.as_quat()
#     theta = 2 * np.arccos(Q[3])
#     print('theta:',theta)
#     if theta < np.pi:
#         return Q
#     else :
#         return -Q


# def get_vQ_vT(M):
#     '''
#     M:[M,4,4]
#     return: vQ(M,4)
#             vT(M,3)
#     '''
#     Q = np.array(M[:,:3,:3])#[M,3,3]
#     T = np.array(M[:,:3,3])#[M,3]

#     vQ = []
#     vT = []
#     for i in range(Q.shape[0]):
#         vQ.append(R2Q(Q[i]))
#         vT.append(T[i])

#     vQ = np.array(vQ)
#     vT = np.array(vT)

#     # vT = T
#     return vQ,vT

def DQS(V, W, vQ, vT):
    """
    V:ndarray[N, 3] vertice
    W:ndarray[N, M] weights
    vQ:ndarray[M, 4] rotation
    vT:ndarray[M, 3] translation

    return: ndarray[N, 3]
    """
    V = np.array(V, dtype = np.float64)
    W = np.array(W, dtype = np.float64)
    vQ = np.array(vQ, dtype = np.float64)
    vT = np.array(vT, dtype = np.float64)

    result = igl.dqs(V, W, vQ, vT)
    return result

def DQS_pytorch(V, W, vQ, vT):
    """
    V:tensor[B, N, 3] vertice
    W:tensor[B, N, M] weights
    vQ:ndarray[B, M, 4] rotation
    vT:ndarray[B, M, 3] translation

    return: tensor[B, N, 3]
    """
    B = V.shape[0]
    result = V.clone()

    V = V.cpu().numpy()
    W = W.cpu().numpy()
    vQ = vQ.cpu().numpy()
    vT = vT.cpu().numpy()

    for i in range(B):
        # transfer_matrix = T[i]
        # vQ, vT = get_vQ_vT(transfer_matrix)

        vertex = np.array(V[i], dtype = np.float64)
        weight = np.array(W[i], dtype = np.float64)
        vQ_i = np.array(vQ[i], dtype = np.float64)
        vT_i = np.array(vT[i], dtype = np.float64)

        result[i] = torch.tensor(igl.dqs(vertex, weight, vQ_i, vT_i))
    
    return result




# def igl_fk(J,M_local):
#     #BE:[24,2]
#     bones={1:0, 2:0, 3:0, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 10:7, 11:8, 12:9, 13:9,
#             14:9, 15:12, 16:13, 17:14, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21}#joint hierarchy{child:father}
#     BE=[]
#     BE.append(np.array([0,0]))
#     for i in range(1,24):
#         BE.append(np.array([bones[i],i]))
#     BE=np.array(BE)
#     #P:[24]
#     P=np.array([[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]])
#     #dQ:[24,3,3],dT[24,3]
#     M_local=np.array(M_local)
#     dQ,dT=returnList(M_local)

#     J=np.array(J,dtype=np.float64,order='C')
#     dQ=np.array(dQ,dtype=np.float64)
#     dT=np.array(dT,dtype=np.float64)
    
#     vQ,vT=igl.forward_kinematics(J,BE,P,dQ,dT)

#     # M=np.zeros((24,4,4))
#     # print(vQ.shape)
#     # print(vT.shape)
#     # M[:,:3,:3]=vQ
#     # M[:,:3,3]=vT
#     return vQ,vT

# def igl_dqs(J,M_local,V,W):
#     vQ,vT=igl_fk(J,M_local)

#     V=np.array(V,dtype=np.float64)
#     W=np.array(W,dtype=np.float64)

#     newV=igl.dqs(V,W,vQ,vT)
#     return newV