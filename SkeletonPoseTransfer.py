import torch
import torch.nn.functional as F
import numpy as np

def corX_normaliztion(f,v1,v2,v3):
    '''
    f:[B,3,1]
    v1 = c1 - f
    v2 = c2 - f
    v3 = c3 - f
    return:[B,3,4]corX four points
    '''
    # v1 = v1/torch.norm(v1, dim=1, keepdim=True)
    # v2 = v2/torch.norm(v2, dim=1, keepdim=True)
    # v3 = v3/torch.norm(v3, dim=1, keepdim=True)
    c1_n = f + v1
    c2_n = f + v2
    c3_n = f + v3
    return torch.cat((f, c1_n, c2_n, c3_n), dim=2)


def batch_get_3children_orient_svd(corA, corB):
    '''
    corA, corB:Four points form a coordinate system[B,3,4]
    return:[B,3,3]rotation matrix
    '''
    #1. Move the center of mass of both coordinate systems to the same point
    mat_A = corA - torch.mean(corA, dim=2, keepdim=True)
    mat_B = corB - torch.mean(corB, dim=2, keepdim=True)
    #2. svd decomposition
    S = mat_A.bmm(mat_B.transpose(1, 2))
    U, _, V = torch.svd(S)
    #3. Calculate the rotation matrix to prevent a reflection matrix
    det_u_v = torch.det(torch.bmm(V, U.transpose(1, 2)))
    det_modify_mat = torch.eye(3, device=U.device).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
    det_modify_mat[:, 2, 2] = det_u_v
    rot_mat = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))

    assert torch.sum(torch.isnan(rot_mat)) == 0, ('3children rot_mat', rot_mat)
    return rot_mat

def rodrigues(v1, v2):
    '''
    Rotation from v1 to v2
    v1, v2:vector[B,3,1]
    return:[B,3,3]rotation matrix
    '''
    v1 = v1/torch.norm(v1, dim=1, keepdim=True)
    v2 = v2/torch.norm(v2, dim=1, keepdim=True)
    axis = torch.cross(v1, v2)
    axis = axis/torch.norm(axis, dim=1, keepdim=True)
    #angle[B,1]
    angle = torch.bmm(v1.permute(0, 2, 1), v2).squeeze(-1)/(torch.norm(v1, dim = 1) *  torch.norm(v2, dim = 1))
    angle = torch.acos(angle)

    cos = torch.cos(angle).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(-1)
    eyes = torch.eye(3, device = axis.device).unsqueeze(0).repeat(axis.shape[0], 1, 1)
    R = torch.zeros(axis.shape[0], 3, 3, device = axis.device)
    R[:, 0, 1] = -axis[:, 2, 0]
    R[:, 0, 2] = axis[:, 1, 0]
    R[:, 1, 0] = -R[:, 0, 1]
    R[:, 1, 2] = -axis[:, 0, 0]
    R[:, 2, 0] = -R[:, 0, 2]
    R[:, 2, 1] = -R[:, 1, 2]

    R = cos * eyes + (1 - cos) * torch.bmm(axis, axis.permute(0, 2, 1)) + sin * R
    return R

def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def SkeletonPoseTransfer(sourceJoints, targetJoints):
    """
    sourceJoints: [B, 29, 3]source joints
    targetJoints: [B, 29, 3]refer joints
    return: [B, 24, 4, 4]rotation matrix
    """
    batch_size = sourceJoints.shape[0]

    parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
        16, 17, 18, 19, 20, 21, 10, 11, 15, 22, 23]
    children = [[1,2,3],  [4],  [5],  [6],  [7],  [8],  [9], [10], [11], [12,13,14], [24], [25], [15], [16], [17], [26], [18], [19],
        [20], [21], [22], [23], [27], [28], [], [], [], [], []]

    
    #1.get R_cube, need axis_angle per joints
    source_vector = sourceJoints.clone()
    source_vector[:,1:] = source_vector[:,1:] - source_vector[:,parents[1:]]

    target_vector = targetJoints.clone()
    target_vector[:,1:] = target_vector[:,1:] - target_vector[:,parents[1:]]
    target_vector[:,0] = source_vector[:,0]

    source_vector = torch.unsqueeze(source_vector, dim=-1)#[B,29,3,1]
    target_vector = torch.unsqueeze(target_vector, dim=-1)#[B,29,3,1]

    source_joints = sourceJoints.clone()
    target_joints = targetJoints - targetJoints[:,0:1] + sourceJoints[:,0:1]
    source_joints = torch.unsqueeze(source_joints, dim=-1)#[B,29,3,1]
    target_joints = torch.unsqueeze(target_joints, dim=-1)#[B,29,3,1]

    R = []
    for i in range(24):
        if len(children[i]) == 3:
            #svd
            corA = corX_normaliztion(
                source_joints[:,i],
                source_vector[:,children[i][0]],
                source_vector[:,children[i][1]],
                source_vector[:,children[i][2]],
            )
            corB = corX_normaliztion(
                target_joints[:,i],
                target_vector[:,children[i][0]],
                target_vector[:,children[i][1]],
                target_vector[:,children[i][2]],
            )
            Ri = batch_get_3children_orient_svd(corA, corB)
            R.append(Ri)
        elif len(children[i]) == 1:
            #rodrigues
            Ri = rodrigues(
                source_vector[:,children[i][0]],
                target_vector[:,children[i][0]],
            )
            R.append(Ri)
        elif len(children[i]) == 0:
            continue
        else:
            print('something error!!!')
    R = torch.stack(R, dim=1)#[B,24,3,3]
    J = source_joints[:,:24]#[B,24,3,1]
    #2.get T, need R_cube & source_j
    T = []

    for i in range(24):
        if i == 0:
            Ji = J[:,i]
            ti = Ji - torch.matmul(R[:,i], J[:,i])
            T.append(
                transform_mat(R[:,i], ti)
            )
        else:
            Ji = torch.matmul(T[parents[i]][:,:3,:3], J[:,i]) + T[parents[i]][:,:3,3:]
            ti = Ji - torch.matmul(R[:,i], J[:,i])
            T.append(
                transform_mat(R[:,i], ti)
            )
    T = torch.stack(T, dim=1)#[B,24,4,4]

    return T