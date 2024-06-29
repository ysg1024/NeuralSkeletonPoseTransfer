import torch

def joint2offset(joint, use_smpl = True):
    '''
    joint:tensor[B,J,3]
    '''
    if use_smpl:
        #SMPL
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    else:
    #SMAL
        parents = [-1,          0,          1,          2,          3,
                    4,          5,          6,          7,          8,
                    9,          6,         11,         12,         13,
                    6,         15,          0,         17,         18,
                    19,          0,         21,         22,         23,
                    0,         25,         26,         27,         28,
                    29,         30,         16]
    offset = torch.zeros((joint.shape[0], joint.shape[1], 3)) #(B,J,3)

    for i in range(joint.shape[1]):
        if i != 0:
            offset[:,i,:] = joint[:,i,:] - joint[:,parents[i],:]
        else:
            offset[:,i,:] = joint[:,i,:]

    return offset