# The code in this document focuses on calculating the mesh 1-ring neighborhood vertex indexes
#
# The returned 1-ring neighborhood index is(point_num, neighbor_num). Since each vertex has a different number of neighbors, the maximum value is used by default and the empty space is replaced by the vertex itself

import numpy as np

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