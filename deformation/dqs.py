from traceback import print_tb
import numpy as np
from scipy.spatial.transform import Rotation

def rotationToQuaternion(R):
    w,x,y,z = 0,0,0,0
    
    if np.trace(R) > 0:
        w = np.sqrt(1 + np.trace(R))/2
        x = (R[2,1] - R[1,2]) / (4*w)
        y = (R[0,2] - R[2,0]) / (4*w)
        z = (R[1,0] - R[0,1]) / (4*w)
    else:
        diagR = np.diag(R)
        if np.max(diagR) == 0:
            t = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / (2*t)
            x = t/2
            y = (R[1,0] + R[0,1]) / (2*t)
            z = (R[2,0] + R[0,2]) / (2*t)
        elif np.max(diagR) == 1:
            t = np.sqrt(1 - R[0,0] + R[1,1] - R[2,2])
            w = (R[0,2] - R[2,0]) / (2*t)
            x = (R[1,0] + R[0,1]) / (2*t)
            y = t/2
            z = (R[2,1] + R[1,2]) / (2*t)
        else:
            t = np.sqrt(1 - R[0,0] - R[1,1] + R[2,2])
            w = (R[1,0] - R[0,1]) / (2*t)
            x = (R[0,2] + R[2,0]) / (2*t)
            y = (R[1,2] + R[2,1]) / (2*t)
            z = t/2
    Qr = np.array([w, x, y, z])
    return Qr

def translationToQuaternion(t):
    Qt = np.array([0, t[0]/2 , t[1]/2 , t[2]/2])
    return Qt

def trasformationToQuaternion(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    Qr = rotationToQuaternion(R)
    Qt = translationToQuaternion(t)
    w,x,y,z = Qr
    u,t0,t1,t2 = Qt
    
    Q = np.zeros(shape = (8,))
    Q[0:4] = Qr
    #Q[4:] = [-x*t0-y*t1-z*t2, w*t0 + y*t2 - z*t1, w*t1 - x*t2 + z*t0, w*t2 + x*t1 - y*t0]
    Q[4:] = np.array([-t0*x-t1*y-t2*z, t0*w+t1*z-t2*y, -t0*z+t1*w+t2*x, t0*y-t1*x+t2*w])
    return Q
def nomalizeQuaternion(Q):
    Q = Q/np.linalg.norm(Q[0:4])
    return Q
def deformeVertexWithQuaternion(v, Q):
    # r=Rotation.from_quat(Q)
    # r.as_matrix()
    c = Q[0:4]
    ce = Q[4:]
    a = c[0]
    ae = ce[0]
    d = c[1:]
    de = ce[1:]

    v = v + 2*np.cross(d,(np.cross(d, v) + a*v)) + 2*(a*de - ae*d + np.cross(d, de))
    return v

def dqs(V, W, M):
    V = V[0].cpu().numpy()
    W = W.permute(0, 2, 1)[0].cpu().numpy()
    Ms = M[0].cpu().numpy()
    newPoseV = V.copy()
    Qs = []
    for j in range(len(Ms)):
        M = Ms[j]
        Q = trasformationToQuaternion(M)
        Qs.append(Q)
    # for i in range(len(Qs)):
    #     temp=np.zeros(4)
    #     temp[0]=Qs[i][1]
    #     temp[1]=Qs[i][2]
    #     temp[2]=Qs[i][3]
    #     temp[3]=Qs[i][0]
    #     print(temp)
    # Qs[19]=-Qs[19]
    
    for i in range(newPoseV.shape[0]):
        Q = np.zeros(shape = (8,))
        for j in range(len(Qs)):
            Q += W[j, i]*Qs[j]
        Q = nomalizeQuaternion(Q)

        newPoseV[i] = deformeVertexWithQuaternion(V[i], Q)
    return newPoseV