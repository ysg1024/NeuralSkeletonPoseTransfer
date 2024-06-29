# This document contains mostly code to read the dataset
#
# SMPLRandomDataset：we generates SMPL parameters with a mix of three strategies and uses the smpl_male model for even numbers and the smpl_female model for odd numbers when gender='mixed'
# SMPLTestDataset：Same as SMPLRandomDataset. Just no data enhancement operations such as rotate pan scale
# SMPLTestPairDataset：Using the pre-generated smpl parameter file, the first half is used as source and the second half is used as refer, pairing in order, there are 4 pairings for men and women when gender='mixed'.
# SMALRandomDataset：Randomly generate smal parameters
# SMALTestPairDataset：Randomly generate a pair of smal parameters

import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import h5py
import igl

from datasets.smpl import SMPLLayer
from datasets.smal import SMALLayer

from utils.OneRingIdx import getFacesOneRingIdx
from utils.SmalParameter import generateShape as generateSMALShape
from utils.SmalParameter import generatePose as generateSMALPose
from utils.SmplParameter import generateShape, generatePose

def getRotationByAxis(angle, axis):
    axis = axis/np.linalg.norm(axis)#Remember it has to be a unit vector
    sinA = np.sin(angle)
    cosA = np.cos(angle)
    a,b,c = axis[0], axis[1], axis[2]
    a2 = a*a
    b2 = b*b
    c2 = c*c
    R = np.array([
    [a2 + (1 - a2)*cosA,   a*b*(1 - cosA) + c*sinA, a*c*(1 - cosA) - b*sinA],
    [a*b*(1 - cosA) - c*sinA, b2 + (1 - b2)*cosA, b*c*(1 - cosA) + a*sinA],
    [a*c*(1 - cosA) + b*sinA, b*c*(1 - cosA) - a*sinA, c2 + (1 - c2)*cosA ]
    ])
    return R

def centre(V):
    left = np.min(V, axis = 0)
    right = np.max(V, axis = 0)
    bboxCentre = (left + right) / 2
    V = V - bboxCentre
    return V, bboxCentre

class SMPLRandomDataset(Dataset):
    def __init__(self, fileDir = '.\data\smpl_model', complexity = "all", gender = "mixed", dataSize = 5000, vertexOrderRandom = True, noise = 0, rotate = False, scale = False, translate = False, centre = True):

        if gender == "mixed":
            self.smplLayer = SMPLLayer(model_root = fileDir, gender = "male")
            self.feSmplLayer = SMPLLayer(model_root = fileDir, gender = "female")
        else:
            self.smplLayer = SMPLLayer(model_root = fileDir, gender = gender)
        self.W = self.smplLayer.weights.cpu().numpy()
        self.dataSize = dataSize
        self.faces = np.array(self.smplLayer.faces, dtype = np.int64)
        self.facesOneRingIdx = getFacesOneRingIdx(self.faces)
        self.vertexOrderRandom = vertexOrderRandom
        self.complexity = complexity# Strategies for generating smpl parameters
        self.gender = gender
        self.noise = noise
        self.rotate = rotate
        self.scale = scale
        self.centre = centre
        self.translate = translate
    def __getitem__(self, item):
        shape = generateShape(1, complexity = self.complexity, device = torch.device('cpu'))
        pose = generatePose(1, complexity = self.complexity, device = torch.device('cpu'))

        smplLayer = self.smplLayer
        if self.gender == "mixed" and item%2 == 1:
            smplLayer = self.feSmplLayer

        V, joints = smplLayer.forward(pose, shape)
        joints = joints[0].cpu().numpy()
        V = V[0].cpu().numpy()

        facesOneRingIdx = self.facesOneRingIdx.copy()

        F = self.faces.copy()
        W = self.W.copy()

        if self.vertexOrderRandom:
            randomIdx = np.arange(V.shape[0])
            reRandomIdx = np.arange(V.shape[0])
            np.random.shuffle(randomIdx)
            reRandomIdx[randomIdx] = np.arange(V.shape[0])[:]
            V = V[randomIdx]
            W = W[randomIdx]
            F = reRandomIdx[F]
            facesOneRingIdx = reRandomIdx[facesOneRingIdx]
            facesOneRingIdx = facesOneRingIdx[randomIdx]

        if self.noise > 0:
            N = igl.per_vertex_normals(V, F)
            p = np.random.rand(1)*self.noise
            mask = np.random.binomial(n = 1, p = p, size = V.shape[0]).reshape((-1, 1))
            normRadio = 0.03 * (np.random.rand(V.shape[0]) - 0.5).reshape((-1, 1)) * mask
            V += N * normRadio

        if self.scale:
            S = np.diag(np.random.uniform(low=2./3., high=3./2., size=[3]))
            V = V.dot(S)
            joints = joints.dot(S)

        if self.rotate:
            R = getRotationByAxis(np.random.uniform(-1,1)*2*np.pi, np.random.uniform(-1,1, (3,)))
            V = V.dot(R)
            joints = joints.dot(R)

        if self.translate:
            t = np.random.uniform(low=-0.5, high=0.5, size=[3])
            V += t
            joints += t

        if self.centre:
            t = np.random.uniform(low=-0.5, high=0.5, size=[3])
            V, bboxCentre = centre(V)
            joints -= bboxCentre

        rigW = np.zeros((24, 6890))
        oneHot = W.argmax(axis = 1)
        rigW[oneHot, np.arange(6890)] = 1
        
        return V, facesOneRingIdx, rigW, joints
    
    def __len__(self):
        return self.dataSize


class SMPLTestDataset(Dataset):
    def __init__(self, fileDir = '.\data', mode = "test", complexity = "simple", gender = "male", vertexOrderRandom = True, centre = True):
        
        self.poses = np.load(os.path.join(fileDir, 'poses_%s_%s.npy'%(complexity, mode)))
        self.shapes = np.load(os.path.join(fileDir, 'shapes_%s_%s.npy'%(complexity, mode)))
        self.poses = torch.tensor(self.poses)
        self.shapes = torch.tensor(self.shapes)
        if gender == "mixed":
            self.smplLayer = SMPLLayer(gender = "male")
            self.feSmplLayer = SMPLLayer(gender = "female")
        else:
            self.smplLayer = SMPLLayer(gender = gender)
        self.W = self.smplLayer.weights.cpu().numpy()
        self.faces = np.array(self.smplLayer.faces, dtype = np.int64)
        self.facesOneRingIdx = getFacesOneRingIdx(self.faces)
        self.vertexOrderRandom = vertexOrderRandom
        self.complexity = complexity
        self.gender = gender
        self.centre = centre
        self.dataSize = self.poses.shape[0]

    def __getitem__(self, item):
        shape = generateShape(1, complexity = self.complexity, device = torch.device('cpu'))
        pose = generatePose(1, complexity = self.complexity, device = torch.device('cpu'))

        smplLayer = self.smplLayer
        if self.gender == "mixed" and item%2 == 1:
            smplLayer = self.feSmplLayer

        V, joints = smplLayer.forward(pose, shape)
        joints = joints[0].cpu().numpy()
        V = V[0].cpu().numpy()

        facesOneRingIdx = self.facesOneRingIdx.copy()

        F = self.faces.copy()
        W = self.W.copy()

        if self.vertexOrderRandom:
            randomIdx = np.arange(V.shape[0])
            reRandomIdx = np.arange(V.shape[0])
            np.random.shuffle(randomIdx)
            reRandomIdx[randomIdx] = np.arange(V.shape[0])[:]
            V = V[randomIdx]
            W = W[randomIdx]
            F = reRandomIdx[F]
            facesOneRingIdx = reRandomIdx[facesOneRingIdx]
            facesOneRingIdx = facesOneRingIdx[randomIdx]

        if self.centre:
            V, bboxCentre = centre(V)
            joints -= bboxCentre

        rigW = np.zeros((24, 6890))
        oneHot = W.argmax(axis = 1)
        rigW[oneHot, np.arange(6890)] = 1
        
        return V, facesOneRingIdx, rigW, joints
    
    def __len__(self):
        return self.dataSize

class SMPLTestPairDataset(Dataset):
    def __init__(self, fileDir = '.\data', complexity = 'simple', mode = "test", gender = 'male', vertexOrderRandom = False):

        self.vertexOrderRandom = vertexOrderRandom
        self.poses = np.load(os.path.join(fileDir, 'poses_%s_%s.npy'%(complexity, mode)))
        self.shapes = np.load(os.path.join(fileDir, 'shapes_%s_%s.npy'%(complexity, mode)))
        self.poses = torch.tensor(self.poses)
        self.shapes = torch.tensor(self.shapes)
        self.gender = gender
        if self.gender == "mixed":
            self.smplLayer = SMPLLayer(model_root = os.path.join(fileDir, "smpl_model"), gender = "male")
            self.feSmplLayer = SMPLLayer(model_root = os.path.join(fileDir, "smpl_model"), gender = "female")
        else:
            self.smplLayer = SMPLLayer(model_root = os.path.join(fileDir, "smpl_model"), gender = gender)
        self.faces = np.array(self.smplLayer.faces, dtype = np.int64)

        self.dataSize = self.poses.shape[0] // 2

    def __getitem__(self, item):
        sPose = self.poses[item].unsqueeze(0)
        sShape = self.shapes[item].unsqueeze(0)

        rPose = self.poses[item + self.dataSize].unsqueeze(0)
        rShape = self.shapes[item + self.dataSize].unsqueeze(0)

        if self.gender == "mixed":
            mixedMode = item % 4 
            if mixedMode  == 0:
                sV, _ = self.smplLayer(sPose, sShape)
                rV, _ = self.feSmplLayer(rPose, rShape)
                tV, _ = self.smplLayer(rPose, sShape)
            elif mixedMode  == 1:
                sV, _ = self.feSmplLayer(sPose, sShape)
                rV, _ = self.smplLayer(rPose, rShape)
                tV, _ = self.feSmplLayer(rPose, sShape)
            elif mixedMode  == 2:
                sV, _ = self.feSmplLayer(sPose, sShape)
                rV, _ = self.feSmplLayer(rPose, rShape)
                tV, _ = self.feSmplLayer(rPose, sShape)
            elif mixedMode  == 3:
                sV, _ = self.smplLayer(sPose, sShape)
                rV, _ = self.smplLayer(rPose, rShape)
                tV, _ = self.smplLayer(rPose, sShape)
        else:
            sV, _ = self.smplLayer(sPose, sShape)
            rV, _ = self.smplLayer(rPose, rShape)
            tV, _ = self.smplLayer(rPose, sShape)

        sV = sV[0].cpu().numpy()
        rV = rV[0].cpu().numpy()
        tV = tV[0].cpu().numpy()
        F = self.faces.copy()


        if self.vertexOrderRandom:
            randomIdx = np.arange(sV.shape[0])
            reRandomIdx = np.arange(sV.shape[0])
            np.random.shuffle(randomIdx)
            reRandomIdx[randomIdx] = np.arange(sV.shape[0])[:]
            sV = sV[randomIdx]
            tV = tV[randomIdx]
            F = reRandomIdx[F]
            
            randomIdx = np.arange(sV.shape[0])
            np.random.shuffle(randomIdx)
            rV = rV[randomIdx]       

        return sV, rV, tV, F

    def __len__(self):
        return self.dataSize

class SMALRandomDataset(Dataset):
    def __init__(self, fileDir = '.\data\smal_model', dataSize = 5000, vertexOrderRandom = True):

        self.smalLayer = SMALLayer(model_root = fileDir)
        self.W = self.smalLayer.weights.cpu().numpy()
        self.dataSize = dataSize
        self.faces = np.array(self.smalLayer.faces, dtype = np.int64)
        self.facesOneRingIdx = getFacesOneRingIdx(self.faces)
        self.vertexOrderRandom = vertexOrderRandom
        smal_data = h5py.File(os.path.join(fileDir, 'smal_data.h5'), "r")
        toys_betas=torch.tensor(np.array(smal_data['toys_betas']), dtype=torch.float32)
        smal_data.close()
        self.toys_betas = toys_betas
    def __getitem__(self, item):
        shape = generateSMALShape(1, self.toys_betas)
        pose = generateSMALPose(1)

        V, joints = self.smalLayer.forward(pose, shape)
        joints = joints[0].cpu().numpy()
        V = V[0].cpu().numpy()

        facesOneRingIdx = self.facesOneRingIdx.copy()

        F = self.faces.copy()
        W = self.W.copy()
        if self.vertexOrderRandom:
            randomIdx = np.arange(V.shape[0])
            reRandomIdx = np.arange(V.shape[0])
            np.random.shuffle(randomIdx)
            reRandomIdx[randomIdx] = np.arange(V.shape[0])[:]
            V = V[randomIdx]
            W = W[randomIdx]
            F = reRandomIdx[F]
            facesOneRingIdx = reRandomIdx[facesOneRingIdx]
            facesOneRingIdx = facesOneRingIdx[randomIdx]

        rigW = np.zeros((33, 3889))
        oneHot = W.argmax(axis = 1)
        rigW[oneHot, np.arange(3889)] = 1
        
        return V, facesOneRingIdx, rigW, joints
    
    def __len__(self):
        return self.dataSize

class SMALTestPairDataset(Dataset):
    def __init__(self, fileDir = '.\data\smal_model', dataSize = 500, vertexOrderRandom = False):

        self.smalLayer = SMALLayer(model_root = fileDir)
        self.W = self.smalLayer.weights.cpu().numpy()
        self.dataSize = dataSize
        self.faces = np.array(self.smalLayer.faces, dtype = np.int64)
        self.facesOneRingIdx = getFacesOneRingIdx(self.faces)
        self.vertexOrderRandom = vertexOrderRandom
        smal_data = h5py.File(os.path.join(fileDir, 'smal_data.h5'), "r")
        toys_betas=torch.tensor(np.array(smal_data['toys_betas']), dtype=torch.float32)
        smal_data.close()
        self.toys_betas = toys_betas

    def __getitem__(self, item):
        shapes = generateSMALShape(2, self.toys_betas)
        poses = generateSMALPose(2)

        sV, sJ = self.smalLayer.forward(poses[0].unsqueeze(0), shapes[0].unsqueeze(0))
        rV, rJ = self.smalLayer.forward(poses[1].unsqueeze(0), shapes[1].unsqueeze(0))
        tV, tJ = self.smalLayer.forward(poses[1].unsqueeze(0), shapes[0].unsqueeze(0))

        sV = sV[0].cpu().numpy()
        rV = rV[0].cpu().numpy()
        tV = tV[0].cpu().numpy()

        
        F = self.faces.copy()

        if self.vertexOrderRandom:
            randomIdx = np.arange(sV.shape[0])
            reRandomIdx = np.arange(sV.shape[0])
            np.random.shuffle(randomIdx)
            reRandomIdx[randomIdx] = np.arange(sV.shape[0])[:]
            sV = sV[randomIdx]
            tV = tV[randomIdx]
            F = reRandomIdx[F]
            
            randomIdx = np.arange(sV.shape[0])
            np.random.shuffle(randomIdx)
            rV = rV[randomIdx]

        return sV, rV, tV, F

    def __len__(self):
        return self.dataSize


class SMPLTestDataset_test_W_C(Dataset):
    '''
    Used to test the weights and joints of neural rigging, the SMPL model at the time of testing does not have to do any operations to upset the vertices or data augmentation, to ensure that the order of vertices in the outputs of gt and the network is consistent
    '''
    def __init__(self, fileDir = '.\data', mode = "test", complexity = "simple", gender = "male", vertexOrderRandom = False, centre = False):
        
        self.poses = np.load(os.path.join(fileDir, 'poses_%s_%s.npy'%(complexity, mode)))
        self.poses = torch.tensor(self.poses)
        self.shapes = np.load(os.path.join(fileDir, 'shapes_%s_%s.npy'%(complexity, mode)))
        self.shapes = torch.tensor(self.shapes)
        if gender == "mixed":
            self.smplLayer = SMPLLayer(model_root = os.path.join(fileDir, 'smpl_model'), gender = "male")
            self.feSmplLayer = SMPLLayer(model_root = os.path.join(fileDir, 'smpl_model'), gender = "female")
        else:
            self.smplLayer = SMPLLayer(model_root = os.path.join(fileDir, 'smpl_model'), gender = gender)
        self.W = self.smplLayer.weights.cpu().numpy()
        self.faces = np.array(self.smplLayer.faces, dtype = np.int64)
        self.facesOneRingIdx = getFacesOneRingIdx(self.faces)
        self.vertexOrderRandom = vertexOrderRandom
        self.complexity = complexity
        self.gender = gender
        self.centre = centre
        self.dataSize = self.poses.shape[0]

    def __getitem__(self, item):
        shape = self.shapes[item].unsqueeze(0)
        pose = self.poses[item].unsqueeze(0)

        smplLayer = self.smplLayer
        if self.gender == "mixed" and item%2 == 1:
            smplLayer = self.feSmplLayer

        V, joints = smplLayer.forward(pose, shape)

        gtV = V.squeeze(0) #torch.Size([6890, 3])
        gtF = torch.tensor(np.array(self.smplLayer.faces,dtype=np.int64)) #(13776, 3)
        gtW = self.smplLayer.weights #torch.Size([6890, 24])
        gtC = joints.squeeze(0) #torch.Size([1, 24, 3])

        return gtV, gtF, gtW, gtC
    
    def __len__(self):
        return self.dataSize

class SMPLTestDataset_test_T_pose(Dataset):
    '''
    T-pose only
    Used to test the weights and joints of neural rigging, the SMPL model at the time of testing does not have to do any operations to upset the vertices or data augmentation, to ensure that the order of vertices in the outputs of gt and the network is consistent
    '''
    def __init__(self, fileDir = '.\data', mode = "test", complexity = "simple", gender = "male", vertexOrderRandom = False, centre = False):
        
        self.poses = np.load(os.path.join(fileDir, 'poses_%s_%s.npy'%(complexity, mode)))
        self.poses = torch.tensor(self.poses)
        self.shapes = np.load(os.path.join(fileDir, 'shapes_%s_%s.npy'%(complexity, mode)))
        self.shapes = torch.tensor(self.shapes)
        if gender == "mixed":
            self.smplLayer = SMPLLayer(model_root = os.path.join(fileDir, 'smpl_model'), gender = "male")
            self.feSmplLayer = SMPLLayer(model_root = os.path.join(fileDir, 'smpl_model'), gender = "female")
        else:
            self.smplLayer = SMPLLayer(model_root = os.path.join(fileDir, 'smpl_model'), gender = gender)
        self.W = self.smplLayer.weights.cpu().numpy()
        self.faces = np.array(self.smplLayer.faces, dtype = np.int64)
        self.facesOneRingIdx = getFacesOneRingIdx(self.faces)
        self.vertexOrderRandom = vertexOrderRandom
        self.complexity = complexity
        self.gender = gender
        self.centre = centre
        self.dataSize = self.poses.shape[0]

    def __getitem__(self, item):
        shape = self.shapes[item].unsqueeze(0)
        pose = self.poses[item].unsqueeze(0)
        pose = torch.zeros_like(pose)

        smplLayer = self.smplLayer
        if self.gender == "mixed" and item%2 == 1:
            smplLayer = self.feSmplLayer

        V, joints = smplLayer.forward(pose, shape)

        gtV = V.squeeze(0) #torch.Size([6890, 3])
        gtF = torch.tensor(np.array(self.smplLayer.faces,dtype=np.int64)) #(13776, 3)
        gtW = self.smplLayer.weights #torch.Size([6890, 24])
        gtC = joints.squeeze(0) #torch.Size([1, 24, 3])

        return gtV, gtF, gtW, gtC
    
    def __len__(self):
        return self.dataSize