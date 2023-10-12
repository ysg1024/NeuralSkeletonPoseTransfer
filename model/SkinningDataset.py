
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from .smpl import SMPLLayer
import os
from .utils import getFacesOneRingIdx
from .smpl_parameter_generation import generatePose, generateShape

class SMPLRandomDataset(Dataset):
    def __init__(self, fileDir = '.\data\smpl_model', complexity = "skinning", gender = "neutral", dataSize = 5000, vertexOrderRandom = True):

        if gender == "mixd":
            self.smplLayer = SMPLLayer(model_root = os.path.join(fileDir, "smpl_model"), gender = "male")
            self.feSmplLayer = SMPLLayer(model_root = os.path.join(fileDir, "smpl_model"), gender = "female")
        else:
            self.smplLayer = SMPLLayer(model_root = os.path.join(fileDir, "smpl_model"), gender = gender)
        self.W = self.smplLayer.weights.cpu().numpy()
        self.dataSize = dataSize
        self.faces = np.array(self.smplLayer.faces, dtype = np.int64)
        self.facesOneRingIdx = getFacesOneRingIdx(self.faces)
        self.vertexOrderRandom = vertexOrderRandom
        self.complexity = complexity
        self.gender = gender
    def __getitem__(self, item):
        shape = generateShape(1, complexity = self.complexity, device = torch.device('cpu'))
        pose = generatePose(1, complexity = self.complexity, device = torch.device('cpu'))

        smplLayer = self.smplLayer
        if self.gender == "mixd" and item%2 == 1:
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
        
        rigW = np.zeros((24, 6890))
        oneHot = W.argmax(axis = 1)
        rigW[oneHot, np.arange(6890)] = 1
        
        return V, facesOneRingIdx, rigW, joints, oneHot
  
    def __len__(self):
        return self.dataSize
