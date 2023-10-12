
from re import T
import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import h5py
from .smpl import SMPLLayer
from .smal import SMALLayer
import igl
from .utils import getFacesOneRingIdx

from .smpl_parameter_generation import generateShape, generatePose

def getRotationByAxis(angle, axis):
    axis = axis/np.linalg.norm(axis)#记住必须是单位向量，否则会出错
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
    def __init__(self, fileDir = '.\data\smpl_model', complexity = "all", gender = "mixed", dataSize = 5000, vertexOrderRandom = True):

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
        self.complexity = complexity
        self.gender = gender

    def __getitem__(self, item):
        shapes = generateShape(2, complexity = self.complexity, device = torch.device('cpu'))
        poses = generatePose(2, complexity = self.complexity, device = torch.device('cpu'))

        smplLayer = self.smplLayer
        if self.gender == "mixed" and item%2 == 1:
            smplLayer = self.feSmplLayer

        sV, sJ = smplLayer.forward(poses[0].unsqueeze(0), shapes[0].unsqueeze(0))
        rV, rJ = smplLayer.forward(poses[1].unsqueeze(0), shapes[1].unsqueeze(0))
        tV, tJ = smplLayer.forward(poses[1].unsqueeze(0), shapes[0].unsqueeze(0))

        sV = sV[0].cpu().numpy()
        rV = rV[0].cpu().numpy()
        tV = tV[0].cpu().numpy()

        sJ = sJ[0].cpu().numpy()
        rJ = rJ[0].cpu().numpy()
        tJ = tJ[0].cpu().numpy()

        sF = self.faces.copy()
        rF = self.faces.copy()
        sW = self.W.copy()
        rW = self.W.copy()

        sV = np.hstack((sV, igl.per_vertex_normals(sV, sF)))
        rV = np.hstack((rV, igl.per_vertex_normals(rV, rF)))

        if self.vertexOrderRandom:
            randomIdx = np.arange(sV.shape[0])
            reRandomIdx = np.arange(sV.shape[0])
            np.random.shuffle(randomIdx)
            reRandomIdx[randomIdx] = np.arange(sV.shape[0])[:]
            sV = sV[randomIdx]
            tV = tV[randomIdx]
            sW = sW[randomIdx]
            sF = reRandomIdx[sF]
            
            randomIdx = np.arange(sV.shape[0])
            reRandomIdx = np.arange(sV.shape[0])
            np.random.shuffle(randomIdx)
            reRandomIdx[randomIdx] = np.arange(sV.shape[0])[:]
            rV = rV[randomIdx]
            rW = rW[randomIdx]
            rF = reRandomIdx[rF]
            
        sFacesOneRingIdx = getFacesOneRingIdx(sF)
        rFacesOneRingIdx = getFacesOneRingIdx(rF)

        sRigW = np.zeros((24, 6890))
        oneHot = sW.argmax(axis = 1)
        sRigW[oneHot, np.arange(6890)] = 1

        rRigW = np.zeros((24, 6890))
        oneHot = rW.argmax(axis = 1)
        rRigW[oneHot, np.arange(6890)] = 1
        
        return sV, sFacesOneRingIdx, sW, sRigW, sJ, rV, rFacesOneRingIdx, rW, rRigW, rJ, tV, tJ
    
    def __len__(self):
        return self.dataSize