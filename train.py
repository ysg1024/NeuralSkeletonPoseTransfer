import os
import sys
import time
import argparse
import torch

from model.SkinningDataset import SMPLRandomDataset
from model.SkinningNet import SkinningNet
from model.utils import TrainLoss
from trainer import Trainer

timeLocalStr = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
logPath = os.path.join('.', "log")
logPath = os.path.join(logPath, 'train-%s.log'%timeLocalStr)

def mesOut(msg):
    global logPath
    logFile = open(logPath, "a")
    logFile.write(msg + "\n")
    logFile.close()

def main(args):
    fileDir = "."
    dataDir = os.path.join(fileDir, 'data')
    stateDictDir = os.path.join(fileDir, 'stateDict')

    net = SkinningNet(pretrain = args.pretrain)
    if args.readState:
        net.load_state_dict(torch.load(os.path.join(stateDictDir, 'net.pkl')))
    
    trainDataset = SMPLRandomDataset(fileDir = dataDir, dataSize = args.dataSize, complexity = args.complexity)
    trainLoss = TrainLoss()
    
    trainer = Trainer(
    net = net,                        #model
    trainLossCalc = trainLoss,              
    batchSize = args.batchSize,
    trainDataset = trainDataset,
    testDataset = None,
    validDataset = None,        
    testLossCalc = None,        
    trainLossName = ['joints', 'weights', 'acc'],
    testLossName = None,                                                                      
    learnningRate = args.lr, 
    epochs = args.epochs,
    device = None,
    opt = None,                 
    scheduler = None, 
    initMsgCallBack = None,
    batchMsgCallBack = None,
    epochMsgCallBack = None,
    finishMsgCallBack = None,
    msgOut = mesOut,
    savePath = stateDictDir,
    saveEpochPath = os.path.join(stateDictDir, 'epochs'),
    args = "readState:%s, pretrain:%s, complexity:%s, model: Neural Skeleton Pose Transfer"%(args.readState, args.pretrain, args.complexity))

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train Neural Skeleton Pose Transfer')
    parser.add_argument('-lr', default = 0.0001, type=float)
    parser.add_argument('-batchSize', default = 4, type=int)
    parser.add_argument('-epochs', default = 200, type=int)
    parser.add_argument('-dataSize', default = 5000, type=int)
    parser.add_argument('-complexity', default = 'skinning', type=str)
    parser.add_argument('-pretrain', default = True, action = 'store_false')
    parser.add_argument('-readState', default = False, action = 'store_true')

    main(parser.parse_args())
