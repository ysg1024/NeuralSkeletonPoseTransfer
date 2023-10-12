import os
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import argparse
import time

from model.PoseNet import PoseNet, ReconstructionLoss
from model.dataset import PoseDataset
from LogPrinter import LogPrinter

def train(net, dataIter, loss, opt, device):
    epochLoss = 0.0
    batchDataNum = 0
    net.train()
    for sV, rV, tV, F, oneRingIdx in dataIter:
        sV, rV, tV = sV.to(device).float(), rV.to(device).float(), tV.to(device).float()
        oneRingIdx = oneRingIdx.to(device).long()
        opt.zero_grad()
        preV = net(sV, rV, oneRingIdx)
        l = loss(preV, sV, tV, oneRingIdx)
        l.backward()
        opt.step()
        epochLoss += float(l)
        batchDataNum += 1
        
    return epochLoss / batchDataNum

def test(net, dataIter, loss, device):
    losses = 0
    batchTime = 0 
    for sV, rV, tV, F, oneRingIdx in dataIter:
        sV, rV, tV = sV.to(device).float(), rV.to(device).float(), tV.to(device).float()
        oneRingIdx = oneRingIdx.to(device).long()
        F = F.to(device).long()
        with torch.no_grad():
            preV = net(sV, rV, oneRingIdx)
            losses = loss(preV, sV, tV, F, oneRingIdx) + losses
        batchTime += 1
        
    return losses / batchTime

def main(args):
    rootFolder = "."
    
    dataPath = os.path.join(rootFolder, 'data')
    stateDickPath = os.path.join(rootFolder, 'stateDict')
    logPath = os.path.join(rootFolder, "log")
    timeLocalStr = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    logPath = os.path.join(logPath, 'trainLog%s.log'%timeLocalStr)
    logPrinter = LogPrinter(logPath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    trainDataset = PoseDataset(dataPath, dataSize = 10000, rNoise = args.rNoise, sNoise = args.sNoise, rDropout = args.rDropout, sDropout = args.sDropout)
    trainIter = DataLoader(trainDataset, num_workers=0, batch_size=args.batchSize, shuffle=True, drop_last=True)

    allTrainNUM = len(trainDataset)
    
    lr = args.lr
    epochs = args.epochs
    net = PoseNet()
    net = net.to(device)

    if args.readState == True:
        net.load_state_dict(torch.load(os.path.join(stateDickPath, "net.pkl")))

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(opt, epochs, eta_min=lr)

    trainLoss = ReconstructionLoss(0.01)

    lossesName = ["PMD", "ELR", "LCD"]
    logPrinter.addInitImformation(args, device, allTrainNUM, 0, 0)
    logPrinter.addDash()
    logPrinter("trainning")
    trainBeginTime = time.time()
    for epoch in range(epochs):
        epochTime = time.time()

        net = net.train()
        tl = train(net, trainIter, trainLoss, opt, device)
        scheduler.step()

        torch.save(net.state_dict(),os.path.join(stateDickPath, 'net_%s.pkl'%timeLocalStr))

        vl = [0, 0, 0]
        
        logPrinter.addEpochImformation(epoch+1, tl, lossesName, vl, time.time() - epochTime)
    logPrinter.addTrainCompleteImformation(time.time() - trainBeginTime)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train NeuralPoseTransfer')
    parser.add_argument('-lr', default = 0.0001, type=float)
    parser.add_argument('-batchSize', default = 16, type=int)
    parser.add_argument('-epochs', default = 100, type=int) 
    parser.add_argument('-rDropout', default = 0.7, type=float)
    parser.add_argument('-sDropout', default = 0, type=float)
    parser.add_argument('-readState', default = False, action = 'store_true')
    parser.add_argument('-rNoise', default = 0.5, type=float)
    parser.add_argument('-sNoise', default = 1, type=float)
    main(parser.parse_args())