import os
import sys
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import time

class Trainer():
    def __init__(self, 
    net,                        #model
    trainLossCalc,              #loss class or function
    batchSize,
    trainDataset,
    testDataset = None,
    validDataset = None,        #validation Dataset, if none: no validate
    testLossCalc = None,        #if none: use train loss
    trainLossName = None,
    testLossName = None,                                                                      
    learnningRate = 0.0001, 
    epochs = 100,
    device = None,
    opt = None,                 #optim
    scheduler = None, 
    initMsgCallBack = None,
    batchMsgCallBack = None,
    epochMsgCallBack = None,
    finishMsgCallBack = None,
    msgOut = None,
    savePath = None,
    saveEpochPath = False,
    args = None):

        self.net = net
        self.trainLossCalc = trainLossCalc
        self.batchSize = batchSize

        self.trainDataset = trainDataset
        self.trainIter = DataLoader(trainDataset, num_workers=0, batch_size=self.batchSize, shuffle=True, drop_last=True)

        self.testDataset = testDataset
        if testDataset is not None:
            self.testIter = DataLoader(testDataset, num_workers=0, batch_size=self.batchSize, shuffle=True, drop_last=False)
        else:
            self.testIter = None

        self.validDataset = validDataset
        if validDataset is not None:
            self.validIter = DataLoader(validDataset, num_workers=0, batch_size=self.batchSize, shuffle=True, drop_last=False)
        else:
            self.validIter = None

        if testLossCalc is not None:
            self.testLossCalc = testLossCalc
        else:
            self.testLossCalc = self.trainLossCalc

        self.lr = learnningRate
        self.epochs = epochs
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if opt is not None:
            self.opt = opt
            
        else:
            self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        #scheduler
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = CosineAnnealingLR(self.opt, self.epochs, eta_min = self.lr)

        if initMsgCallBack is not None:
            self.initMsg = initMsgCallBack
        else:
            self.initMsg = self.defualtInitMsg

        if batchMsgCallBack is not None:
            self.batchMsg = batchMsgCallBack
        else:
            self.batchMsg = self.defualtBatchMsg

        if epochMsgCallBack is not None:
            self.epochMsg = epochMsgCallBack
        else:
            self.epochMsg = self.defualtEpochMsg

        if finishMsgCallBack is not None:
            self.finishMsg = finishMsgCallBack
        else:
            self.finishMsg = self.defualtFinishMsg

        self.msgOut = msgOut if msgOut is not None else print
        self.trainLossName = trainLossName
        self.testLossName = testLossName if testLossName is not None else trainLossName

        self.savePath = savePath
        self.saveEpochPath = saveEpochPath
        self.args = args

        self.net = self.net.to(self.device)
        self.initMsg(self.args)

    def train(self):
        trainLosses = None
        trainBeginTime = time.time()
        self.msgOut("trainning...")
        for epoch in range(self.epochs):
            self.net = self.net.train()

            epochBeginTime = time.time()
            completedNum = 0
            runTimes = 0
            epochLosses = None
            for inputs in self.trainIter:
                inputs = self.toDevice(inputs)

                self.opt.zero_grad()
                pred = self.net(inputs)
                losses = self.trainLossCalc(pred, inputs)
                l = sum(losses)
                l.backward()
                self.opt.step()

                batchLoss = self.lossToNumpy(losses)
                if epochLosses is None:
                    epochLosses = batchLoss
                else:
                    epochLosses += batchLoss
                runTimes += 1
                completedNum += self.batchSize
                self.batchMsg(completedNum, len(self.trainDataset), epochBeginTime, time.time(), batchLoss, self.trainLossName)
            epochLosses /= runTimes

            self.defualtEpochMsg(epoch, (time.time() - epochBeginTime) / 60, self.scheduler.get_last_lr()[0], epochLosses, self.trainLossName)
            self.scheduler.step()
            if trainLosses is None:
                trainLosses = epochLosses
            else:
                trainLosses += epochLosses

            if self.validIter is not None:
                self.msgOut("validation...")
                self.inference(self.validIter)

            if self.saveEpochPath is not None:
                torch.save(self.net.state_dict(), os.path.join(self.saveEpochPath, 'net_%d.pkl'%epoch))

        trainLosses /= self.epochs
        self.msgOut("train completed!")
        self.finishMsg((time.time() - trainBeginTime) / 60, trainLosses, self.trainLossName)

        if self.savePath is not None:
            torch.save(self.net.state_dict(), os.path.join(self.savePath, 'net_%s.pkl'%(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))))

        if self.testIter is not None:
            self.test()

    def test(self):
        self.msgOut("testing...")
        self.inference(self.testIter)

    def inference(self, dataIter):
        beginTime = time.time()
        runTimes = 0
        testLosses = None
        self.net = self.net.eval()

        for inputs in dataIter:
            inputs = self.toDevice(inputs)
            with torch.no_grad():
                pred = self.net(inputs)
                losses = self.testLossCalc(pred, inputs)

            batchLoss = self.lossToNumpy(losses)
            if testLosses is None:
                testLosses = batchLoss
            else:
                testLosses += batchLoss
            runTimes += 1
        testLosses /= runTimes
        self.finishMsg((time.time() - beginTime) / 60, testLosses, self.testLossName)

    def toDevice(self, data):
        if type(data) == list:
            for i in range(len(data)):
                data[i] = data[i].to(self.device)
        return data

    def lossToNumpy(self, losses):
        lossItems = []
        for loss in losses:
            lossItems.append(float(loss))
        return np.array(lossItems)

    def defualtInitMsg(self, args = None):
        msg = "Init completed! "
        msg += "learning rate:%f, "%self.lr
        msg += "batch size:%d, "%self.batchSize
        msg += "epochs:%d, "%self.epochs
        msg += "device:%s, "%self.device
        msg += "train dataset size:%d, "%len(self.trainDataset)
        if self.testIter is not None:
            msg += "test dataset size:%d, "%len(self.testDataset)
        if self.validIter is not None:
            msg += "validation dataset size:%d, "%len(self.validDataset)

        if args is not None:
            msg += "other information: %s"%args
        self.msgOut(msg)

    def defualtBatchMsg(self, completedNum, allNum, beginTime, endTime, losses, lossName):
        #don't do anything
        pass

    def defualtEpochMsg(self, epoch, usedTime, lr, trainLosses, trainlossName = None):
        msg ='\n'
        msg += "epoch:%d, "%(epoch+1)
        msg += "used %0.2f m, "%usedTime
        msg += "learning rate: %f, "%lr
        for i, loss in enumerate(trainLosses):
            if trainlossName is None:
                msg += "loss %d: %f, "%(i, loss)
            else:
                msg += "%s loss: %f, "%(trainlossName[i], loss)
        msg += "all loss: %f. "%sum(trainLosses)
        self.msgOut(msg)

    def defualtFinishMsg(self, usedTime, losses, lossName):
        msg = "used %0.2f m, "%usedTime
        for i, loss in enumerate(losses):
            if lossName is None:
                msg += "loss %d: %f, "%(i, loss)
            else:
                msg += "%s loss: %f, "%(lossName[i], loss)
        msg += "all loss: %f. "%sum(losses)
        self.msgOut(msg)
        