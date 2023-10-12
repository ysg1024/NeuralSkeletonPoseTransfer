
import torch
import torch.nn as nn
import torch.nn.functional as Function


class TransformNet(nn.Module):
    def __init__(self):
        super(TransformNet, self).__init__()
        self.k = 3
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        nn.init.constant_(self.transform.weight, 0)
        nn.init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x, idx = None):
        batch_size = x.size(0)
        if idx is not None:
            x = getGraphFeature(x, k = self.k, idx = idx)
        else:
            x = getGraphFeature(x, k = self.k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = Function.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = Function.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class NoneLayer(nn.Module):
    def __init__(self):
        super(NoneLayer, self).__init__()
    def forward(self, x):
        return x

def getNormLayer(channel, layer = 32, norm = "InstanceNorm1d"):
    if norm == "BatchNorm1d":
        layerNorm = nn.BatchNorm1d(channel)
    elif norm == "InstanceNorm1d":
        layerNorm = nn.InstanceNorm1d(channel)
    elif norm == "BatchNorm2d":
        layerNorm = nn.BatchNorm2d(channel)
    elif norm == "InstanceNorm2d":
        layerNorm = nn.InstanceNorm2d(channel)
    elif norm == "GroupNorm1d" or norm == "GroupNorm2d":
        layerNorm = nn.GroupNorm(layer, channel)
    else:
        layerNorm = NoneLayer()
    return layerNorm


def knn(x, k=20):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def getGraphFeature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    device = x.device
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    else:
        k = idx.shape[2]
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    
    x = x.transpose(2, 1).contiguous() 
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class DGCNNLayer(nn.Module):
    def __init__(self, intputNum, mlp, norm = "BatchNorm", knnNum = 20, bias = False):
        super(DGCNNLayer, self).__init__()
        self.k = knnNum

        modelList = []
        intputNum = int(intputNum * 2)
        for outputNum in mlp:
            modelList.append(nn.Conv2d(intputNum, outputNum, kernel_size=1, bias = bias))
            modelList.append(getNormLayer(outputNum, norm = norm+'2d'))
            modelList.append(nn.LeakyReLU(negative_slope=0.2))
            intputNum = outputNum
        self.conv = nn.Sequential(*modelList)

    def forward(self, x, idx = None):
        x = getGraphFeature(x, k = self.k, idx = idx)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x

class DGCNNCatLayer(nn.Module):
    def __init__(self, intputNum, outputNum, doubleConv = False, norm = "BatchNorm", knnNum = 20, bias = False):
        super(DGCNNCatLayer, self).__init__()
        if doubleConv:
            layerMlp = [outputNum, outputNum]
        else:
            layerMlp = [outputNum]
        self.DLayer = DGCNNLayer(intputNum, layerMlp, norm = norm, knnNum = knnNum, bias = bias)
        self.GLayer = DGCNNLayer(intputNum, layerMlp, norm = norm, knnNum = knnNum, bias = bias)

        self.mlp =  nn.Sequential(nn.Conv1d(2*outputNum, outputNum, kernel_size=1, bias = bias),
                                    getNormLayer(outputNum, norm = norm+'1d'),
                                    nn.LeakyReLU(negative_slope=0.2))
    def forward(self, x, gIdx):
        x1 = self.DLayer(x)
        x2 = self.GLayer(x, gIdx)
        x = torch.cat((x1, x2), dim = 1)
        x = self.mlp(x)
        return x

class DGCNN_origin(nn.Module):
    def __init__(self, inChannel = 3, layers = [64, 128, 256], norm = "BatchNorm", transform = False, knnNum = 20, bias = False):
        super(DGCNN_origin, self).__init__()
        self.layers = nn.ModuleList()
        intputNum = inChannel
        for outputNum in layers:
            self.layers.append(DGCNNLayer(intputNum, mlp = [outputNum], norm = norm, knnNum = knnNum, bias = bias))
            intputNum = outputNum
        if transform == True:
            self.transformNet = TransformNet()
        else:
            self.transformNet = None
    def forward(self, x):
        if self.transformNet is not None:
            t = self.transformNet(x)
            x = torch.bmm(t, x)
        outs = []
        outs.append(x)
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        x = torch.cat(outs, dim=1)        
        return x

class DGCNN(nn.Module):
    def __init__(self, inChannel = 3, layers = [64, 128, 256], doubleConv = False, norm = "BatchNorm", transform = False, knnNum = 20, bias = False):
        super(DGCNN, self).__init__()
        self.layers = nn.ModuleList()
        intputNum = inChannel
        for outputNum in layers:
            self.layers.append(DGCNNCatLayer(intputNum, outputNum, doubleConv = doubleConv, norm = norm, knnNum = knnNum, bias = bias))
            intputNum = outputNum
        if transform == True:
            self.transformNet = TransformNet()
        else:
            self.transformNet = None
    def forward(self, x, gIdx):
        if self.transformNet is not None:
            t = self.transformNet(x, gIdx)
            x = torch.bmm(t, x)
        outs = []
        outs.append(x)
        for layer in self.layers:
            x = layer(x, gIdx)
            outs.append(x)
        x = torch.cat(outs, dim=1)      #[B, 3 + sum(layers), N]      
        return x