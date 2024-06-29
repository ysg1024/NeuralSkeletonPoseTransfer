import torch

def L1Norm(test_data, gt):
    l1loss = torch.nn.L1Loss()
    return float(l1loss(test_data, gt))

def L2Norm(test_data, gt):
    l2loss = torch.nn.MSELoss()
    return float(l2loss(test_data, gt))