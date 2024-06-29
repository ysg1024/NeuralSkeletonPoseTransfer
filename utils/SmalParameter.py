# This file focuses on the code for randomly generating the shape and pose parameters of the smal model
#
# Since neither NPT nor 3D-coreNet has smal code, we implemented the code ourselves

import numpy as np
import math
import torch

def generateShape(batch_size, toys_betas, device = torch.device("cpu")):
    '''
    return:[batch_size, 41]
    '''
    # Randomize the shape parameters from the official 41 toys parameters
    id=np.random.choice(range(41), size=(batch_size), replace=True)
    betas = torch.zeros((batch_size, 41), device=device)
    for i in range(batch_size):
        betas[i]=toys_betas[id[i]]

    return betas

def pose_rand(min, max):
    return math.radians(np.random.uniform(min,max,1)[0])

def generatePose(batch_size, device = torch.device("cpu")):
    n_bone = 33
    poses = torch.zeros((batch_size,n_bone,3), device=device)

    for i in range(batch_size):
        pose_i = torch.tensor(
            [
            #body
            [0,0,0],#0
            [pose_rand(-5,5),pose_rand(-5,5),pose_rand(-5,5)],#1
            [pose_rand(-5,5),pose_rand(-5,5),pose_rand(-5,5)],#2
            [pose_rand(-10,10),pose_rand(-5,5),pose_rand(-5,5)],#3
            [pose_rand(-10,10),pose_rand(-10,10),pose_rand(-20,20)],#4
            [pose_rand(-10,10),pose_rand(-5,5),pose_rand(-20,20)],#5
            [pose_rand(-5,5),pose_rand(-10,10),pose_rand(-15,15)],#6
            #left front leg
            [pose_rand(0,20),pose_rand(-60,25),pose_rand(-5,5)],#7
            [pose_rand(-5,10),pose_rand(-5,5),pose_rand(-5,5)],#8
            [pose_rand(-5,5),pose_rand(-10,30),pose_rand(-5,5)],#9
            [pose_rand(-5,5),pose_rand(-5,5),pose_rand(-5,5)],#10
            #right front leg
            [pose_rand(-20,0),pose_rand(-60,25),pose_rand(-5,5)],#11
            [pose_rand(-10,5),pose_rand(-5,5),pose_rand(-5,5)],#12
            [pose_rand(-5,5),pose_rand(-10,30),pose_rand(-5,5)],#13
            [pose_rand(-5,5),pose_rand(-5,5),pose_rand(-5,5)],#14
            #head
            [pose_rand(-20,20),pose_rand(-30,30),pose_rand(-30,30)],#15
            [pose_rand(-5,5),pose_rand(-20,30),pose_rand(-25,25)],#16
            #left hind leg
            [pose_rand(0,20),pose_rand(-15,15),pose_rand(-5,5)],#17
            [pose_rand(-5,15),pose_rand(-10,30),pose_rand(-5,5)],#18
            [pose_rand(-5,5),pose_rand(-5,5),pose_rand(-5,5)],#19
            [pose_rand(-5,5),pose_rand(-5,5),pose_rand(-5,5)],#20
            #right hind leg
            [pose_rand(-20,0),pose_rand(-15,15),pose_rand(-5,5)],#21
            [pose_rand(-15,5),pose_rand(-10,30),pose_rand(-5,5)],#22
            [pose_rand(-5,5),pose_rand(-5,5),pose_rand(-5,5)],#23
            [pose_rand(-5,5),pose_rand(-5,5),pose_rand(-5,5)],#24
            #tail
            [pose_rand(-5,5),pose_rand(-60,60),pose_rand(-60,60)],#25
            [pose_rand(-5,5),pose_rand(-5,5),pose_rand(-5,5)],#26
            [pose_rand(-5,5),pose_rand(-5,5),pose_rand(-5,5)],#27
            [pose_rand(-5,5),pose_rand(-30,30),pose_rand(-30,30)],#28
            [pose_rand(-5,5),pose_rand(-5,5),pose_rand(-5,5)],#29
            [pose_rand(-5,5),pose_rand(-5,5),pose_rand(-5,5)],#30
            [pose_rand(-5,5),pose_rand(-5,5),pose_rand(-5,5)],#31
            #mouth
            [pose_rand(0,0),pose_rand(0,5),pose_rand(0,0)],#32
            ],
            device=device
        )
        poses[i]=pose_i
    return poses.reshape(batch_size,-1)
