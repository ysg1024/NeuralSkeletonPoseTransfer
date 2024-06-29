# This file focuses on the code for randomly generating the shape and pose parameters of the smpl model
#
# Including NPT, NBS, and ours generation strategies

import torch
import math
import numpy as np
import random

# NPT
# generate_pose_simple()The pose parameter is randomly generated under both NPT's straight or Nonstraight strategies
# generate_shape_simple()We used the strategy in NBS directly and did not use the original NPT strategy
def rad(angle):
    rad = 0
    rad = math.radians(angle)
    return rad

def lit(angle_start, angle_stop):
    random_rad = 0
    random_angle = 0
    random_angle =  random.uniform(angle_start, angle_stop)
    # print(random_angle)
    random_rad = rad(random_angle)
    return random_rad

def random_pose_straight():
  # np.random.seed(9608)
    pose = np.zeros((24, 3))
    #left arm
    arm_y_l = lit(-30, 30)
    arm_z_l = lit(-30, 30)
    pose[13]=[0, arm_y_l, arm_z_l]
    pose[16]=[0, arm_y_l, arm_z_l]

    pose[18] = [0, lit(-60, 0), 0]
    pose[20] = [lit(-10,10), lit(-10, 10), lit(-10,10)]
    pose[22] = [lit(-5,5), lit(0,10), lit(-10,0)]

    #right arm
    arm_y_r = lit(-30, 30)
    arm_z_r = lit(-30, 30)
    pose[14]=[0, arm_y_r, arm_z_r]
    pose[17]=[0, arm_y_r, arm_z_r]

    pose[19] = [0, lit(0, 60), 0]
    pose[21] = [lit(-10,10), lit(-10, 10), lit(-10,10)]
    pose[23] = [lit(-5,5), lit(-10,0), lit(0,10)]

    # #left leg
    pose[1] = [lit(-90, 0), 0, lit(0, 5)]
    pose[4] = [lit(0, 10), 0, 0]
    pose[7] = [lit(-10,20), lit(-10,10), lit(-1,1)]
    # # pose[10]=[rad(-20), 0, 0]

    # #right leg
    pose[2] = [lit(-90, 0), 0, lit(-5, 0)]
    pose[5] = [lit(0, 10), 0, 0]
    pose[8] = [lit(-10,10),  lit(-10,10), lit(-1,1)]
    # # pose[11]=[rad(), 0, 0]

    neck = lit(-1,1)
    pose[15] = [neck,neck,neck]
    pose[12] = [neck,neck,neck]

    bone = lit(-1,1)
    pose[9]=[bone,bone,bone]
    pose[6]=[bone,bone,bone]
    pose[3]=[bone,bone,bone]

    pose[0]=[lit(-2,2),lit(-2,2),lit(-2,2)]
    # print("pose done")
    return pose
def random_pose():
  # np.random.seed(9608)

    pose = np.zeros((24, 3))
    # left arm
    arm_y_l = lit(-30, 30)
    arm_z_l = lit(-30, 30)
    pose[13] = [0, arm_y_l, arm_z_l]
    pose[16] = [0, arm_y_l, arm_z_l]

    pose[18] = [0, lit(-60, 0), 0]
    pose[20] = [lit(-20,20), lit(-20, 20), lit(-20,20)]
    pose[22] = [lit(-5,5), lit(0,10), lit(-10,0)]

    # right arm
    arm_y_r = lit(-30, 30)
    arm_z_r = lit(-30, 30)
    pose[14] = [0, arm_y_r, arm_z_r]
    pose[17] = [0, arm_y_r, arm_z_r]

    pose[19] = [0, lit(0, 60), 0]
    pose[21] = [lit(-20,20), lit(-20, 20), lit(-20,20)]
    pose[23] = [lit(-5,5), lit(-10,0), lit(0,10)]

    # #left leg
    pose[1] = [lit(-90, 0), 0, lit(0, 40)]
    pose[4] = [lit(0, 100), 0, 0]
    pose[7] = [lit(-10,10), lit(-10,10), lit(-1,1)]
    # # pose[10]=[rad(-20), 0, 0]

    # #right leg
    pose[2] = [lit(-90, 0), 0, lit(-40, 0)]
    pose[5] = [lit(0, 100), 0, 0]
    pose[8] = [lit(-10,10),  lit(-10,10), lit(-1,1)]
    # # pose[11]=[rad(), 0, 0]

    neck = lit(-1,1)
    pose[15] = [neck,neck,neck]
    pose[12] = [neck,neck,neck]

    bone = lit(-1,1)
    pose[9]=[bone,bone,bone]
    pose[6]=[bone,bone,bone]
    pose[3]=[bone,bone,bone]

    pose[0]=[lit(-2,2),lit(-2,2),lit(-2,2)]
    # print("pose done")
    return pose

def generate_pose_simple(batch_size, device):
    pose = np.zeros((batch_size, 24, 3))
    for i in range(batch_size):
        choice = np.random.choice([0, 1], size=1, replace=False, p=None)[0]
        if choice == 0:
            pose[i] = random_pose()
        else:
            pose[i] = random_pose_straight()
    return torch.tensor(pose, device = device).float()

def generate_shape_simple(batch_size, device):
    bound = 4
    betas = torch.rand((batch_size, 10), device=device)
    betas = (betas - 0.5) * 2
    betas = betas * bound
    return torch.clamp(betas, -bound, bound)

# NBS
def generate_pose_skinning(batch_size, device, uniform=False, factor=1, root_rot=False, n_bone=None, ee=None):
    if n_bone is None: n_bone = 24
    if ee is not None:
        if root_rot:
            ee.append(0)
        n_bone_ = n_bone
        n_bone = len(ee)
    axis = torch.randn((batch_size, n_bone, 3), device=device)
    axis /= axis.norm(dim=-1, keepdim=True)

    angle = torch.randn((batch_size, n_bone, 1), device=device) * np.pi / 6 * factor
    angle.clamp(-np.pi, np.pi)
    poses = axis * angle
    if ee is not None:
        res = torch.zeros((batch_size, n_bone_, 3), device=device)
        for i, id in enumerate(ee):
            res[:, id] = poses[:, i]
        poses = res
    poses = poses.reshape(batch_size, -1)
    if not root_rot:
        poses[..., :3] = 0
    return poses

def generate_shape_skinning(batch_size, device = torch.device("cpu")):
    bound = 4
    betas = torch.rand((batch_size, 10), device=device)
    betas = (betas - 0.5) * 2
    betas = betas * bound
    return torch.clamp(betas, -bound, bound)

# Ours
def pose_rand(min, max):
    return math.radians(np.random.uniform(min,max,1)[0])
def shape_rand(min, max):
    return np.random.uniform(min,max,1)[0]

def generate_pose_complicated(batch_size, device):
    n_bone = 24
    poses = torch.zeros((batch_size,n_bone,3), device=device)

    for i in range(batch_size):
        pose_i = torch.tensor(
            [[0,0,0],
            [pose_rand(-100,60),pose_rand(-10,20),pose_rand(-10,60)],#1
            [pose_rand(-100,60),pose_rand(-20,10),pose_rand(-60,10)],#2
            [pose_rand(-20,30),pose_rand(-10,10),pose_rand(-30,30)],#3
            [pose_rand(-10,130),pose_rand(-10,10),pose_rand(-10,10)],#4
            [pose_rand(-10,130),pose_rand(-10,10),pose_rand(-10,10)],#5
            [pose_rand(-10,20),pose_rand(-10,10),pose_rand(-10,10)],#6
            [pose_rand(-30,60),pose_rand(-10,30),pose_rand(-10,10)],#7
            [pose_rand(-30,60),pose_rand(-30,10),pose_rand(-10,10)],#8
            [pose_rand(-10,10),pose_rand(-10,10),pose_rand(-10,10)],#9
            [pose_rand(-10,10),pose_rand(-10,10),pose_rand(-10,10)],#10
            [pose_rand(-10,10),pose_rand(-10,10),pose_rand(-10,10)],#11
            [pose_rand(-45,45),pose_rand(-45,45),pose_rand(-20,20)],#12
            [pose_rand(-10,10),pose_rand(-10,10),pose_rand(-10,10)],#13
            [pose_rand(-10,10),pose_rand(-10,10),pose_rand(-10,10)],#14
            [pose_rand(-10,10),pose_rand(-10,10),pose_rand(-10,10)],#15
            [pose_rand(-10,10),pose_rand(-90,10),pose_rand(-60,10)],#16
            [pose_rand(-10,10),pose_rand(-10,90),pose_rand(-10,60)],#17
            [pose_rand(-20,20),pose_rand(-130,10),pose_rand(-10,10)],#18
            [pose_rand(-20,20),pose_rand(-10,130),pose_rand(-10,10)],#19
            [pose_rand(-10,10),pose_rand(-10,10),pose_rand(-30,75)],#20
            [pose_rand(-10,10),pose_rand(-10,10),pose_rand(-75,30)],#21
            [pose_rand(-10,10),pose_rand(-10,10),pose_rand(-45,10)],#22
            [pose_rand(-10,10),pose_rand(-10,10),pose_rand(-10,45)]],#23
            device=device
        )
        poses[i]=pose_i
    return poses.reshape(batch_size,-1)

def generate_shape_complicated(batch_size, device):
    # Most of the shape parameters are normal between [-2,2], beyond [-4,4] they look abnormal.
    # Method 1: Random generation within a set interval
    # betas = torch.zeros((batch_size, 10), device=device)
    # for i in range(batch_size):
    #     betas[i]=torch.tensor(
    #         [shape_rand(-3,3),
    #         shape_rand(-4,3),
    #         shape_rand(-3,3),
    #         shape_rand(-3,3),
    #         shape_rand(-5,5),
    #         shape_rand(-4,4),
    #         shape_rand(-4,4),
    #         shape_rand(-5,5),
    #         shape_rand(-6,6),
    #         shape_rand(-3,6)],
    #         device=device
    #         )
   
    # Method 2: Generated parameters conform to a normal distribution
    bound = 4
    betas = torch.normal(0, 2, (batch_size, 10), device=device) / 2
    betas = torch.clamp(betas, -bound, bound)
    return betas

def generateShape(batchSize, complexity = 'all', device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    if complexity == "simple":# NPT
        shapes = generate_shape_simple(batchSize, device)
    elif complexity == "complicated":# Ours
        shapes = generate_shape_complicated(batchSize, device)
    elif complexity == "skinning":# NBS
        shapes = generate_shape_skinning(batchSize, device)
    elif complexity == "all":
        compStr = ["simple", "complicated", "skinning"]
        seletedComplexity = np.random.choice(compStr, size=1, replace=False, p=None)[0]
        return generateShape(batchSize = batchSize, complexity = seletedComplexity, device = device)
    return shapes

def generatePose(batchSize, complexity = 'all', device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    if complexity == "simple":# NPT
        poses = generate_pose_simple(batchSize, device)
    elif complexity == "complicated":# Ours
        poses = generate_pose_complicated(batchSize, device)
    elif complexity == "skinning":# NBS
        poses = generate_pose_skinning(batchSize, device)
    elif complexity == "all":
        compStr = ["simple", "complicated", "skinning"]
        seletedComplexity = np.random.choice(compStr, size=1, replace=False, p=None)[0]
        return generatePose(batchSize = batchSize, complexity = seletedComplexity, device = device)
    return poses