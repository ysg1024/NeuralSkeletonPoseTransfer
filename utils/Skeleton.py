import numpy as np

def getBonesEdge(num_joints = 24):
    bones={1:0, 2:0, 3:0, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 10:7, 11:8, 12:9, 13:9,
            14:9, 15:12, 16:13, 17:14, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21}#joint hierarchy{child:father}
    bones_29={1:0, 2:0, 3:0, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 10:7, 11:8, 12:9, 13:9,
            14:9, 15:12, 16:13, 17:14, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21,
            24:10, 25:11, 26:15, 27:22, 28:23}#joint hierarchy{child:father}
    bones_edges=[]
    for i in range(1,num_joints):
        if(num_joints == 24):
            bones_edges.append(np.array([i,bones[i]]))
        elif(num_joints == 29):
            bones_edges.append(np.array([i,bones_29[i]]))
        else:
            bones_edges.append(np.array([i,bones_29[i]]))
            
    bones_edges=np.array(bones_edges)
    return bones_edges