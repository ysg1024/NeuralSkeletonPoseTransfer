'''
reference:
https://github.com/bharat-b7/MultiGarmentNetwork
'''

import pickle
import numpy as np

def  read_mapping_file(path='./assets/hresMapping.pkl'):
    '''
    https://blog.csdn.net/Netceor/article/details/107352007
    '''
    with open(path, 'rb') as f:
        mapping, garment_faces = pickle.load(f, encoding='iso-8859-1')
    return mapping, garment_faces

def get_garment_weight(smpl_weight):
    '''
    smpl_weight:[6890,24]
    '''
    mapping, garment_faces = read_mapping_file()

    weights_hres = np.hstack([
            np.expand_dims(
                np.mean(
                    mapping.dot(np.repeat(np.expand_dims(smpl_weight[:, i], -1), 3)).reshape(-1, 3)#[27554*3,6890*3].dot([6890*3]).reshape(-1,3)
                    , axis=1),
                axis=-1)
            for i in range(24)
        ])

    return weights_hres
