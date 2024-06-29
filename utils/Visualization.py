import meshplot as mp
import numpy as np

def showWeight(v, f, w):
    '''
    w:tensor[v,24]
    random_color:ndarray[24,3]
    '''
    random_color=np.random.rand(24,3)
    random_color=np.array(
        [[0.60844865, 0.67606524, 0.89751869],
       [0.34662374, 0.87924273, 0.16330115],
       [0.99286059, 0.53298634, 0.27923872],
       [0.10853724, 0.35786464, 0.43192915],
       [0.84132311, 0.08994834, 0.5649118 ],
       [0.78389742, 0.7244921 , 0.97544642],
       [0.98516718, 0.91215656, 0.21931679],
       [0.98336802, 0.4267378 , 0.1509739 ],
       [0.70121629, 0.3388067 , 0.60773002],
       [0.78736424, 0.01482043, 0.45515315],
       [0.39336423, 0.95850855, 0.40600845],
       [0.99976509, 0.15790639, 0.4450985 ],
       [0.05931249, 0.31969211, 0.39730457],
       [0.43026329, 0.34496678, 0.19833945],
       [0.09241686, 0.36364845, 0.63692125],
       [0.27824061, 0.7471906 , 0.0462318 ],
       [0.61524596, 0.40103437, 0.68041187],
       [0.51378424, 0.80270686, 0.30132781],
       [0.14208116, 0.45105504, 0.2458831 ],
       [0.21886638, 0.53077705, 0.47668985],
       [0.29557562, 0.06410305, 0.1366658 ],
       [0.13489016, 0.12206894, 0.09888925],
       [0.1997769 , 0.38859844, 0.50672111],
       [0.47818286, 0.52523424, 0.41818795]]
    )
    w=np.array(w)
    v_color=w.dot(random_color)
    viewer=mp.plot(v,f,c=v_color,shading={"flat":True})
    # viewer=mp.plot(v,c=v_color,shading={"point_size":0.2})