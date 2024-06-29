import numpy as np
import igl

from typing import Tuple
import torch

def cot_laplacian(
    verts: torch.Tensor, faces: torch.Tensor, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the Laplacian matrix with cotangent weights and the inverse of the
    face areas.
    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        faces: tensor of shape (F, 3) containing the vertex indices of each face
    Returns:
        2-element tuple containing
        - **L**: Sparse FloatTensor of shape (V,V) for the Laplacian matrix.
           Here, L[i, j] = cot a_ij + cot b_ij iff (i, j) is an edge in meshes.
           See the description above for more clarity.
        - **inv_areas**: FloatTensor of shape (V,) containing the inverse of sum of
           face areas containing each vertex
    """
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    # pyre-fixme[16]: `float` has no attribute `clamp_`.
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=eps).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # For each vertex, compute the sum of areas for triangles containing it.
    idx = faces.view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=verts.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(-1, 1)

    return L, inv_areas

def getBetaMatrix(LT,p,lamda=20):
    I = np.identity(LT.shape[0])

    # B = np.linalg.inv(I + lamda*LT) # ??:The Kernel crashed while executing code in the the current cell or a previous cell. 

    I = torch.tensor(I)
    LT = torch.tensor(LT)
    B = torch.linalg.inv(I + lamda*LT)
    B = np.array(B)



    B = B**p
    return np.array(B)

def smoothWeight(v,f,rigid_w):
    Laplacian1 = igl.cotmatrix(v,f)

    L, inv_areas = cot_laplacian(torch.tensor(v),torch.tensor(f).long())
    L = L.to_dense()
    Laplacian2 = (L-torch.diag(torch.sum(L,dim=0)))/2
    Laplacian2 = np.array(Laplacian2)
    
    Laplacian = Laplacian2


    for i in range(Laplacian.shape[0]):
        Laplacian[i] = Laplacian[i] / Laplacian[i,i]

    B=getBetaMatrix(Laplacian, p=1, lamda=20)


    
    # BW=B.dot(rigid_w) # ??:The Kernel crashed while executing code in the the current cell or a previous cell. 

    B = torch.tensor(B)
    rigid_w = torch.tensor(rigid_w, dtype=torch.float64)

    BW = torch.mm(B,rigid_w)
    BW = np.array(BW)

    return BW#(v,J)