import torch
import numpy as np
import yastn.yastn as yastn
from yastn.yastn.tn.fpeps._peps import Peps
from yastn.yastn.tn.fpeps._geometry import SquareLattice
from yastn.yastn.backend import backend_torch
from yastn.yastn.sym import sym_none, sym_Z2


def sigma(i):
    return 2 * i - 1

def Ising_critical_beta():
    return np.log(1 + np.sqrt(2)) / 2

def Ising_dense(beta, h=0):
    config = yastn.make_config(
        backend=backend_torch,
        sym=sym_none,
        fermionic=False,
        default_dtype="complex128",
        tensordot_policy="no_fusion",
        default_device="cpu",
    )

    T_data = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    T_data[i, j, k, l] = np.exp(
                        beta * sigma(i) * sigma(j) + beta * sigma(j) * sigma(l) +\
                        beta * sigma(l) * sigma(k) + beta * sigma(k) * sigma(i) +\
                        beta * h / 2 * (sigma(i) + sigma(j) + sigma(k) + sigma(l))
                    )

    T = yastn.Tensor(config=config, s=(1, 1, -1, -1))
    T.set_block(Ds=(2,2,2,2), val=T_data)


    tensors={
        (0,0):T,
        (0,1):T,
        (1,0):T,
        (1,1):T
    }
    return Peps(SquareLattice((2, 2), 'infinite'), tensors=tensors)