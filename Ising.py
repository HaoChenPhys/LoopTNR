import torch
import numpy as np
import yastn.yastn as yastn
from yastn.yastn.tn.fpeps._peps import Peps
from yastn.yastn.tn.fpeps._geometry import SquareLattice, RectangularUnitcell
from yastn.yastn.backend import backend_torch
from yastn.yastn.sym import sym_none, sym_Z2


def sigma(i):
    return 2 * i - 1

def Ising_critical_beta():
    return np.log(1 + np.sqrt(2)) / 2

def Ising_dense(beta, h=0):
    """
    A single tensor represents a \sqrt{2} \times \sqrt{2} patch.
    """
    config = yastn.make_config(
        backend=backend_torch,
        sym=sym_none,
        fermionic=False,
        default_dtype="float64",
        tensordot_policy="no_fusion",
        default_device="cpu",
    )

    T_data = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    T_data[i, j, k, l] = np.exp(
                        beta * sigma(i) * sigma(j) + beta * sigma(j) * sigma(k) +\
                        beta * sigma(l) * sigma(k) + beta * sigma(l) * sigma(i) +\
                        beta * h / 2 * (sigma(i) + sigma(j) + sigma(k) + sigma(l))
                    )

    T = yastn.Tensor(config=config, s=(1, 1, -1, -1))
    T.set_block(Ds=(2,2,2,2), val=T_data)

    # pattern = {(0,0):0, (0,1):1, (1,0):1, (1,1):0}
    # tensors={
    #     (0,0):T,
    #     (0,1):T
    # }

    # better performance
    pattern = {(0,0):0, (0,1):1, (1,0):2, (1,1):3}
    tensors={
        (0,0):T,
        (0,1):T,
        (1,0):T,
        (1,1):T,
    }
    return Peps(RectangularUnitcell(pattern, 'infinite'), tensors=tensors)

def Ising_Z2_symmetric(beta):
    """
    A single tensor represents a 1x1 patch. Obtained from the dual representation (bond picture) of the partition function.
    """
    config = yastn.make_config(
        backend=backend_torch,
        sym=sym_Z2,
        fermionic=False,
        default_dtype="float64",
        tensordot_policy="no_fusion",
        default_device="cpu",
    )

    legs = [
        yastn.Leg(config, s=1, t=(0, 1), D=(1, 1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(1, 1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(1, 1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(1, 1)),
    ]


    T = yastn.zeros(config=config, legs=legs)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if (i+j+k+l) % 2 == 0:
                        T.set_block(ts=(i, j, k, l), Ds=(1,1,1,1), val=np.exp(beta*(i+j+k+l-2)))

    # pattern = {(0,0):0, (0,1):1, (1,0):1, (1,1):0}
    # tensors={
    #     (0,0):T,
    #     (0,1):T
    # }

    # better performance
    pattern = {(0,0):0, (0,1):1, (1,0):2, (1,1):3}
    tensors={
        (0,0):T,
        (0,1):T,
        (1,0):T,
        (1,1):T,
    }
    return Peps(RectangularUnitcell(pattern, 'infinite'), tensors=tensors)

def F_exact(beta):
    N = 10000;
    maglambda = 1/(np.sinh(2*beta)**2)
    x = np.linspace(0,np.pi,N+1)
    y = np.log(np.cosh(2*beta)*np.cosh(2*beta) + (1/maglambda)*np.sqrt(
            1+maglambda**2 -2*maglambda*np.cos(2*x)))
    return -1/beta*((np.log(2)/2) + 0.25*sum(y[1:(N+1)] + y[:N])/N)

def Ising_post_processing(TNR):
    # only for 2x2
    norm = trace_2x2(TNR)
    for site in TNR.psi.sites():
        TNR.psi[site] = TNR.psi[site]/norm**(1/4)
    return norm**(1/4)


def trace_2x2(TNR):
    t1, t2, t3, t4 = TNR.psi[(0,0)], TNR.psi[(0,1)], TNR.psi[(1,1)], TNR.psi[(1, 0)]
    L = yastn.tensordot(t1, t4, axes=((0, 2), (2, 0)))
    R = yastn.tensordot(t2, t3, axes=((0, 2), (2, 0)))
    return yastn.tensordot(L, R, axes=((0, 1, 2, 3), (1, 0, 3, 2))).to_number().real
