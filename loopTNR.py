import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from dataclasses import dataclass
from typing import Union

import yastn.yastn as yastn
from yastn.yastn.tensor import Tensor
from yastn.yastn import decompress_from_1d
from yastn.yastn.tn.fpeps._peps import Peps, Peps2Layers
from yastn.yastn.backend import backend_torch

from Ising import Ising_dense, Ising_critical_beta

@dataclass()
class Ent_projectors():
    r""" Dataclass for entanglement-filtering projectors associated with Peps lattice site. """
    t : Union[Tensor, None] = None  # top
    l : Union[Tensor, None] = None  # left
    b : Union[Tensor, None] = None  # bottom
    r : Union[Tensor, None] = None  # right

@dataclass()
class Decomp_T():
    r""" Dataclass for the four decomposed tensors, which is used to form the coarse-grained tensor.
    convention:

         0                    2
          \                  /
          tl--2          0--tr
          |                 |
          1                 1

          1                 0
          |                 |
          br--2          1--br
        /                    \
       0                      2
    """

    tl : Union[Tensor, None] = None  # top-left
    tr : Union[Tensor, None] = None  # top-right
    br : Union[Tensor, None] = None  # bottom-right
    bl : Union[Tensor, None] = None  # bottom-left

def same_sectors(A, B):
    if [l.tD for l in A.get_legs()] != [l.tD for l in B.get_legs()]:
        return False
    return True

def QR_L(L, T):
    # A single step of the QR decomposition from the left with 3 in-coming legs
    #       |     |
    #        1   2
    #         v v
    # --L-0-<--T--<-3-----
    # =
    #       |     |
    #        1   2
    #         v v
    # ----0-<--Q--<-3--R-
    tmp = L @ T
    _, R = tmp.qr(axes=((0, 1, 2), (3,)), sQ=T.s[3])
    return R/R.norm(p='inf')

def QR_R(R, T):
    # A single step of the QR decomposition from the right with 3 in-coming legs
    #        |     |
    #         1   2
    #          v v
    # -----0-<--T--<-3-R--
    # =
    #        |     |
    #         1   2
    #          v v
    # -L---0-<--Q--<-3----
    tmp = T @ R
    _, L = tmp.qr(axes=((1, 2, 3), (0,)), Qaxis=0, Raxis=1, sQ=T.s[0])
    return L/L.norm(p='inf')

def find_L(psi, max_sweeps, conv_check):
    '''
    Compute the L tensors for a 4-site PBC MPS used in arXiv:1512.04938v2.
    '''
    L_list = [None] * len(psi)
    L = yastn.eye(config=psi[0].config,legs=psi[0].get_legs(axes=0), isdiag=False)
    err = [np.inf]
    num_step, pos = 0, 0
    L_list[pos] = L
    for i in range(max_sweeps):
        next = (pos+1)%len(psi)
        L = QR_L(L, psi[pos])
        L_prev, L_list[next] = L_list[next], L
        if L_prev and same_sectors(L, L_prev):
            err.append((L - L_prev).norm())

        pos = next
        num_step += 1

        if conv_check(err):
            break

    return L_list, err[-1]

def find_R(psi, max_sweeps, conv_check):
    '''
    Compute the R tensors for a 4-site PBC MPS used in arXiv:1512.04938v2.
    '''
    R_list = [None] * len(psi)
    R = yastn.eye(config=psi[-1].config,legs=psi[-1].get_legs(axes=3).conj(), isdiag=False)
    err = [np.inf]
    N = len(psi)
    num_step, pos = 0, N-1
    R_list[pos] = R
    for i in range(max_sweeps):
        next = (pos-1)%N
        R = QR_R(R, psi[pos])
        R_prev, R_list[next] = R_list[next], R
        if R_prev and same_sectors(R, R_prev):
            err.append((R - R_prev).norm())

        pos = next
        num_step += 1

        if conv_check(err):
            break
    return R_list, err[-1]

def get_proj(L, R):
    u, s, v = (L@R).svd(axes=(0, 1), sU=L.s[1], policy='fullrank')
    s_inv = s.rsqrt(cutoff=1e-12)
    PR = R @ v.T.conj() @ s_inv
    PL = s_inv @ u.T.conj() @ L
    return PR, PL

def projectors(psi, max_sweeps, conv_check):
    L_list, L_err = find_L(psi, max_sweeps, conv_check)
    R_list, R_err = find_R(psi, max_sweeps, conv_check)
    # print("Entanglement filtering convergece error: ", L_err, R_err)
    PLs, PRs = [None]*len(L_list), [None]*len(R_list)
    N = len(L_list)
    for i in range(N):
        PR, PL = get_proj(L_list[i], R_list[(i-1)%N])
        PLs[i] = PL
        PRs[(i-1)%N] = PR
    return PLs, PRs

def TM_psiA_psiA(psi_As, pos=None):
    if pos is None:
        TM_AA = []
        for i in range(len(psi_As)):
            TM_AA.append(psi_As[i].tensordot(psi_As[i], axes=((1, 2), (1, 2)), conj=(0, 1)))
            TM_AA[-1] = TM_AA[-1].transpose(axes=(0, 2, 1, 3))
            # 1---psi_A[i]^*---3
            #      /\
            #      \/
            # 0---psi_A[i]-----2
        return TM_AA
    else:
        TM_AA = psi_As[pos].tensordot(psi_As[pos], axes=((1, 2), (1, 2)), conj=(0, 1))
        return TM_AA.transpose(axes=(0, 2, 1, 3))


def TM_psiB_psiB(psi_Bs, pos=None):
    if pos is None:
        TM_BB = []
        for i in range(len(psi_Bs)):
            TM_BB.append(psi_Bs[i].tensordot(psi_Bs[i], axes=(1, 1), conj=(0, 1)))
            TM_BB[-1] = TM_BB[-1].transpose(axes=(0, 2, 1, 3))
            # 1---psi_B[i]^*---3
            #       |
            #       |
            # 0---psi_B[i]-----2
        return TM_BB
    else:
        TM_BB = psi_Bs[pos].tensordot(psi_Bs[pos], axes=(1, 1), conj=(0, 1))
        return TM_BB.transpose(axes=(0, 2, 1, 3))

def TM_psiA_psi_B(psi_As, psi_Bs, pos=None):
    if pos is None:
        TM_AB = []
        for i in range(len(psi_As)):
            BB = psi_Bs[2*i]@psi_Bs[2*i+1]
            TM_AB.append(psi_As[i].tensordot(BB, axes=((1, 2), (1, 2)), conj=(0, 1)))
            TM_AB[-1] = TM_AB[-1].transpose(axes=(0, 2, 1, 3))
            # 1---psi_B[2*i]^*--psi_B[2*i+1]^*---3
            #             \      /
            #              \    /
            #               \  /
            # 0-------------psi_A[i]-----------2
        return TM_AB
    else:
        BB = psi_Bs[2*pos]@psi_Bs[2*pos+1]
        TM_AB = psi_As[pos].tensordot(BB, axes=((1, 2), (1, 2)), conj=(0, 1))
        return TM_AB.transpose(axes=(0, 2, 1, 3))

def id_R(T):
    legs = T.get_legs(axes=(2, 3))
    res = yastn.tensordot(yastn.eye(config=T.config, legs=(legs[0].conj(), legs[0]), isdiag=False),\
                           yastn.eye(config=T.config, legs=(legs[1].conj(), legs[1]), isdiag=False), axes=((), ()))
    return res

def id_L(T):
    legs = T.get_legs(axes=(0, 1))
    res = yastn.tensordot(yastn.eye(config=T.config, legs=(legs[0], legs[0].conj()), isdiag=False),\
                        yastn.eye(config=T.config, legs=(legs[1], legs[1].conj()), isdiag=False), axes=((), ()))
    return res

def env_L_list(TM, start, pos, env_Ls=[]):
    r"""
    Compute the left effective environment from TM, from position `start` to position `pos`.
    If `env_Ls` is provided, it reuses the data of `env_Ls`.
    """
    if len(env_Ls) == 0:
        assert start == 0, "Starting position must be 0 when env_Ls is empty!"
        env_Ls = [None] * len(TM)
        env_Ls[0] = id_L(TM[0])

    for i in range(start, pos):
        res = env_Ls[i].tensordot(TM[i], axes=((2, 3), (0, 1)))
        env_Ls[i+1] = res

    return env_Ls

def env_R_list(TM, start, pos, env_Rs=[]):
    r"""
    Compute the left effective environment from TM, from position `start` to position `pos`.
    If `env_Rs` is provided, it reuses the data of `env_Rs`.
    """

    if len(env_Rs) == 0:
        assert start == len(TM)-1, "Starting position must be len(TM)-1 when env_Rs is empty!"
        env_Rs = [None] * len(TM)
        env_Rs[-1] = id_R(TM[-1])
    for i in range(start, pos, -1):
        res = TM[i].tensordot(env_Rs[i], axes=((2, 3), (0, 1)))
        env_Rs[i-1] = res

    return env_Rs

def NT(env_L, env_R, Bi):
    # compute the action of N on psi_Bs
    #   /----------------------\
    #   |--\ /---0 1 2----\ /--|
    #       L      |       R
    #   |--/ \-----Bi-----/ \--|
    #   \----------------------/

    res = env_L.tensordot(Bi, axes=(2, 0))
    res = res.tensordot(env_R, axes=((0, 1, 4), (2, 3, 0)))
    return res

def W(pos, env_L, env_R, psi_As, psi_Bs):
    # pos: position of the B tensor in the 8-site MPS
    # compute W tensor:

    # odd positions:
    #   /-----------------------------\
    #   |--\ /--B_{i-1}-0 1 2----\ /--|
    #       L        \   /        R
    #   |--/ \--------A_{i//2}---/ \--|
    #   \-----------------------------/

    # even positions:
    #   /------------------------------\
    #   |--\ /---0 1 2-B_{i+1}----\ /--|
    #       L       \    /         R
    #   |--/ \--------A_{i//2}----/ \--|
    #   \------------------------------/

    if pos % 2 == 1:
        res = env_L.tensordot(psi_Bs[pos-1], axes=(3, 0), conj=(0, 1))
        res = res.tensordot(psi_As[pos//2], axes=((2, 3), (0, 1)))
        res = res.tensordot(env_R, axes=((0, 1, 4), (2, 3, 0)))
    else:
        res = psi_Bs[pos+1].tensordot(env_R, axes=(2, 1), conj=(1, 0))
        res = psi_As[pos//2].tensordot(res, axes=((2, 3), (1, 2)))
        res = env_L.tensordot(res, axes=((0, 1, 2), (3, 4, 0)))

    return res

def solve_T(NT, W, T0):
    r"""
    Solve the linear system for T in the equation NT(T) = W.
    """

    v0, meta_T = T0.compress_to_1d(meta=None)
    W_data, meta_W = W.compress_to_1d(meta=None)

    to_tensor= lambda x: T0.config.backend.to_tensor(x if np.sum(np.array(x.strides)<0)==0 else x.copy() , dtype=T0.yastn_dtype, device=T0.device)
    to_numpy= lambda x: T0.config.backend.to_numpy(x)

    def mv(v):
        T = decompress_from_1d(to_tensor(v), meta_T)
        res = NT(T)
        res_data, _= res.compress_to_1d(meta=meta_W)
        return to_numpy(res_data)
    v0 = to_numpy(v0)

    A = LinearOperator((v0.size, v0.size), matvec=mv)
    sol, info = cg(A, to_numpy(W_data), x0=v0)
    if info != 0:
        print(f"info={info}")
    return decompress_from_1d(to_tensor(sol), meta_T)

def get_norm(TM):
    res = id_L(TM[0])
    for i in range(len(TM)):
        res = res.tensordot(TM[i], axes=((2, 3), (0, 1)))
    return yastn.trace(res, axes=((0, 1), (2, 3)))

class LoopTNR(Peps):
    def __init__(self, psi, D_max):
        r"""
        Object that contains the data used in Loop-TNR algorithm.

        Parameters
        ----------
        psi: yastn.tn.Peps
            PEPS lattice representing the 2D partition function.
            If ``psi`` has physical legs, a double-layer PEPS with no physical legs is formed.
        """
        assert psi.geometry.dims[0] == psi.geometry.dims[1], "LoopTNR only supports square lattices."
        super().__init__(psi.geometry)
        self.D_max = D_max
        self.site_rg_map = {}
        self.psi = Peps2Layers(psi) if psi.has_physical() else psi
        self.psi_rg = Peps(self.geometry)
        # self.init_psi_rg(D_max)

        # dict for converting leg orders between decomposed tensors and psi_Bs
        self.decomp_T_to_B = {
            0: (1, 0, 2), 1: (0, 1, 2),
            2: (1, 0, 2), 3: (0, 2, 1),
            4: (1, 2, 0), 5: (2, 1, 0),
            6: (2, 1, 0), 7: (2, 1, 0)
        }
        self.B_to_decomp_T = {
            0: (1, 0, 2), 1: (0, 1, 2),
            2: (1, 0, 2), 3: (0, 2, 1),
            4: (2, 0, 1), 5: (2, 1, 0),
            6: (2, 1, 0), 7: (2, 1, 0)
        }

        # dict for mapping positions in psi_Bs to the directions in decomposed tensors
        self.Bpos_to_dirn = {
            0: "tr", 1: "bl", 2: "br", 3: "tl",
            4: "bl", 5: "tr", 6: "tl", 7: "br",
        }

    def init_psi_rg(self, D_max):
        '''
        Initialize decomposed tensors by performing truncated SVD (Levin-Nave RG).
        '''
        for site in self.psi_rg.sites():
            self.psi_rg[site] = Decomp_T()
        self.site_rg_map = {}

        site = self.sites()[0]
        site_rg = self.sites()[0]
        for _ in range(self.dims[0]):
            for _ in range(self.dims[1]):
                tl, tr, bl, br = self.site2index(site), self.site2index(self.nn_site(site, "r")), self.site2index(self.nn_site(site, "b")), self.site2index(self.nn_site(site, "br"))

                if self.psi_rg[site_rg].tl is None:
                    if (tl, "br") in self.site_rg_map:
                        self.psi_rg[site_rg].tl = self.psi_rg[self.site_rg_map[(tl, "br")][1]].tl
                    else:
                        u, s, v = self.psi[tl].svd_with_truncation(axes=((0, 1), (2, 3)), sU=1, policy='fullrank',
                                                                    D_total=D_max, truncate_multiplets=True)
                        site_rg_t = self.site2index(self.psi_rg.nn_site(site_rg, "t"))
                        self.psi_rg[site_rg_t].br = u@s.sqrt()
                        self.psi_rg[site_rg].tl = s.sqrt()@v
                        self.site_rg_map[(tl, "br")] = (site_rg_t, site_rg)

                if self.psi_rg[site_rg].tr is None:
                    if (tr, "tr") in self.site_rg_map:
                        self.psi_rg[site_rg].tr = self.psi_rg[self.site_rg_map[(tr, "tr")][0]].tr
                    else:
                        u, s, v = self.psi[tr].svd_with_truncation(axes=((1, 2), (0, 3)), sU=1, policy='fullrank',
                                                                    D_total=D_max, truncate_multiplets=True)
                        site_rg_r = self.site2index(self.psi_rg.nn_site(site_rg, "r"))
                        self.psi_rg[site_rg].tr = u@s.sqrt()
                        self.psi_rg[site_rg_r].bl = s.sqrt()@v
                        self.site_rg_map[(tr, "tr")] = (site_rg, site_rg_r)

                if self.psi_rg[site_rg].br is None:
                    if (br, "br") in self.site_rg_map:
                        self.psi_rg[site_rg].br = self.psi_rg[self.site_rg_map[(br, "br")][0]].br
                    else:
                        u, s, v = self.psi[br].svd_with_truncation(axes=((0, 1), (2, 3)), sU=1, policy='fullrank',
                                                                    D_total=D_max, truncate_multiplets=True)
                        site_rg_b = self.site2index(self.psi_rg.nn_site(site_rg, "b"))
                        self.psi_rg[site_rg].br = u@s.sqrt()
                        self.psi_rg[site_rg_b].tl = s.sqrt()@v
                        self.site_rg_map[(br, "br")] = (site_rg, site_rg_b)

                if self.psi_rg[site_rg].bl is None:
                    if (bl, "tr") in self.site_rg_map:
                        self.psi_rg[site_rg].bl = self.psi_rg[self.site_rg_map[(bl, "tr")][1]].bl
                    else:
                        u, s, v = self.psi[bl].svd_with_truncation(axes=((1, 2), (0, 3)), sU=1, policy='fullrank',
                                                                    D_total=D_max, truncate_multiplets=True)
                        site_rg_l = self.site2index(self.psi_rg.nn_site(site_rg, "l"))
                        self.psi_rg[site_rg_l].tr = u@s.sqrt()
                        self.psi_rg[site_rg].bl = s.sqrt()@v
                        self.site_rg_map[(bl, "tr")] = (site_rg_l, site_rg)

                site = self.nn_site(site, (-1, 1))
                site_rg = self.nn_site(site_rg, (0, 1))

            site = self.nn_site(site, (1, 1))
            site_rg = self.nn_site(site_rg, (1, 0))


    def entanglement_filtering(self, site_tl, max_sweeps, conv_check):
        r"""
        Perform entanglement-filtering for the plaquette specified by site_tl (top-left corner).
        """
        if max_sweeps > 0:
            psi_As = self.psiA_mps(site_tl)
            PLs, PRs = projectors(psi_As, max_sweeps, conv_check)
            for i in range(len(psi_As)):
                psi_As[i] = PLs[i]@psi_As[i]@PRs[i]

            # update PEPS tensors
            site_tr, site_bl, site_br = self.nn_site(site_tl, "r"), self.nn_site(site_tl, "b"), self.nn_site(site_tl, "br")
            self.psi[site_tl] = psi_As[0].transpose(axes=(2, 1, 0, 3))
            self.psi[site_tr] = psi_As[1].transpose(axes=(1, 0, 3, 2))
            self.psi[site_br] = psi_As[2].transpose(axes=(0, 3, 2, 1))
            self.psi[site_bl] = psi_As[3].transpose(axes=(3, 2, 1, 0))

    def psiA_mps(self, site_tl):
        '''
        Construct the periodic 4-site MPS (clockwisely) from the square lattice network.
                    |        |
                ---T0(tl)---T1(tr)-
                    |        |
                    |        |
                ---T2(bl)---T3(br)-
                    |        |
        '''
        site_tr, site_bl, site_br = self.nn_site(site_tl, "r"), self.nn_site(site_tl, "b"), self.nn_site(site_tl, "br")
        psi_As = [self.psi[site_tl].transpose(axes=(2, 1, 0, 3)),
                self.psi[site_tr].transpose(axes=(1, 0, 3, 2)),
                self.psi[site_br].transpose(axes=(0, 3, 2, 1)),
                self.psi[site_bl].transpose(axes=(3, 2, 1, 0))
                ]
        return psi_As

    def psiB_mps(self, site_tl):
        '''
        Construct the periodic 8-site MPS (clockwisely) from the square lattice network.

                     |     |
                     T1----T2
                    /       \                       1
                ---T0        T3---                  |
                   |         |          ,       0---Ti---2
                ---T7        T4---
                    \       /
                     T6----T5
                     |     |
        '''
        if (self.site2index(site_tl), "tr") not in self.site_rg_map:
            return None

        psi_Bs = [None] * 8
        site_tr, site_bl, site_br = self.nn_site(site_tl, "r"), self.nn_site(site_tl, "b"), self.nn_site(site_tl, "br")
        site_rg1, site_rg2 = self.site_rg_map[(self.site2index(site_tl), "tr")]
        psi_Bs[0] = self.psi_rg[site_rg1].tr.transpose(axes=self.decomp_T_to_B[0])
        psi_Bs[1] = self.psi_rg[site_rg2].bl.transpose(axes=self.decomp_T_to_B[1])
        site_rg1, site_rg2 = self.site_rg_map[(self.site2index(site_tr), "br")]
        psi_Bs[2] = self.psi_rg[site_rg1].br.transpose(axes=self.decomp_T_to_B[2])
        psi_Bs[3] = self.psi_rg[site_rg2].tl.transpose(axes=self.decomp_T_to_B[3])
        site_rg1, site_rg2 = self.site_rg_map[(self.site2index(site_br), "tr")]
        psi_Bs[4] = self.psi_rg[site_rg2].bl.transpose(axes=self.decomp_T_to_B[4])
        psi_Bs[5] = self.psi_rg[site_rg1].tr.transpose(axes=self.decomp_T_to_B[5])
        site_rg1, site_rg2 = self.site_rg_map[(self.site2index(site_bl), "br")]
        psi_Bs[6] = self.psi_rg[site_rg2].tl.transpose(axes=self.decomp_T_to_B[6])
        psi_Bs[7] = self.psi_rg[site_rg1].br.transpose(axes=self.decomp_T_to_B[7])

        return psi_Bs

    def loop_optimize(self, max_sweeps, threshold=1e-6):
        # initialization
        self.init_psi_rg(self.D_max)
        psi_A_dict, psi_B_dict = {}, {}
        TM_AB_dict, TM_BB_dict = {}, {}
        env_AB_L_dict, env_AB_R_dict = {}, {}
        env_BB_L_dict, env_BB_R_dict = {}, {}
        env_L_start, env_R_start = {}, {}
        AA_norm, truncate_err = {}, {}

        # Due to the choice of init_psi_rg, the starting site of the loop is shifted.
        site = self.nn_site(self.sites()[0], 'b')
        for _ in range(self.dims[0]):
            for _ in range(self.dims[1]):
                site = self.site2index(site)
                psi_A_dict[site] = self.psiA_mps(site)
                psi_B_dict[site] = self.psiB_mps(site)
                AA_norm[site] = get_norm(TM_psiA_psiA(psi_A_dict[site]))
                TM_AB_dict[site] = TM_psiA_psi_B(psi_A_dict[site], psi_B_dict[site])
                TM_BB_dict[site] = TM_psiB_psiB(psi_B_dict[site])
                env_AB_L_dict[site], env_AB_R_dict[site] = [], []
                env_BB_L_dict[site], env_BB_R_dict[site] = [], []
                env_L_start[site], env_R_start[site] = 0, 7
                truncate_err[site] = np.inf
                site = self.nn_site(site, (-1, 1))
            site = self.nn_site(site, (1, 1))


        drs = [(-1, -1), (-1, 0), (0, 0), (0, -1)] # vectors pointing to the the top-left corner of the edge-sharing plaquettes
        sp_optim_pos = [(5, 4), (7, 6), (1, 0), (3, 2)] # indices of the optimized sites in the edge-sharing psi_Bs

        for step in range(max_sweeps):
            converged = True
            # loop over plaquettes (not rg lattice sites)
            site = self.nn_site(self.sites()[0], 'b')
            for _ in range(self.dims[0]):
                for _ in range(self.dims[1]):
                    loop_sites = self.site2index(site), self.site2index(self.nn_site(site, "r")), self.site2index(self.nn_site(site, "br")), self.site2index(self.nn_site(site, "b"))

                    tl = loop_sites[0]
                    psi_A1, psi_B1 = psi_A_dict[tl], psi_B_dict[tl]
                    B1_pos = 0
                    for i, s in enumerate(loop_sites):
                        dr = drs[i]
                        optim_pos = sp_optim_pos[i]
                        sp = self.site2index(self.nn_site(s, dr))
                        psi_A2, psi_B2 = psi_A_dict[sp], psi_B_dict[sp]
                        for j, B2_pos in enumerate(optim_pos):
                            env_AB_L_dict[tl] = env_L_list(TM_AB_dict[tl], env_L_start[tl]//2, B1_pos//2, env_Ls=env_AB_L_dict[tl])
                            env_AB_R_dict[tl] = env_R_list(TM_AB_dict[tl], env_R_start[tl]//2, B1_pos//2, env_Rs=env_AB_R_dict[tl])
                            env_BB_L_dict[tl] = env_L_list(TM_BB_dict[tl], env_L_start[tl], B1_pos, env_Ls=env_BB_L_dict[tl])
                            env_BB_R_dict[tl] = env_R_list(TM_BB_dict[tl], env_R_start[tl], B1_pos, env_Rs=env_BB_R_dict[tl])

                            env_AB_L_dict[sp] = env_L_list(TM_AB_dict[sp], env_L_start[sp]//2, B2_pos//2, env_Ls=env_AB_L_dict[sp])
                            env_AB_R_dict[sp] = env_R_list(TM_AB_dict[sp], env_R_start[sp]//2, B2_pos//2, env_Rs=env_AB_R_dict[sp])
                            env_BB_L_dict[sp] = env_L_list(TM_BB_dict[sp], env_L_start[sp], B2_pos, env_Ls=env_BB_L_dict[sp])
                            env_BB_R_dict[sp] = env_R_list(TM_BB_dict[sp], env_R_start[sp], B2_pos, env_Rs=env_BB_R_dict[sp])

                            W1 = W(B1_pos, env_AB_L_dict[tl][B1_pos//2], env_AB_R_dict[tl][B1_pos//2], psi_A1, psi_B1).transpose(self.B_to_decomp_T[B1_pos])
                            W2 = W(B2_pos, env_AB_L_dict[sp][B2_pos//2], env_AB_R_dict[sp][B2_pos//2], psi_A2, psi_B2).transpose(self.B_to_decomp_T[B2_pos])

                            def N12(T):
                                NT1 = NT(env_BB_L_dict[tl][B1_pos], env_BB_R_dict[tl][B1_pos], T.transpose(self.decomp_T_to_B[B1_pos])).transpose(self.B_to_decomp_T[B1_pos])
                                NT2 = NT(env_BB_L_dict[sp][B2_pos], env_BB_R_dict[sp][B2_pos], T.transpose(self.decomp_T_to_B[B2_pos])).transpose(self.B_to_decomp_T[B2_pos])
                                return NT1+NT2

                            # solve the linear system for T
                            dirn = "tr" if i % 2 == 0 else "br"
                            site_rg = self.site_rg_map[(s, dirn)][j] if B1_pos <= 3 else self.site_rg_map[(s, dirn)][1-j]
                            T0 = getattr(self.psi_rg[site_rg], self.Bpos_to_dirn[B1_pos])
                            T = solve_T(N12, W1+W2, T0)

                            # update T, TM, env_start
                            setattr(self.psi_rg[site_rg], self.Bpos_to_dirn[B1_pos], T)
                            psi_B1[B1_pos] = T.transpose(self.decomp_T_to_B[B1_pos])
                            psi_B2[B2_pos] = T.transpose(self.decomp_T_to_B[B2_pos])
                            TM_AB_dict[tl][B1_pos//2] = TM_psiA_psi_B(psi_A1, psi_B1, pos=B1_pos//2)
                            TM_BB_dict[tl][B1_pos] = TM_psiB_psiB(psi_B1, pos=B1_pos)
                            TM_AB_dict[sp][B2_pos//2] = TM_psiA_psi_B(psi_A2, psi_B2, pos=B2_pos//2)
                            TM_BB_dict[sp][B2_pos] = TM_psiB_psiB(psi_B2, pos=B2_pos)
                            env_L_start[tl], env_R_start[tl] = B1_pos, B1_pos
                            env_L_start[sp], env_R_start[sp] = B2_pos, B2_pos

                            B1_pos += 1
                    BB_norm1 = TM_BB_dict[tl][7].tensordot(env_BB_L_dict[tl][7], axes=((0, 1, 2, 3), (2, 3, 0, 1)))
                    AB_norm1 = TM_AB_dict[tl][3].tensordot(env_AB_L_dict[tl][3], axes=((0, 1, 2, 3), (2, 3, 0, 1)))
                    err = (AA_norm[tl] + BB_norm1 - AB_norm1 - AB_norm1.conj()).to_number()/ AA_norm[tl].to_number()
                    converged = (converged and (np.abs(err - truncate_err[tl])) < threshold)
                    truncate_err[tl] = err.real
                    # print(f"iteration {step}, top-left corner {tl}: truncation error={truncate_err[tl]}")
                    site = self.nn_site(site, (-1, 1))
                site = self.nn_site(site, (1, 1))

            if converged:
                break

            return truncate_err


    def rg(self, max_sweeps, filter_max_sweeps=1000, filter_threshold=1e-6, loop_threshold=1e-6):
        r"""
        Perform the Loop-TNR RG procedure on the PEPS lattice.
        """

        for site in self.psi.sites():
            self.entanglement_filtering(site, filter_max_sweeps, conv_check=lambda err: err[-1] < filter_threshold)


        # loop compression
        truncate_err = self.loop_optimize(max_sweeps, loop_threshold)
        max_truncate_err = max(truncate_err.values())

        # assemble loop MPSs into new PEPS tensors
        self.sync_decomp_T_()
        site_rg = self.sites()[0]
        for _ in range(self.dims[0]):
            for _ in range(self.dims[1]):
                L = yastn.tensordot(self.psi_rg[site_rg].tl, self.psi_rg[site_rg].bl, axes=(1, 1))
                R = yastn.tensordot(self.psi_rg[site_rg].br, self.psi_rg[site_rg].tr, axes=(0, 1))
                self.psi[site_rg] = yastn.tensordot(L, R, axes=((1, 3), (2, 0)))
                # self.psi[site_rg] = self.psi[site_rg]/self.psi[site_rg].norm(p='inf')
                site_rg = self.nn_site(site_rg, (0, 1))
            site_rg = self.nn_site(site_rg, (1, 0))
        return max_truncate_err

    def sync_decomp_T_(self):
        site = self.sites()[0]
        site_rg = self.sites()[0]
        for _ in range(self.dims[0]):
            for _ in range(self.dims[1]):
                tl, tr, bl, br = self.site2index(site), self.site2index(self.nn_site(site, "r")), self.site2index(self.nn_site(site, "b")), self.site2index(self.nn_site(site, "br"))
                self.psi_rg[site_rg].tl = self.psi_rg[self.site_rg_map[(tl, "br")][1]].tl
                self.psi_rg[site_rg].tr = self.psi_rg[self.site_rg_map[(tr, "tr")][0]].tr
                self.psi_rg[site_rg].br = self.psi_rg[self.site_rg_map[(br, "br")][0]].br
                self.psi_rg[site_rg].bl = self.psi_rg[self.site_rg_map[(bl, "tr")][1]].bl

                site = self.nn_site(site, (-1, 1))
                site_rg = self.nn_site(site_rg, (0, 1))

            site = self.nn_site(site, (1, 1))
            site_rg = self.nn_site(site_rg, (1, 0))




if __name__ == "__main__":
    beta_c = Ising_critical_beta()
    psi = Ising_dense(beta=beta_c, h=0.0)
    D_max = 8
    max_sweeps = 20
    loop_tnr = LoopTNR(psi, D_max)

    for step in range(20):
        truncate_err = loop_tnr.rg(max_sweeps, filter_max_sweeps=5000, filter_threshold=1e-5, loop_threshold=1e-6)
        print(f"RG Step {step+1}, max truncation error: {truncate_err}")

