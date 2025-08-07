import numpy as np
import torch
from torch.optim import LBFGS

import logging, argparse, pickle
from dataclasses import dataclass
from typing import Union

import yastn.yastn as yastn
from yastn.yastn.tensor import Tensor
from yastn.yastn.tn.fpeps._peps import Peps, Peps2Layers

from Ising import *
from ALS_utility import *

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

def nuclear_norm(A):
    """
    Compute the nuclear norm of the matrix A, defined as the sum of the singular values.
    """
    _, S, _ = A.svd(compute_uv=True, policy='fullrank')
    return S.trace().to_number()


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

    def indep_decomp_T(self):
        '''
        Return a generator that yields independent decomp_T tensors.
        '''
        marked = set()
        site = self.sites()[0]
        for _ in range(self.dims[0]):
            for _ in range(self.dims[1]):
                tl, tr, bl, br = self.site2index(site), self.site2index(self.nn_site(site, "r")), self.site2index(self.nn_site(site, "b")), self.site2index(self.nn_site(site, "br"))
                if (tl, "br") not in marked:
                    marked.add((tl, "br"))
                    yield self.psi_rg[self.site_rg_map[(tl, "br")][1]].tl
                if (tr, "tr") not in marked:
                    marked.add((tr, "tr"))
                    yield self.psi_rg[self.site_rg_map[(tr, "tr")][0]].tr
                if (br, "br") not in marked:
                    marked.add((br, "br"))
                    yield self.psi_rg[self.site_rg_map[(br, "br")][0]].br
                if (bl, "tr") not in marked:
                    marked.add((bl, "tr"))
                    yield self.psi_rg[self.site_rg_map[(bl, "tr")][1]].bl
                site = self.nn_site(site, (-1, 1))
            site = self.nn_site(site, (1, 1))

    def get_decomp_T_data(self):
        return list(T._data for T in self.indep_decomp_T())

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

    @torch.no_grad()
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
                tl, tr, bl, br = site, self.nn_site(site, "r"), self.nn_site(site, "b"), self.nn_site(site, "br")
                tl_ind, tr_ind, bl_ind, br_ind = self.site2index(tl), self.site2index(tr), self.site2index(bl), self.site2index(br)
                if self.psi_rg[site_rg].tl is None:
                    if (tl_ind, "br") in self.site_rg_map:
                        self.psi_rg[site_rg].tl = self.psi_rg[self.site_rg_map[(tl_ind, "br")][1]].tl
                    else:
                        u, s, v = self.psi[tl].svd_with_truncation(axes=((0, 1), (2, 3)), sU=1, policy='fullrank',
                                                                    tol=1e-12, D_total=D_max, truncate_multiplets=True)
                        site_rg_t = self.psi_rg.nn_site(site_rg, "t")
                        self.psi_rg[site_rg_t].br = u@s.sqrt()
                        self.psi_rg[site_rg].tl = s.sqrt()@v
                        self.site_rg_map[(tl_ind, "br")] = (site_rg_t, site_rg)

                if self.psi_rg[site_rg].tr is None:
                    if (tr_ind, "tr") in self.site_rg_map:
                        self.psi_rg[site_rg].tr = self.psi_rg[self.site_rg_map[(tr_ind, "tr")][0]].tr
                    else:
                        u, s, v = self.psi[tr].svd_with_truncation(axes=((1, 2), (0, 3)), sU=1, policy='fullrank',
                                                                    tol=1e-12, D_total=D_max, truncate_multiplets=True)
                        site_rg_r = self.psi_rg.nn_site(site_rg, "r")
                        self.psi_rg[site_rg].tr = u@s.sqrt()
                        self.psi_rg[site_rg_r].bl = s.sqrt()@v
                        self.site_rg_map[(tr_ind, "tr")] = (site_rg, site_rg_r)

                if self.psi_rg[site_rg].br is None:
                    if (br_ind, "br") in self.site_rg_map:
                        self.psi_rg[site_rg].br = self.psi_rg[self.site_rg_map[(br_ind, "br")][0]].br
                    else:
                        u, s, v = self.psi[br].svd_with_truncation(axes=((0, 1), (2, 3)), sU=1, policy='fullrank',
                                                                    tol=1e-12, D_total=D_max, truncate_multiplets=True)
                        site_rg_b = self.psi_rg.nn_site(site_rg, "b")
                        self.psi_rg[site_rg].br = u@s.sqrt()
                        self.psi_rg[site_rg_b].tl = s.sqrt()@v
                        self.site_rg_map[(br_ind, "br")] = (site_rg, site_rg_b)

                if self.psi_rg[site_rg].bl is None:
                    if (bl_ind, "tr") in self.site_rg_map:
                        self.psi_rg[site_rg].bl = self.psi_rg[self.site_rg_map[(bl_ind, "tr")][1]].bl
                    else:
                        u, s, v = self.psi[bl].svd_with_truncation(axes=((1, 2), (0, 3)), sU=1, policy='fullrank',
                                                                    tol=1e-12, D_total=D_max, truncate_multiplets=True)
                        site_rg_l = self.psi_rg.nn_site(site_rg, "l")
                        self.psi_rg[site_rg_l].tr = u@s.sqrt()
                        self.psi_rg[site_rg].bl = s.sqrt()@v
                        self.site_rg_map[(bl_ind, "tr")] = (site_rg_l, site_rg)

                site = self.nn_site(site, (-1, 1))
                site_rg = self.nn_site(site_rg, (0, 1))

            site = self.nn_site(site, (1, 1))
            site_rg = self.nn_site(site_rg, (1, 0))


    def entanglement_filtering(self, site_tl, max_sweeps, conv_check):
        r"""
        Perform entanglement-filtering for the plaquette specified by site_tl (top-left corner).
        """
        sites = (site_tl, self.nn_site(site_tl, "r"), self.nn_site(site_tl, "br"), self.nn_site(site_tl, "b"))
        transpose_order = [(2, 1, 0, 3), (1, 0, 3, 2), (0, 3, 2, 1), (3, 2, 1, 0)]
        if max_sweeps > 0:
            psi_As = self.psiA_mps(site_tl)
            PLs, PRs = projectors(psi_As, max_sweeps, conv_check)

            # update PEPS tensors
            for i in range(len(PLs)):
                psi_Ai = self.psi[sites[i]].transpose(transpose_order[i]) # to mps
                psi_Ai = PLs[i]@psi_Ai@PRs[i] # projection
                self.psi[sites[i]] = psi_Ai.transpose(transpose_order[i]) # back to peps

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

    @torch.no_grad()
    def loop_optimize(self, max_sweeps, threshold=1e-6):
        """
        Perform alternating-least-squares optimization to compress the periodic mps.
        """
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
                ind = self.site2index(site)
                psi_A_dict[ind] = self.psiA_mps(site)
                psi_B_dict[ind] = self.psiB_mps(site)
                AA_norm[ind] = get_norm(TM_psiA_psiA(psi_A_dict[ind]))
                TM_AB_dict[ind] = TM_psiA_psi_B(psi_A_dict[ind], psi_B_dict[ind])
                TM_BB_dict[ind] = TM_psiB_psiB(psi_B_dict[ind])
                env_AB_L_dict[ind], env_AB_R_dict[ind] = [], []
                env_BB_L_dict[ind], env_BB_R_dict[ind] = [], []
                env_L_start[ind], env_R_start[ind] = 0, 7
                truncate_err[ind] = np.inf
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
                    loop_sites = site, self.nn_site(site, "r"), self.nn_site(site, "br"), self.nn_site(site, "b")

                    tl_ind = self.site2index(loop_sites[0])
                    psi_A1, psi_B1 = psi_A_dict[tl_ind], psi_B_dict[tl_ind]
                    B1_pos = 0
                    for i, s in enumerate(loop_sites):
                        dr = drs[i]
                        optim_pos = sp_optim_pos[i]
                        sp_ind = self.site2index(self.nn_site(s, dr))
                        psi_A2, psi_B2 = psi_A_dict[sp_ind], psi_B_dict[sp_ind]
                        for j, B2_pos in enumerate(optim_pos):
                            env_AB_L_dict[tl_ind] = env_L_list(TM_AB_dict[tl_ind], env_L_start[tl_ind]//2, B1_pos//2, env_Ls=env_AB_L_dict[tl_ind])
                            env_AB_R_dict[tl_ind] = env_R_list(TM_AB_dict[tl_ind], env_R_start[tl_ind]//2, B1_pos//2, env_Rs=env_AB_R_dict[tl_ind])
                            env_BB_L_dict[tl_ind] = env_L_list(TM_BB_dict[tl_ind], env_L_start[tl_ind], B1_pos, env_Ls=env_BB_L_dict[tl_ind])
                            env_BB_R_dict[tl_ind] = env_R_list(TM_BB_dict[tl_ind], env_R_start[tl_ind], B1_pos, env_Rs=env_BB_R_dict[tl_ind])
                            W1 = W(B1_pos, env_AB_L_dict[tl_ind][B1_pos//2], env_AB_R_dict[tl_ind][B1_pos//2], psi_A1, psi_B1).transpose(self.B_to_decomp_T[B1_pos])
                            env_L_start[tl_ind], env_R_start[tl_ind] = max(B1_pos, env_L_start[tl_ind]), min(B1_pos, env_R_start[tl_ind])

                            env_AB_L_dict[sp_ind] = env_L_list(TM_AB_dict[sp_ind], env_L_start[sp_ind]//2, B2_pos//2, env_Ls=env_AB_L_dict[sp_ind])
                            env_AB_R_dict[sp_ind] = env_R_list(TM_AB_dict[sp_ind], env_R_start[sp_ind]//2, B2_pos//2, env_Rs=env_AB_R_dict[sp_ind])
                            env_BB_L_dict[sp_ind] = env_L_list(TM_BB_dict[sp_ind], env_L_start[sp_ind], B2_pos, env_Ls=env_BB_L_dict[sp_ind])
                            env_BB_R_dict[sp_ind] = env_R_list(TM_BB_dict[sp_ind], env_R_start[sp_ind], B2_pos, env_Rs=env_BB_R_dict[sp_ind])

                            W2 = W(B2_pos, env_AB_L_dict[sp_ind][B2_pos//2], env_AB_R_dict[sp_ind][B2_pos//2], psi_A2, psi_B2).transpose(self.B_to_decomp_T[B2_pos])
                            env_L_start[sp_ind], env_R_start[sp_ind] = max(B2_pos, env_L_start[sp_ind]), max(B2_pos, env_R_start[sp_ind])

                            def N12(T):
                                NT1 = NT(env_BB_L_dict[tl_ind][B1_pos], env_BB_R_dict[tl_ind][B1_pos], T.transpose(self.decomp_T_to_B[B1_pos])).transpose(self.B_to_decomp_T[B1_pos])
                                NT2 = NT(env_BB_L_dict[sp_ind][B2_pos], env_BB_R_dict[sp_ind][B2_pos], T.transpose(self.decomp_T_to_B[B2_pos])).transpose(self.B_to_decomp_T[B2_pos])
                                return NT1+NT2

                            # solve the linear system for T
                            dirn = "tr" if i % 2 == 0 else "br"
                            s_ind = self.site2index(s)
                            site_rg = self.site_rg_map[(s_ind, dirn)][j] if B1_pos <= 3 else self.site_rg_map[(s_ind, dirn)][1-j]
                            T0 = getattr(self.psi_rg[site_rg], self.Bpos_to_dirn[B1_pos])
                            T = solve_T(N12, W1+W2, T0)

                            # update T, TM, env_start
                            setattr(self.psi_rg[site_rg], self.Bpos_to_dirn[B1_pos], T)
                            psi_B1[B1_pos] = T.transpose(self.decomp_T_to_B[B1_pos])
                            psi_B2[B2_pos] = T.transpose(self.decomp_T_to_B[B2_pos])
                            TM_AB_dict[tl_ind][B1_pos//2] = TM_psiA_psi_B(psi_A1, psi_B1, pos=B1_pos//2)
                            TM_BB_dict[tl_ind][B1_pos] = TM_psiB_psiB(psi_B1, pos=B1_pos)
                            TM_AB_dict[sp_ind][B2_pos//2] = TM_psiA_psi_B(psi_A2, psi_B2, pos=B2_pos//2)
                            TM_BB_dict[sp_ind][B2_pos] = TM_psiB_psiB(psi_B2, pos=B2_pos)
                            env_L_start[tl_ind], env_R_start[tl_ind] = min(B1_pos, env_L_start[tl_ind]), max(B1_pos, env_R_start[tl_ind])
                            env_L_start[sp_ind], env_R_start[sp_ind] = min(B2_pos, env_L_start[sp_ind]), max(B2_pos, env_R_start[sp_ind])
                            B1_pos += 1

                    BB_norm1 = TM_BB_dict[tl_ind][7].tensordot(env_BB_L_dict[tl_ind][7], axes=((0, 1, 2, 3), (2, 3, 0, 1)))
                    AB_norm1 = TM_AB_dict[tl_ind][3].tensordot(env_AB_L_dict[tl_ind][3], axes=((0, 1, 2, 3), (2, 3, 0, 1)))
                    err = (AA_norm[tl_ind] + BB_norm1 - AB_norm1 - AB_norm1.conj()).to_number()/ AA_norm[tl_ind].to_number()
                    # print(f"step-{step:d} site: {tl_ind}: {err}")
                    converged = (converged and (np.abs(err - truncate_err[tl_ind]) < threshold))
                    truncate_err[tl_ind] = err.real
                    site = self.nn_site(site, (-1, 1))
                site = self.nn_site(site, (1, 1))

            if converged:
                break

        return truncate_err

    def nuclear_norm_loss(self):
        loss = 0.0
        cnt = 0
        for T in self.indep_decomp_T():
            loss += nuclear_norm(T.fuse_legs(axes=(0, (1, 2))))
            + nuclear_norm(T.fuse_legs(axes=(1, (2, 0))))
            + nuclear_norm(T.fuse_legs(axes=(2, (0, 1))))
            cnt += 3
        return loss/cnt

    def loop_optimize_AD(self, epochs, loop_threshold, mu=0.0):
        """
        Perform AD optimization to compress the periodic mps.
        mu is the hyper-parameter for nuclear-norm regularization.
        """
        data_list = self.get_decomp_T_data()
        for data in data_list: data.requires_grad_(True)
        optimizer = LBFGS(data_list,lr=1.0, max_iter=20, history_size=25, line_search_fn='strong_wolfe')

        psi_A_dict, AA_norm_dict = {}, {}
        overlap = 0.0

        def init():
            nonlocal psi_A_dict, AA_norm_dict
            site = self.nn_site(self.sites()[0], 'b')
            for _ in range(self.dims[0]):
                for _ in range(self.dims[1]):
                    ind = self.site2index(site)
                    psi_A = self.psiA_mps(site)
                    psi_A_dict[ind] = psi_A
                    AA_norm_dict[ind] = get_norm(TM_psiA_psiA(psi_A))
                    site = self.nn_site(site, (-1, 1))
                site = self.nn_site(site, (1, 1))

        def overlap_loss():
            nonlocal overlap
            site = self.nn_site(self.sites()[0], 'b')
            loss = 0.0
            marked = set()
            for _ in range(self.dims[0]):
                for _ in range(self.dims[1]):
                    ind = self.site2index(site)
                    if ind not in marked: # avoid repeated computations
                        psi_B = self.psiB_mps(site)
                        AA_norm = AA_norm_dict[ind]
                        AB_norm = get_norm(TM_psiA_psi_B(psi_A_dict[ind], psi_B))
                        BB_norm = get_norm(TM_psiB_psiB(psi_B))
                        loss += (AA_norm + BB_norm - AB_norm - AB_norm.conj()).to_number().real / AA_norm.to_number().real
                        marked.add(ind)
                    site = self.nn_site(site, (-1, 1))
                site = self.nn_site(site, (1, 1))
            overlap = loss/len(marked)
            return loss

        def closure():
            optimizer.zero_grad()
            loss = overlap_loss() + mu*self.nuclear_norm_loss()
            loss.backward()
            return loss

        init()
        loss_history = [torch.inf]
        for epoch in range(1, epochs+1):
            loss = optimizer.step(closure)
            # print(f"epoch-{epoch:d}: loss={loss}")
            if torch.isclose(loss, torch.zeros(1, dtype=loss.dtype)) or torch.abs(loss - loss_history[-1])/torch.abs(loss) < loop_threshold:
                break
            loss_history.append(loss)

        for data in data_list:
            data.detach_()
        return overlap

    def rg(self, max_sweeps, method="AD", filter_max_sweeps=1000, filter_threshold=1e-6, loop_threshold=1e-6, mu=1e-8):
        r"""
        Perform the Loop-TNR RG procedure on the PEPS lattice.

        Parameters
        ----------
        max_sweeps: int
            maximum number of sweeps in loop_optimize / maximum number of epochs in loop_optimize_AD
        method: str
            "AD" (automatic differentiation) or "ALS" (alternating least squares)
        filter_threshold: float
            convergence threshold of entanglement-filtering
        loop_threshold: float
            convergence threshold of loop optimizattion
        mu: float
            hyper-parameter for the nuclear-norm regularization
        """
        if method == "ALS":
            # marked = set()
            # for site in self.psi.sites():
            #     ind = self.psi.site2index(site)
            #     if ind not in marked:
            #         self.entanglement_filtering(site, filter_max_sweeps, conv_check=lambda err_prev, err: abs(err_prev - err) < filter_threshold)
            #         marked.add(ind)
            # loop compression
            truncate_err = self.loop_optimize(max_sweeps, loop_threshold)
            truncate_err = sum(truncate_err.values())/len(truncate_err)
            self.sync_decomp_T_()

        elif method == "AD":
            self.init_psi_rg(self.D_max)
            truncate_err = self.loop_optimize_AD(max_sweeps, loop_threshold, mu=mu)
        else:
            raise Exception(f"method {method} is not implemented!")

        # assemble loop MPSs into new PEPS tensors
        site_rg = self.sites()[0]
        for _ in range(self.dims[0]):
            for _ in range(self.dims[1]):
                L = yastn.tensordot(self.psi_rg[site_rg].tl, self.psi_rg[site_rg].bl, axes=(1, 1))
                R = yastn.tensordot(self.psi_rg[site_rg].br, self.psi_rg[site_rg].tr, axes=(0, 1))
                self.psi[site_rg] = yastn.tensordot(L, R, axes=((1, 3), (2, 0)))
                site_rg = self.nn_site(site_rg, (0, 1))
            site_rg = self.nn_site(site_rg, (1, 0))
        return truncate_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--D_max", type=int, default=8)
    parser.add_argument("--max_sweeps", type=int, default=100)
    parser.add_argument("--max_rg_steps", type=int, default=25)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--method", type=str, default="AD")
    parser.add_argument("--filter_threshold", type=float, default=1e-9)
    parser.add_argument("--filter_max_sweeps", type=int, default=int(5e3))
    parser.add_argument("--loop_threshold", type=float, default=1e-9)
    parser.add_argument("--output", type=str, default="tmp")
    parser.add_argument("--num_threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    logfile = args.output + ".log"
    logging.basicConfig(
        filename=logfile,              # <— log file path (relative or absolute)
        level=logging.INFO,
        format="%(message)s",
    )


    beta_c = Ising_critical_beta()
    D_max, max_sweeps, max_rg_steps, mu = args.D_max, args.max_sweeps, args.max_rg_steps, args.mu
    filter_max_sweeps, filter_threshold, loop_threshold = args.filter_max_sweeps, args.filter_threshold, args.loop_threshold
    method = args.method

    f_history = []
    psi = Ising_Z2_symmetric(beta=beta_c)
    loop_tnr = LoopTNR(psi, D_max)

    res = 0
    logging.info(f"beta={beta_c:.6f}, method={method}, D_max={D_max:d}, max_sweeps={max_sweeps:d}, max_rg_steps={max_rg_steps}")
    logging.info(f"parameters: mu={mu}, filter_max_sweeps={filter_max_sweeps}, filter_threshold={filter_threshold}, loop_threshold={loop_threshold}")
    for step in range(max_rg_steps):
        truncate_err = loop_tnr.rg(max_sweeps, method=method, filter_max_sweeps=filter_max_sweeps, filter_threshold=filter_threshold, loop_threshold=loop_threshold, mu=mu)
        norm = Ising_post_processing(loop_tnr)
        res += np.log(norm)/2**(step+1)
        logging.info(f"step-{step:d}: truncation error={truncate_err}, norm={norm}")
        f = -1/beta_c *(res + np.log(trace_2x2(loop_tnr))/2**(step+3)).real
        f_history.append(f.numpy())
    logging.info(f_history)

    F_file = args.output + "_free_energies_conv"
    with open(F_file, "wb") as handle:
        pickle.dump(f_history, handle)


if __name__ == "__main__":
    main()