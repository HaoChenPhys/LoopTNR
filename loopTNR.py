import numpy as np
import torch
from torch.optim import LBFGS

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
        masked = set()
        site = self.sites()[0]
        for _ in range(self.dims[0]):
            for _ in range(self.dims[1]):
                tl, tr, bl, br = self.site2index(site), self.site2index(self.nn_site(site, "r")), self.site2index(self.nn_site(site, "b")), self.site2index(self.nn_site(site, "br"))
                if (tl, "br") not in masked:
                    masked.add((tl, "br"))
                    yield self.psi_rg[self.site_rg_map[(tl, "br")][1]].tl
                if (tr, "tr") not in masked:
                    masked.add((tr, "tr"))
                    yield self.psi_rg[self.site_rg_map[(tr, "tr")][0]].tr
                if (br, "br") not in masked:
                    masked.add((br, "br"))
                    yield self.psi_rg[self.site_rg_map[(br, "br")][0]].br
                if (bl, "tr") not in masked:
                    masked.add((bl, "tr"))
                    yield self.psi_rg[self.site_rg_map[(bl, "tr")][1]].bl
                site = self.nn_site(site, (-1, 1))
            site = self.nn_site(site, (1, 1))

    def get_decomp_T_data(self):
        return list(T._data for T in self.indep_decomp_T())


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
                tl, tr, bl, br = self.site2index(site), self.site2index(self.nn_site(site, "r")), self.site2index(self.nn_site(site, "b")), self.site2index(self.nn_site(site, "br"))

                if self.psi_rg[site_rg].tl is None:
                    if (tl, "br") in self.site_rg_map:
                        self.psi_rg[site_rg].tl = self.psi_rg[self.site_rg_map[(tl, "br")][1]].tl
                    else:
                        u, s, v = self.psi[tl].svd_with_truncation(axes=((0, 1), (2, 3)), sU=1, policy='fullrank',
                                                                    tol=1e-12, D_total=D_max, truncate_multiplets=True)
                        site_rg_t = self.site2index(self.psi_rg.nn_site(site_rg, "t"))
                        self.psi_rg[site_rg_t].br = u@s.sqrt()
                        self.psi_rg[site_rg].tl = s.sqrt()@v
                        self.site_rg_map[(tl, "br")] = (site_rg_t, site_rg)

                if self.psi_rg[site_rg].tr is None:
                    if (tr, "tr") in self.site_rg_map:
                        self.psi_rg[site_rg].tr = self.psi_rg[self.site_rg_map[(tr, "tr")][0]].tr
                    else:
                        u, s, v = self.psi[tr].svd_with_truncation(axes=((1, 2), (0, 3)), sU=1, policy='fullrank',
                                                                    tol=1e-12, D_total=D_max, truncate_multiplets=True)
                        site_rg_r = self.site2index(self.psi_rg.nn_site(site_rg, "r"))
                        self.psi_rg[site_rg].tr = u@s.sqrt()
                        self.psi_rg[site_rg_r].bl = s.sqrt()@v
                        self.site_rg_map[(tr, "tr")] = (site_rg, site_rg_r)

                if self.psi_rg[site_rg].br is None:
                    if (br, "br") in self.site_rg_map:
                        self.psi_rg[site_rg].br = self.psi_rg[self.site_rg_map[(br, "br")][0]].br
                    else:
                        u, s, v = self.psi[br].svd_with_truncation(axes=((0, 1), (2, 3)), sU=1, policy='fullrank',
                                                                    tol=1e-12, D_total=D_max, truncate_multiplets=True)
                        site_rg_b = self.site2index(self.psi_rg.nn_site(site_rg, "b"))
                        self.psi_rg[site_rg].br = u@s.sqrt()
                        self.psi_rg[site_rg_b].tl = s.sqrt()@v
                        self.site_rg_map[(br, "br")] = (site_rg, site_rg_b)

                if self.psi_rg[site_rg].bl is None:
                    if (bl, "tr") in self.site_rg_map:
                        self.psi_rg[site_rg].bl = self.psi_rg[self.site_rg_map[(bl, "tr")][1]].bl
                    else:
                        u, s, v = self.psi[bl].svd_with_truncation(axes=((1, 2), (0, 3)), sU=1, policy='fullrank',
                                                                    tol=1e-12, D_total=D_max, truncate_multiplets=True)
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

    @torch.no_grad()
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
                    # print(f"step-{step:d} site: {tl}: {err}")
                    converged = (converged and (np.abs(err - truncate_err[tl]) < threshold))
                    truncate_err[tl] = err.real
                    # print(f"iteration {step}, top-left corner {tl}: truncation error={truncate_err[tl]}")
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
                    site = self.site2index(site)
                    psi_A = self.psiA_mps(site)
                    psi_A_dict[site] = psi_A
                    AA_norm_dict[site] = get_norm(TM_psiA_psiA(psi_A))
                    site = self.nn_site(site, (-1, 1))
                site = self.nn_site(site, (1, 1))

        def overlap_loss():
            nonlocal overlap
            site = self.nn_site(self.sites()[0], 'b')
            loss = 0.0
            for _ in range(self.dims[0]):
                for _ in range(self.dims[1]):
                    site = self.site2index(site)
                    psi_B = self.psiB_mps(site)
                    AA_norm = AA_norm_dict[site]
                    AB_norm = get_norm(TM_psiA_psi_B(psi_A_dict[site], psi_B))
                    BB_norm = get_norm(TM_psiB_psiB(psi_B))
                    loss += (AA_norm + BB_norm - AB_norm - AB_norm.conj()).to_number().real / AA_norm.to_number().real
                    site = self.nn_site(site, (-1, 1))
                site = self.nn_site(site, (1, 1))
            overlap = loss
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
            if torch.abs(loss - loss_history[-1]) < loop_threshold:
                break
            loss_history.append(loss)

        for data in data_list:
            data.detach_()
        return overlap

    def rg(self, max_sweeps, filter_max_sweeps=1000, filter_threshold=1e-6, loop_threshold=1e-6, mu=1e-8):
        r"""
        Perform the Loop-TNR RG procedure on the PEPS lattice.
        """
        # for site in self.psi.sites():
        #     self.entanglement_filtering(site, filter_max_sweeps, conv_check=lambda err_prev, err: abs(err_prev - err) < filter_threshold)

        # loop compression
        # truncate_err = self.loop_optimize(max_sweeps, loop_threshold)
        # truncate_err = max(truncate_err.values())
        # self.sync_decomp_T_()

        self.init_psi_rg(self.D_max)
        truncate_err = self.loop_optimize_AD(max_sweeps, loop_threshold, mu=mu)

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

    D_max = 16
    max_sweeps = 100

    f_history = []
    # psi = Ising_dense(beta=beta, h=0.0)
    psi = Ising_Z2_symmetric(beta=beta_c)
    loop_tnr = LoopTNR(psi, D_max)

    res = 0
    for step in range(20):
        truncate_err = loop_tnr.rg(max_sweeps, filter_max_sweeps=0, loop_threshold=1e-9, mu=1e-6)
        norm = Ising_post_processing(loop_tnr)
        res += np.log(norm)/2**(step+1)
        print(f"beta={beta_c:.6f}, step-{step:d}: truncation error={truncate_err}, norm={norm}")
        f = -1/beta_c *(res + np.log(trace_2x2(loop_tnr))/2**(step+3)).real
        f_history.append(f)
    print(f_history)