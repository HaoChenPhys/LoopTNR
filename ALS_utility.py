import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

import yastn.yastn as yastn
from yastn.yastn import decompress_from_1d


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
    err_prev = np.inf
    num_step, pos = 0, 0
    L_list[pos] = L
    for i in range(max_sweeps):
        next = (pos+1)%len(psi)
        L = QR_L(L, psi[pos])
        L_prev, L_list[next] = L_list[next], L
        if L_prev and same_sectors(L, L_prev):
            err = (L - L_prev).norm()
            if conv_check(err_prev, err):
                break
            err_prev = err

        pos = next
        num_step += 1


    return L_list, err

def find_R(psi, max_sweeps, conv_check):
    '''
    Compute the R tensors for a 4-site PBC MPS used in arXiv:1512.04938v2.
    '''
    R_list = [None] * len(psi)
    R = yastn.eye(config=psi[-1].config,legs=psi[-1].get_legs(axes=3).conj(), isdiag=False)
    err_prev = np.inf
    N = len(psi)
    num_step, pos = 0, N-1
    R_list[pos] = R
    for i in range(max_sweeps):
        next = (pos-1)%N
        R = QR_R(R, psi[pos])
        R_prev, R_list[next] = R_list[next], R
        if R_prev and same_sectors(R, R_prev):
            err = ((R - R_prev).norm())
            if conv_check(err_prev, err):
                break
            err_prev = err

        pos = next
        num_step += 1

    return R_list, err

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