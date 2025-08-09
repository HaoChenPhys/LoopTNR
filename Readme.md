# LoopTNR

Implementation of the [loop-TNR](https://arxiv.org/abs/1512.04938) algorithm, inspired by [TNRKit](https://github.com/VictorVanthilt/TNRKit.jl).

## Highlights

* Symmetric tensor manipulations powered by [YASTN](https://github.com/yastn/yastn).

* Alternating least squares optimization of the loop MPS, with entanglement-filtering.

* Generic multi-site unit cell.

    We optimize disjoint square plaquettes arranged with offsets (0, 2) or (2, 0). Contracting these plaquettes reproduces the partition function. The corresponding octagons do not share edges, so loop updates can be performed sequentially.
    In practice, we allow the 8-site MPS attached to each octagons to have independent tensors, which improves performance. This choice implies a minimal iPEPS unit cell of 2×2 (four independent site tensors).


<!-- * AD-based optimization of the loop MPS, with additional [nuclear-norm regularization](https://arxiv.org/abs/2306.17479) -->

<!-- TODO: CTMRG -->