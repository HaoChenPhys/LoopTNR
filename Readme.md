# LoopTNR

Implementation of the [loop-TNR](https://arxiv.org/abs/1512.04938) algorithm.

It supports

* symmetric tensor manipulations powered by [YASTN](https://github.com/yastn/yastn)

* generic multi-site unit cell

* Alternating least squares optimization of the loop MPS, with entanglement-filtering

* AD-based optimization of the loop MPS, with additional [nuclear-norm regularization](https://arxiv.org/abs/2306.17479)
