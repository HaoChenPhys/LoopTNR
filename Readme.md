# LoopTNR

Implementation of the [loop-TNR](https://arxiv.org/abs/1512.04938) algorithm, inspired by [TNRKit](https://github.com/VictorVanthilt/TNRKit.jl).

It supports

* symmetric tensor manipulations powered by [YASTN](https://github.com/yastn/yastn)

* generic multi-site unit cell

* Alternating least squares optimization of the loop MPS, with entanglement-filtering

* AD-based optimization of the loop MPS, with additional [nuclear-norm regularization](https://arxiv.org/abs/2306.17479)
