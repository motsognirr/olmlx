"""HQQ (Half-Quadratic Quantization) for MLX weights.

Data-free weight quantization: no calibration set needed. Solves a
half-quadratic optimisation per weight matrix to find better scale/bias
parameters than naive min/max affine quantisation.

Loaded at model startup when ``OLMLX_WEIGHT_QUANT=hqq:<bits>`` is set
or the per-model ``weight_quant`` field in ``models.json`` is configured.
"""
