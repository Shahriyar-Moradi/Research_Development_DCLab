"""
gated_delta_chunked.py
======================

Chunked (chunk-parallel) gated delta rule for TRAINING Qwen3.5 with MLX.

Why: in training mode, mlx_lm runs Qwen3.5's linear-attention layers (Gated DeltaNet) as a Python loop over every
token (``gated_delta_ops``), because its fast Metal kernel has no backward pass. For backprop that loop keeps one
state per token: about 2 MB per token per layer, so a 12k-token sample needs over 20 GB for a single layer and
builds a graph of tens of thousands of tiny operations. This module computes the same recurrence chunk by chunk
(64 tokens): inside a chunk everything is a few matrix products, and only one state per chunk is kept, about 64×
less memory and far fewer operations. It is the standard chunked form of the gated delta rule (as in
flash-linear-attention's reference ``chunk_gated_delta_rule``), written with differentiable MLX ops; the
triangular inverse is computed by repeated squaring, so no solver is needed.

``patch_qwen3_5()`` routes Qwen3.5's training-time calls here and leaves inference (cache, masks, the kernel)
untouched. ``python mac/gated_delta_chunked.py`` checks values and gradients against mlx_lm's own sequential
implementation.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def chunked_gated_delta(q, k, v, log_g, beta, chunk: int = 64):
    """q, k: [B, T, Hk, Dk]; v: [B, T, Hv, Dv]; log_g, beta: [B, T, Hv] (log of the per-step decay; write strength).
    Returns y: [B, T, Hv, Dv] and the final state [B, Hv, Dv, Dk] (mlx_lm's layout), starting from a zero state."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Hv // Hk > 1:
        q = mx.repeat(q, Hv // Hk, axis=2)
        k = mx.repeat(k, Hv // Hk, axis=2)
    out_dtype = q.dtype
    q, k, v = (x.astype(mx.float32) for x in (q, k, v))
    log_g, beta = log_g.astype(mx.float32), beta.astype(mx.float32)
    pad = (-T) % chunk
    if pad:  # padded steps: no write (k = 0, beta = 0) and no decay (log_g = 0), so the state is unchanged
        q, k, v = (mx.pad(x, [(0, 0), (0, pad), (0, 0), (0, 0)]) for x in (q, k, v))
        log_g, beta = (mx.pad(x, [(0, 0), (0, pad), (0, 0)]) for x in (log_g, beta))
    n = (T + pad) // chunk
    C = chunk

    def to_chunks(x):  # [B, T', H, D] -> [B, H, N, C, D]
        return x.reshape(B, n, C, Hv, -1).transpose(0, 3, 1, 2, 4)

    qc, kc, vc = to_chunks(q), to_chunks(k), to_chunks(v)
    gc = log_g.reshape(B, n, C, Hv).transpose(0, 3, 1, 2)  # [B, H, N, C]
    bc = beta.reshape(B, n, C, Hv).transpose(0, 3, 1, 2)
    gcum = mx.cumsum(gc, axis=-1)
    kb = kc * bc[..., None]
    vb = vc * bc[..., None]

    idx = mx.arange(C)
    lower = idx[:, None] >= idx[None, :]
    strict = idx[:, None] > idx[None, :]
    diff = gcum[..., :, None] - gcum[..., None, :]
    decay = mx.where(lower, mx.exp(mx.where(lower, diff, 0.0)), 0.0)  # exp(γi − γj) for j ≤ i, else 0

    # A strictly lower; T = (I − A)^(-1) = (I + A)(I + A²)(I + A⁴)… since A is nilpotent (A^C = 0)
    A = mx.where(strict, -(kb @ kc.swapaxes(-1, -2)) * decay, 0.0)
    eye = mx.eye(C, dtype=mx.float32)
    Tm, P = eye + A, A
    for _ in range(int(math.ceil(math.log2(C))) - 1):
        P = P @ P
        Tm = Tm + Tm @ P
    u = Tm @ vb  # [B, H, N, C, Dv]
    w = Tm @ (kb * mx.exp(gcum)[..., None])  # [B, H, N, C, Dk]

    S = mx.zeros((B, Hv, Dk, Dv), dtype=mx.float32)
    outs = []
    for i in range(n):
        qi, ki, ui, wi, gi = qc[:, :, i], kc[:, :, i], u[:, :, i], w[:, :, i], gcum[:, :, i]
        attn = (qi @ ki.swapaxes(-1, -2)) * decay[:, :, i]
        v_new = ui - wi @ S
        outs.append((qi * mx.exp(gi)[..., None]) @ S + attn @ v_new)
        last = gi[..., -1]
        S = S * mx.exp(last)[..., None, None] + (ki * mx.exp(last[..., None] - gi)[..., None]).swapaxes(-1, -2) @ v_new
    y = mx.stack(outs, axis=2)  # [B, H, N, C, Dv]
    y = y.transpose(0, 2, 3, 1, 4).reshape(B, n * C, Hv, Dv)[:, :T]
    return y.astype(out_dtype), S.swapaxes(-1, -2)


def patch_qwen3_5(chunk: int = 64) -> None:
    """Use the chunked rule for Qwen3.5's training-time calls (no cache state, no mask); inference is unchanged."""
    from mlx_lm.models import gated_delta as G
    from mlx_lm.models import qwen3_5 as Q

    original = G.gated_delta_update

    def update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *, use_kernel=True, lower_bound=None,
               allow_neg_eigval=False):
        if use_kernel or state is not None or mask is not None or lower_bound is not None:
            return original(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel,
                            lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
        beta = mx.sigmoid(b) * (2.0 if allow_neg_eigval else 1.0)
        log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)  # log of compute_g
        return chunked_gated_delta(q, k, v, log_g, beta, chunk)

    Q.gated_delta_update = update


def _self_test():
    for shape in [(1, 150, 2, 4, 16, 8, 32), (2, 128, 3, 3, 32, 16, 64), (1, 300, 4, 8, 16, 16, 64)]:
        _check(*shape)
    print("ok: the chunked rule matches mlx_lm's sequential gated delta rule (values, final state, gradients)")


def _check(B, T, Hk, Hv, Dk, Dv, chunk):
    from mlx_lm.models.gated_delta import compute_g, gated_delta_ops
    mx.random.seed(T)
    q = mx.random.normal((B, T, Hk, Dk)) * 0.3
    k = mx.random.normal((B, T, Hk, Dk))
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    v = mx.random.normal((B, T, Hv, Dv))
    a = mx.random.normal((B, T, Hv))
    b = mx.random.normal((B, T, Hv))
    A_log = mx.random.normal((Hv,)) * 0.5
    dt = mx.random.normal((Hv,)) * 0.1

    def ref(q, k, v, a, b):
        return gated_delta_ops(q, k, v, compute_g(A_log, a, dt), mx.sigmoid(b))

    def new(q, k, v, a, b):
        return chunked_gated_delta(q, k, v, -mx.exp(A_log) * nn.softplus(a + dt), mx.sigmoid(b), chunk=chunk)

    y0, s0 = ref(q, k, v, a, b)
    y1, s1 = new(q, k, v, a, b)
    err_y = mx.abs(y0 - y1).max().item()
    err_s = mx.abs(s0 - s1).max().item()

    def loss(f):
        return lambda q, k, v, a, b: (f(q, k, v, a, b)[0] * mx.arange(Dv)).sum()

    g0 = mx.grad(loss(ref), argnums=(0, 1, 2, 3, 4))(q, k, v, a, b)
    g1 = mx.grad(loss(new), argnums=(0, 1, 2, 3, 4))(q, k, v, a, b)
    err_g = max(mx.abs(x - y).max().item() for x, y in zip(g0, g1))
    print(f"B={B} T={T} Hk={Hk} Hv={Hv} chunk={chunk}: max |Δy| {err_y:.2e}  max |Δstate| {err_s:.2e}  max |Δgrad| {err_g:.2e}")
    assert err_y < 1e-4 and err_s < 1e-4 and err_g < 1e-3, "chunked rule does not match the sequential one"


if __name__ == "__main__":
    _self_test()
