"""EAGLE-style speculative decoding (autoregressive draft conditioned on target hidden states).

Reference: arxiv 2401.15077 (EAGLE: Efficient and Accurate Speculative
Decoding via Single-Layer Draft).

EAGLE differs from DFlash:

- **Draft architecture**: an autoregressive transformer head (typically
  1-2 layers) that predicts the *next hidden state* in feature space;
  the LM head (shared with target via ``bind()``) maps that hidden to
  the next-token distribution. Compare DFlash's block-diffusion draft
  that predicts ``block_size`` tokens in parallel from one shared
  context.
- **Conditioning**: each draft step takes ``concat(h_prev,
  embed(token_prev))`` — where ``h_prev`` is either the target's
  last-layer hidden (for the first draft step after a target forward)
  or the draft's own previous output (for steps 2..block_size).
- **Training signal**: standard autoregressive next-token CE on the
  target's tokens, using the target's last-layer hiddens as the
  conditioning input. Single-layer training is much cheaper than
  DFlash's block-diffusion scheme.

Acceptance rate is the dominant lever for any speculative decoding
scheme. EAGLE's published numbers report mean accepted tokens of
3-4 per verify (out of ``block_size+1`` candidates) on Llama-class
targets — roughly 2-3x what DFlash achieves with the same draft
parameter budget, due to the autoregressive-vs-parallel prediction
asymmetry.
"""
