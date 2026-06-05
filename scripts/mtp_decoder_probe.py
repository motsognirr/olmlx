"""Drive MTPDecoder directly to debug real-path acceptance.

Compares the as-shipped decoder (accumulating draft cache across blocks) vs a
variant that resets the draft cache at the start of each step (each block drafts
fresh from the target hidden). Prints per-step accepted counts.
"""

import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load

from olmlx.engine.mtp.decoder import MTPDecoder
from olmlx.engine.mtp.draft_model import MTPConfig, load_mtp_draft

TARGET = "unsloth/Qwen3.6-27B-MLX-8bit"
HEAD = "mlx-community/Qwen3.6-27B-MTP-4bit"
PROMPT = "Explain in 3 sentences why the sky is blue, then list two other colors the sky can appear."


def run(decoder, prompt_ids, n_steps, reset_draft_each_step):
    decoder.prefill(prompt_ids)
    accepts = []
    for _ in range(n_steps):
        if reset_draft_each_step:
            decoder._draft_cache = decoder._draft.make_cache()
        _accepted, n_draft = decoder.step()
        accepts.append(n_draft)
    total_accepted = sum(accepts)
    total_proposed = decoder._block_size * len(accepts)
    return total_accepted / total_proposed, accepts


def main():
    print("loading...", flush=True)
    model, tok = load(TARGET)
    hd = Path(snapshot_download(HEAD))
    cfg = MTPConfig.from_dict(json.loads((hd / "config.json").read_text()))
    print(f"concat_hidden_first default = {cfg.concat_hidden_first}", flush=True)
    draft = load_mtp_draft(hd, cfg)

    ids = mx.array([tok.encode(PROMPT)], dtype=mx.int32)

    # Sweep chain-norm (h_new = x vs norm(x)) by monkeypatching __call__.

    orig_call = type(draft).__call__

    def make_call(chain_post):
        def _call(self, token_ids, h_prev, cache=None, compute_logits=True):
            emb = self.embed_tokens(token_ids)
            h = self.pre_fc_norm_hidden(h_prev)
            e = self.pre_fc_norm_embedding(emb)
            parts = [h, e] if self.concat_hidden_first else [e, h]
            x = self.fc(mx.concatenate(parts, axis=-1))
            L = x.shape[1]
            mask = None
            if L > 1:
                from mlx_lm.models.base import create_causal_mask

                mask = create_causal_mask(L, offset=cache[0].offset if cache else 0)
            x = self.layers[0](x, mask=mask, cache=cache[0] if cache else None)
            normed = self.norm(x)
            h_new = normed if chain_post else x
            if compute_logits:
                return self.lm_head(normed), h_new
            return None, h_new

        return _call

    for chain_post in (False, True):
        type(draft).__call__ = make_call(chain_post)
        dec = MTPDecoder(model, draft, block_size=cfg.block_size)
        rate, accepts = run(dec, ids, n_steps=40, reset_draft_each_step=False)
        tag = f"chain={'post-norm' if chain_post else 'pre-norm'}"
        print(f"{tag}: acceptance={rate:.3f}  first10={accepts[:10]}", flush=True)
        dec.reset()
    type(draft).__call__ = orig_call


if __name__ == "__main__":
    main()
