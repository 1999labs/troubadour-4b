"""Canonical constrained scoring, independent of padding or output projection shape."""
from __future__ import annotations


def choice_probabilities(model, token_lists, label_ids, activation_cache=None):
    import mlx.core as mx
    ids=mx.array([label_ids[c] for c in 'ABU'])
    values=[]
    # BF16 quantized kernels can round differently when only the final hidden
    # vector is projected. Use the full output projection, one unpadded input
    # at a time, exactly as scalar model(tokens)[0, -1].
    for tokens in token_lists:
        if activation_cache is None:
            logits=model(mx.array([tokens]))[0,-1,ids]
        else:
            from .train_cached_choice import suffix_logits
            hidden=activation_cache.get([tokens])
            logits=suffix_logits(model,hidden,mx.array([len(tokens)]),activation_cache.split,ids,full_projection=True)[0]
        probabilities=mx.softmax(logits,axis=-1)
        mx.eval(probabilities)
        values.append(probabilities.tolist())
    return values
