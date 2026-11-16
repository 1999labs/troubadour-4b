"""Efficient constrained next-token scoring for Qwen; no oracle during inference."""
from __future__ import annotations


def choice_probabilities(model, token_lists, label_ids, activation_cache=None):
    import mlx.core as mx
    if activation_cache is not None:
        from .train_cached_choice import suffix_logits
        h=activation_cache.get(token_lists)
        logits=suffix_logits(model,h,mx.array([len(p) for p in token_lists]),activation_cache.split,mx.array([label_ids[c] for c in 'ABU']))
        probs=mx.softmax(logits,axis=-1);mx.eval(probs)
        return probs.tolist()
    # Right padding cannot influence earlier positions under causal attention.
    lengths=[len(t) for t in token_lists]
    width=max(lengths)
    x=mx.array([t+[0]*(width-len(t)) for t in token_lists])
    hidden=model.model(x)
    last=hidden[mx.arange(len(lengths)),mx.array(lengths)-1]
    if model.args.tie_word_embeddings:
        logits=model.model.embed_tokens.as_linear(last)
    else:
        logits=model.lm_head(last)
    ids=mx.array([label_ids[c] for c in ('A','B','U')])
    probs=mx.softmax(logits[:,ids],axis=-1)
    mx.eval(probs)
    return probs.tolist()
