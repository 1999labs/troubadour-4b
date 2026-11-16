"""Choice LoRA with reusable activations from frozen lower transformer blocks.

Caching changes computation cost, not the model or its input. Cache keys include
pinned base revision, tokenizer template, split layer, and exact prompt tokens.
Only input prefixes are cached; answer tokens and labels never enter activations.
"""
import os
os.environ.setdefault('AGX_RELAX_CDM_CTXSTORE_TIMEOUT', '1')
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import lora
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tuner.trainer import iterate_batches as token_batches
from . import train_choice
from .evaluate import REVISION
from .template import TEMPLATE


def bottom_hidden(model, tokens, split):
    h = model.model.embed_tokens(tokens)
    mask = create_attention_mask(h, None)
    for layer in model.layers[:split]:
        h = layer(h, mask, None)
    return h


def suffix_logits(model, hidden, lengths, split, label_ids):
    h = hidden
    mask = create_attention_mask(h, None)
    for layer in model.layers[split:]:
        h = layer(h, mask, None)
    last = model.model.norm(h)[mx.arange(h.shape[0]), lengths - 1]
    logits = model.model.embed_tokens.as_linear(last) if model.args.tie_word_embeddings else model.lm_head(last)
    return logits[:, label_ids]


class ActivationCache:
    def __init__(self, model, split, directory='artifacts/frozen-cache'):
        self.model, self.split = model, split
        identity = json.dumps({'revision': REVISION, 'split': split, 'template': TEMPLATE, 'mlx': version('mlx'), 'mlx_lm': version('mlx-lm'), 'cache_protocol': 1}, sort_keys=True)
        self.root = Path(directory) / hashlib.sha256(identity.encode()).hexdigest()[:20]
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'identity.json').write_text(identity)
        # A cache is invalid if any earlier block or embedding can change.
        for name, _ in tree_flatten(model.trainable_parameters()):
            if not name.startswith('model.layers.') or int(name.split('.')[2]) < split:
                raise ValueError('Cache would include trainable parameters: ' + name)

    def get(self, prefixes):
        paths = [self.root / (hashlib.sha256(json.dumps(p).encode()).hexdigest() + '.safetensors') for p in prefixes]
        values = [mx.load(str(p))['hidden'] if p.exists() else None for p in paths]
        missing = [i for i, v in enumerate(values) if v is None]
        if missing:
            width = max(len(prefixes[i]) for i in missing)
            x = mx.array([prefixes[i] + [0] * (width-len(prefixes[i])) for i in missing])
            computed = bottom_hidden(self.model, x, self.split)
            mx.eval(computed)
            for j, i in enumerate(missing):
                h = computed[j, :len(prefixes[i])]
                temporary = paths[i].with_suffix('.partial.safetensors')
                mx.save_safetensors(str(temporary), {'hidden': h})
                temporary.replace(paths[i])
                values[i] = h
        width = max(len(p) for p in prefixes)
        return mx.stack([mx.pad(h, [(0, width-h.shape[0]), (0, 0)]) for h in values])


def train_model(*args, **kwargs):
    model = kwargs['model']
    trainable = [int(n.split('.')[2]) for n, _ in tree_flatten(model.trainable_parameters()) if n.startswith('model.layers.')]
    split = min(trainable)
    cache = ActivationCache(model, split)
    label_ids = train_choice._label_ids
    id_list = label_ids.tolist()

    def batches(**options):
        for tokens, lengths in token_batches(**options):
            ids, meta = tokens.tolist(), lengths.tolist()
            prefixes, targets = [], []
            for row, (offset, end) in zip(ids, meta):
                if not 0 < offset < end or row[offset] not in id_list:
                    raise ValueError('Expected an untruncated A/B/U answer after prompt')
                prefixes.append(row[:offset])
                targets.append(id_list.index(row[offset]))
            hidden = cache.get(prefixes)
            yield hidden, mx.array(list(zip([len(p) for p in prefixes], targets)))

    def loss(model, hidden, metadata):
        logits = suffix_logits(model, hidden, metadata[:, 0], split, label_ids)
        return nn.losses.cross_entropy(logits.astype(mx.float32), metadata[:, 1], reduction='mean'), mx.array(hidden.shape[0])

    kwargs.update(loss=loss, iterate_batches=batches)
    print(json.dumps({'frozen_blocks_cached': split, 'trainable_blocks': len(model.layers)-split, 'cache': str(cache.root)}), flush=True)
    return train_choice._original_train(*args, **kwargs)


if __name__ == '__main__':
    mx.set_cache_limit(256*1024*1024)
    mx.set_memory_limit(10*1024*1024*1024)
    lora.load = train_choice.load_model
    lora.train = train_model
    lora.main()
