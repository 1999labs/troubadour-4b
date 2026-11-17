"""Same cached choice loss; balance A/B/U within every optimizer update."""
import os
os.environ.setdefault('AGX_RELAX_CDM_CTXSTORE_TIMEOUT','1')
from collections import Counter
import numpy as np
import mlx.core as mx
from mlx_lm import lora
from . import train_cached_choice as cached
from . import train_choice
from .group_sampling import group_indices, balanced_indices

_original_batches=cached.token_batches
REBALANCE_EXISTING_LABELS=False


def grouped_batches(dataset,batch_size,max_seq_length,loop=False,seed=None,comm_group=None):
    if not loop:
        yield from _original_batches(dataset,batch_size,max_seq_length,loop=False,seed=seed,comm_group=comm_group)
        return
    if batch_size!=2:raise ValueError('Grouped training uses microbatch2 with accumulation3')
    if comm_group is not None and comm_group.size()!=1:raise ValueError('This sampler is single-device')
    labels=train_choice._label_ids.tolist()
    sampler_seed=int(np.random.randint(0,2**31)) if seed is None else seed
    if REBALANCE_EXISTING_LABELS:
        targets=[dataset[i][0][dataset[i][1]] for i in range(len(dataset))]
        indices=balanced_indices(targets,sampler_seed)
    else:
        indices=group_indices(len(dataset),sampler_seed)
    for group in indices:
        examples=[dataset[j] for j in group]
        targets=[tokens[offset] for tokens,offset in examples]
        if Counter(targets)!=Counter({c:2 for c in labels}):raise ValueError('Each update must contain two A, two B and two U targets')
        for start in range(0,6,2):
            rows=examples[start:start+2];width=max(len(tokens) for tokens,_ in rows)
            if width>max_seq_length:raise ValueError('Grouped examples must not be truncated')
            x=mx.array([list(tokens)+[0]*(width-len(tokens)) for tokens,_ in rows])
            metadata=mx.array([[offset,len(tokens)] for tokens,offset in rows])
            yield x,metadata


_original_train_model=cached.train_model

def train_model(*args,**kwargs):
    if kwargs['args'].grad_accumulation_steps!=3:raise ValueError('Accumulate exactly three microbatches per update')
    return _original_train_model(*args,**kwargs)


if __name__=='__main__':
    mx.set_cache_limit(256*1024*1024);mx.set_memory_limit(10*1024*1024*1024)
    cached.token_batches=grouped_batches
    lora.load=train_choice.load_model;lora.train=train_model;lora.main()
