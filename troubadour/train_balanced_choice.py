"""Cached choice training with balanced updates from the existing choice dataset."""
import os
os.environ.setdefault('AGX_RELAX_CDM_CTXSTORE_TIMEOUT','1')
import mlx.core as mx
from mlx_lm import lora
from . import train_grouped_choice as grouped
from . import train_cached_choice as cached
from . import train_choice

if __name__=='__main__':
    mx.set_cache_limit(256*1024**2);mx.set_memory_limit(10*1024**3)
    grouped.REBALANCE_EXISTING_LABELS=True
    cached.token_batches=grouped.grouped_batches
    lora.load=train_choice.load_model;lora.train=grouped.train_model
    lora.main()
