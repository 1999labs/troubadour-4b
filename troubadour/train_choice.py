"""LoRA supervised classification using the same three-token decision as evaluation."""
import os
os.environ.setdefault('AGX_RELAX_CDM_CTXSTORE_TIMEOUT','1')
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import lora
from .train import load_non_thinking

_label_ids=None
_original_train=lora.train


def load_model(*args,**kwargs):
    global _label_ids
    model,tok=load_non_thinking(*args,**kwargs)
    labels=[tok.encode(c,add_special_tokens=False) for c in ('A','B','U')]
    assert all(len(v)==1 for v in labels)
    _label_ids=mx.array([v[0] for v in labels])
    return model,tok


def classification_loss(model,batch,lengths):
    # The answer is at `offset`; its predictor is hidden state `offset-1`.
    # Causal attention prevents future target tokens from entering that state.
    hidden=model.model(batch[:,:-1])
    offsets=lengths[:,0]
    last=hidden[mx.arange(batch.shape[0]),offsets-1]
    logits=model.model.embed_tokens.as_linear(last) if model.args.tie_word_embeddings else model.lm_head(last)
    logits=logits[:,_label_ids].astype(mx.float32)
    target_tokens=batch[mx.arange(batch.shape[0]),offsets]
    target=(target_tokens[:,None]==_label_ids[None,:]).astype(mx.int32).argmax(axis=1)
    loss=nn.losses.cross_entropy(logits,target,reduction='mean')
    return loss,mx.array(batch.shape[0])


def train_model(*args,**kwargs):
    kwargs['loss']=classification_loss
    return _original_train(*args,**kwargs)

if __name__=='__main__':
    mx.set_cache_limit(256*1024*1024)
    mx.set_memory_limit(10*1024*1024*1024)
    lora.load=load_model
    lora.train=train_model
    lora.main()
