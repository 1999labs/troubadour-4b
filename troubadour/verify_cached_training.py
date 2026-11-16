"""Check cached/full forward probabilities and LoRA gradients before cached training."""
import os
os.environ.setdefault('AGX_RELAX_CDM_CTXSTORE_TIMEOUT','1')
import argparse
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.tuner.utils import linear_to_lora_layers
from .train_cached_choice import ActivationCache,suffix_logits
from .train import load_non_thinking
from .evaluate import BASE
from .worlds import messages


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--adapter');args=parser.parse_args()
    mx.set_cache_limit(256*1024*1024);mx.set_memory_limit(10*1024*1024*1024)
    model,tok=load_non_thinking(BASE)
    model.freeze()
    if args.adapter:
        config=json.loads((Path(args.adapter)/'adapter_config.json').read_text())
        if config['num_layers']!=8:raise ValueError('Audit expects eight adapted layers')
        parameters=config['lora_parameters']
    else:
        parameters={'rank':8,'scale':16.0,'dropout':0.0}
    mx.random.seed(17)
    linear_to_lora_layers(model,8,parameters)
    if args.adapter:
        model.load_weights(str(Path(args.adapter)/'adapters.safetensors'),strict=False)
    else:
        # Nonzero adapters exercise both LoRA factors, without training a model.
        model.load_weights([(n,mx.random.normal(w.shape).astype(w.dtype)*0.001) for n,w in tree_flatten(model.trainable_parameters()) if n.endswith('lora_b')],strict=False)
    model.eval()
    rows=[json.loads(l) for l in Path('data/choice_episodes.jsonl').read_text().splitlines()][:2]
    inputs=[tok.apply_chat_template(messages(r,completion=False),tokenize=True,return_dict=False,add_generation_prompt=True) for r in rows]
    width=max(map(len,inputs));x=mx.array([p+[0]*(width-len(p)) for p in inputs]);lengths=mx.array(list(map(len,inputs)))
    ids=mx.array([tok.encode(c,add_special_tokens=False)[0] for c in 'ABU']);targets=mx.array(['ABU'.index(r['answer']) for r in rows])
    cache=ActivationCache(model,28)
    hidden=cache.get(inputs);mx.eval(hidden)
    restored=cache.get(inputs);mx.eval(restored)
    assert mx.array_equal(hidden,restored).item()
    def full_logits(m):
        h=m.model(x)[mx.arange(len(inputs)),lengths-1]
        return m.model.embed_tokens.as_linear(h)[:,ids].astype(mx.float32)
    a=full_logits(model);b=suffix_logits(model,hidden,lengths,28,ids).astype(mx.float32)
    pa=mx.softmax(a);pb=mx.softmax(b);mx.eval(pa,pb)
    forward_error=mx.max(mx.abs(pa-pb)).item()
    assert forward_error<1e-5,(pa,pb)
    def full_loss(m):return nn.losses.cross_entropy(full_logits(m),targets,reduction='mean')
    def cached_loss(m):return nn.losses.cross_entropy(suffix_logits(m,hidden,lengths,28,ids).astype(mx.float32),targets,reduction='mean')
    la,ga=nn.value_and_grad(model,full_loss)(model);mx.eval(la,ga)
    lb,gb=nn.value_and_grad(model,cached_loss)(model);mx.eval(lb,gb)
    g1=dict(tree_flatten(ga));g2=dict(tree_flatten(gb));assert g1.keys()==g2.keys()
    numerator=sum(mx.sum((g1[k].astype(mx.float32)-g2[k].astype(mx.float32))**2).item() for k in g1)
    denominator=sum(mx.sum(g1[k].astype(mx.float32)**2).item() for k in g1)
    relative_error=(numerator/max(denominator,1e-30))**.5
    assert relative_error<1e-4,relative_error
    from .choice import choice_probabilities
    labels=dict(zip('ABU',ids.tolist()))
    plain=choice_probabilities(model,inputs,labels)
    fast=choice_probabilities(model,inputs,labels,cache)
    assert plain==fast,(plain,fast)
    result={'adapter':args.adapter or 'deterministic nonzero audit adapters','inference_probabilities_exact':True,'examples':len(rows),'cached_frozen_layers':28,'max_probability_error':forward_error,'full_loss':la.item(),'cached_loss':lb.item(),'relative_gradient_l2_error':relative_error,'cache_roundtrip_exact':True,'future_answer_tokens_in_cache':False}
    Path('artifacts/cached_training_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
