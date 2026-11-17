"""Check optimized and cached scores against scalar full-vocabulary model logits."""
import argparse
import json
from pathlib import Path
from .evaluate import resolve_base_path
from .worlds import messages
from .choice import choice_probabilities
from .template import configure_tokenizer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--adapter')
    parser.add_argument('--mixed-prefixes',action='store_true')
    parser.add_argument('--out',default='artifacts/scoring_parity.json')
    args=parser.parse_args()
    import mlx.core as mx
    from mlx_lm import load
    mx.set_cache_limit(256*1024**2);mx.set_memory_limit(6*1024**3)
    model,tok=load(str(resolve_base_path()),adapter_path=args.adapter)
    configure_tokenizer(tok);model.freeze()
    rows=[json.loads(line) for line in Path('data/dev.jsonl').read_text().splitlines()][:4]
    ks=[0,1,4,8] if args.mixed_prefixes else [8]*4
    inputs=[]
    for row,k in zip(rows,ks):
        episode=dict(row,observations=row['observations'][:k])
        inputs.append(tok.apply_chat_template(messages(episode,completion=False),tokenize=True,return_dict=False,add_generation_prompt=True))
    labels={c:tok.encode(c,add_special_tokens=False)[0] for c in 'ABU'}
    expected=[]
    for tokens in inputs:
        logits=model(mx.array([tokens]))[0,-1]
        probabilities=mx.softmax(logits[mx.array(list(labels.values()))]);mx.eval(probabilities)
        expected.append(probabilities.tolist())
    actual=choice_probabilities(model,inputs,labels)
    from .train_cached_choice import ActivationCache
    cached=choice_probabilities(model,inputs,labels,ActivationCache(model,28))
    errors={}
    choices=lambda values:[max(range(3),key=row.__getitem__) for row in values]
    for name,values in [('batched',actual),('cached',cached)]:
        errors[name]=max(abs(a-b) for x,y in zip(expected,values) for a,b in zip(x,y))
        assert errors[name]<1e-3,(name,errors[name],values,expected)
        assert choices(values)==choices(expected),(name,values,expected)
    report={'adapter':args.adapter,'examples':4,'observation_counts':ks,'prompt_lengths':list(map(len,inputs)),
            'probability_tolerance':1e-3,'max_probability_differences':errors,'same_predictions':True,
            'scalar':expected,'batched':actual,'cached':cached}
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
