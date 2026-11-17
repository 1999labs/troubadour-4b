"""Batched constrained-choice evaluation, identical tokens and labels to scalar scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from .choice import choice_probabilities
from .evaluate import BASE,BASE_PATH,REVISION,safe_power,summarize,resolve_base_path
from .worlds import Oracle,messages


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/dev.jsonl');p.add_argument('--out',required=True);p.add_argument('--adapter');p.add_argument('--limit',type=int);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--ks',default='8');p.add_argument('--style',default='table',choices=['table','prose','symbols']);p.add_argument('--activation-cache',action='store_true');a=p.parse_args()
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    source=Path(a.data);data=[json.loads(l) for l in source.read_text().splitlines()]
    if a.limit:data=data[:a.limit]
    ks=[int(k) for k in a.ks.split(',')]
    from .template import renderer_sha256
    from importlib.metadata import version
    cfg={k:v for k,v in vars(a).items() if k!='activation_cache' or v}|{'protocol':'constrained-choice-v3','renderer_sha256':renderer_sha256(),'runtime_versions':{n:version(n) for n in ['mlx','mlx-lm','transformers']},'base':BASE,'revision':REVISION,'data_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    if a.adapter:cfg['adapter_sha256']=hashlib.sha256((Path(a.adapter)/'adapters.safetensors').read_bytes()).hexdigest()
    meta=out.with_suffix('.meta.json')
    if meta.exists() and json.loads(meta.read_text())!=cfg:raise ValueError('Use a different output file after changing configuration')
    meta.write_text(json.dumps(cfg,indent=2))
    done=[json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    finished={(r['episode_id'],r['k']) for r in done}
    jobs=[(r,k) for r in data for k in ks if (r['id'],k) not in finished]
    import mlx.core as mx
    from mlx_lm import load
    model,tok=load(str(resolve_base_path()),adapter_path=a.adapter)
    from .template import configure_tokenizer
    configure_tokenizer(tok)
    activation_cache=None
    if a.activation_cache:
        if a.adapter:
            config=json.loads((Path(a.adapter)/'adapter_config.json').read_text())
            if config['num_layers']>8:raise ValueError('Cached evaluator only permits the final eight blocks to change')
        from .train_cached_choice import ActivationCache
        model.freeze()
        activation_cache=ActivationCache(model,28)
    labels={c:tok.encode(c,add_special_tokens=False) for c in ('A','B','U')}
    assert all(len(v)==1 for v in labels.values())
    label_ids={c:v[0] for c,v in labels.items()};oracle=Oracle()
    for offset in range(0,len(jobs),a.batch_size):
        safe_power();batch=jobs[offset:offset+a.batch_size];inputs=[];answers=[]
        for row,k in batch:
            assert k<=len(row['observations'])
            episode=dict(row);episode['observations']=row['observations'][:k]
            answers.append(oracle.answer(episode['observations'],row['query']))
            inputs.append(tok.apply_chat_template(messages(episode,style=a.style,completion=False),tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False))
        started=time.time();probabilities=choice_probabilities(model,inputs,label_ids,activation_cache);seconds=time.time()-started
        for (row,k),ids,(expected,n),pvals in zip(batch,inputs,answers,probabilities):
            probs=dict(zip(('A','B','U'),pvals));prediction=max(probs,key=probs.get)
            result={'episode_id':row['id'],'society_pair':row['society_pair'],'family':row['family'],'k':k,'style':a.style,'expected':expected,'prediction':prediction,'raw':prediction,'label_probs':probs,'consistent_rules':n,'prompt_tokens':len(ids),'seconds':seconds/len(batch)}
            if 'contrast_group' in row:result['contrast_group']=row['contrast_group']
            with out.open('a') as f:f.write(json.dumps(result)+'\n')
            done.append(result)
        mx.clear_cache()
        if offset%(a.batch_size*5)==0:print(json.dumps({'completed':len(done),'summary':summarize(done)}),flush=True)
    report={'overall':summarize(done),'by_k':{str(k):summarize([r for r in done if r['k']==k]) for k in ks}}
    out.with_suffix('.summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
