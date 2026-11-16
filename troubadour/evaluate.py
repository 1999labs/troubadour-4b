"""Reproducible, resumable evaluation. No simulator metadata reaches the model."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from .worlds import Oracle, messages

BASE = 'mlx-community/Qwen3-4B-Instruct-2507-4bit'
REVISION = '50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'
BASE_PATH = Path.home()/'.cache/huggingface/hub'/('models--'+BASE.replace('/','--'))/'snapshots'/REVISION


def resolve_base_path():
    if BASE_PATH.exists():return BASE_PATH
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(BASE,revision=REVISION))


def parse_answer(text):
    text=text.strip()
    return text if text in ('A','B','U') else 'INVALID'


def summarize(rows):
    if not rows:return {}
    counts=Counter(r['expected'] for r in rows)
    correct=Counter(r['expected'] for r in rows if r['prediction']==r['expected'])
    bypair=defaultdict(list)
    for r in rows:bypair[r['episode_id'].removesuffix('-reverse')+'-'+str(r['k'])].append(r)
    pairs=[v for v in bypair.values() if len(v)==2]
    contrast=defaultdict(list)
    for r in rows:
        if r.get('contrast_group'):contrast[r['contrast_group']].append(r)
    contrast_pairs=[v for v in contrast.values() if len(v)==2]
    answerable=[r for r in rows if r['expected']!='U']
    unknown=[r for r in rows if r['prediction']=='U']
    return {'n':len(rows),'accuracy':sum(r['prediction']==r['expected'] for r in rows)/len(rows),
        'balanced_accuracy':sum(correct[c]/counts[c] for c in counts)/len(counts),
        'answerable_accuracy':sum(r['prediction']==r['expected'] for r in answerable)/len(answerable) if answerable else None,
        'uncertainty_recall':correct['U']/counts['U'] if counts['U'] else None,
        'uncertainty_precision':sum(r['expected']=='U' for r in unknown)/len(unknown) if unknown else None,
        'paired_both_correct':sum(all(r['prediction']==r['expected'] for r in p) for p in pairs)/len(pairs) if pairs else None,
        'contrast_both_queries_correct':sum(all(r['prediction']==r['expected'] for r in pair) for pair in contrast_pairs)/len(contrast_pairs) if contrast_pairs else None,
        'invalid':sum(r['prediction']=='INVALID' for r in rows),
        'expected_counts':dict(counts),'prediction_counts':dict(Counter(r['prediction'] for r in rows)),
        'seconds':sum(r.get('seconds',0) for r in rows)}


def safe_power():
    r=subprocess.run(['pmset','-g','batt'],capture_output=True,text=True)
    match=re.search(r'(\d+)%;',r.stdout)
    if match and int(match[1])<=18:
        status=r.stdout.split(';')[1].strip() if ';' in r.stdout else ''
        if status not in ('charging','charged'):
            raise RuntimeError('Battery at or below 18%; evaluation safely checkpointed')



def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/dev.jsonl');ap.add_argument('--out',required=True)
    ap.add_argument('--adapter');ap.add_argument('--limit',type=int);ap.add_argument('--style',default='table',choices=['table','prose','symbols'])
    ap.add_argument('--mode',choices=['choice','generate'],default='choice');ap.add_argument('--ks',default='8');ap.add_argument('--backend',default='mlx',choices=['mlx','api']);ap.add_argument('--api-model')
    args=ap.parse_args();out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    source=Path(args.data);data=[json.loads(l) for l in source.read_text().splitlines()]
    if args.limit:data=data[:args.limit]
    ks=[int(x) for x in args.ks.split(',')]
    cfg=vars(args)|{'base':BASE,'base_revision':REVISION,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    if args.adapter:
        cfg['adapter_sha256']=hashlib.sha256((Path(args.adapter)/'adapters.safetensors').read_bytes()).hexdigest()
    meta=out.with_suffix('.meta.json')
    if meta.exists() and json.loads(meta.read_text())!=cfg:raise ValueError('Evaluation config changed; use a new output file')
    meta.write_text(json.dumps(cfg,indent=2))
    done=[json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    completed={(r['episode_id'],r['k']) for r in done};oracle=Oracle()
    if args.backend=='mlx':
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        model,tokenizer=load(str(resolve_base_path()),adapter_path=args.adapter)
        from .template import configure_tokenizer
        configure_tokenizer(tokenizer)
        label_ids={c:tokenizer.encode(c,add_special_tokens=False) for c in ('A','B','U')}
        assert all(len(v)==1 for v in label_ids.values()),label_ids
        sampler=make_sampler(temp=0)
    else:
        if args.mode=='choice':raise ValueError('API mode requires --mode generate; constrained logprob scoring is local only')
        import httpx
        client=httpx.Client(timeout=120)
    for row in data:
        for k in ks:
            if (row['id'],k) in completed:continue
            if k>len(row['observations']):raise ValueError('Too many demonstrations')
            safe_power()
            episode=dict(row);episode['observations']=row['observations'][:k];episode['k']=k
            expected,n=oracle.answer(episode['observations'],episode['query'])
            msg=messages(episode,style=args.style,completion=False)
            started=time.time();probs={};token_count=None
            if args.backend=='mlx':
                prompt=tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                token_count=len(tokenizer.encode(prompt))
                text=''
                if args.mode=='choice':
                    ids=tokenizer.encode(prompt,add_special_tokens=False)
                    logits=model(mx.array([ids]))[0,-1]
                    choices=mx.array([label_ids[c][0] for c in ('A','B','U')])
                    p=mx.softmax(logits[choices]);mx.eval(p)
                    probs=dict(zip(('A','B','U'),p.tolist()))
                    text=max(probs,key=probs.get)
                    del logits, p
                else:
                    for idx,response in enumerate(stream_generate(model,tokenizer,prompt,max_tokens=256,sampler=sampler)):
                        text+=response.text
                        if idx==0:
                            probs={c:float(mx.exp(response.logprobs[v[0]]).item()) for c,v in label_ids.items()}
                mx.clear_cache()
            else:
                endpoint=os.environ.get('TROUBADOUR_API_BASE','').rstrip('/')
                if not endpoint:raise ValueError('Set TROUBADOUR_API_BASE to an OpenAI-compatible API base URL')
                r=client.post(endpoint+'/chat/completions',headers={'Authorization':'Bearer '+os.environ['TROUBADOUR_API_KEY']},json={'model':args.api_model,'messages':msg,'temperature':0,'max_tokens':256})
                r.raise_for_status();text=r.json()['choices'][0]['message']['content']
            result={'episode_id':row['id'],'society_pair':row['society_pair'],'family':row['family'],'k':k,'style':args.style,'expected':expected,'prediction':parse_answer(text),'raw':text,'label_probs':probs,'consistent_rules':n,'prompt_tokens':token_count,'seconds':time.time()-started}
            with out.open('a') as f:f.write(json.dumps(result)+'\n');f.flush()
            done.append(result)
            if len(done)%10==0:print(json.dumps({'completed':len(done),'latest':text,'summary':summarize(done)}),flush=True)
    report={'overall':summarize(done),'by_k':{str(k):summarize([r for r in done if r['k']==k]) for k in ks}}
    out.with_suffix('.summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
