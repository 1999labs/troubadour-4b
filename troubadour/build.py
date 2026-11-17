import argparse
import hashlib
from collections import Counter
import json
from pathlib import Path
import random
from .worlds import *


def eval_rows(masks, seed, episodes_per_pair=4):
    rng=random.Random(seed);oracle=Oracle();rows=[]
    for mask in masks:
        for j in range(episodes_per_pair):
            # One third explicitly ambiguous; two thirds identifiable. Query state unseen.
            row=make_episode(mask,rng,oracle,8,want_known=(j%3 != 0))
            rows.extend([row,complement(row,oracle)])
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-size',type=int,default=6000);ap.add_argument('--final',action='store_true');args=ap.parse_args()
    root=Path('data');root.mkdir(exist_ok=True)
    manifest=split_manifest()
    (root/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
    if args.final:
        if not Path('artifacts/selection.json').exists():raise SystemExit('Freeze selection before generating final data')
        selection=json.loads(Path('artifacts/selection.json').read_text())
        checksum=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        if selection.get('split_sha256')!=checksum:raise SystemExit('Selection must freeze the exact split manifest hash')
        rows=eval_rows(manifest['final'],192718,3)
        transfer=eval_rows(manifest['final_transfer'],192719,3)
        write_jsonl(root/'final.jsonl',rows);write_jsonl(root/'final_transfer.jsonl',transfer)
        print('Final generated',len(rows),len(transfer));return
    oracle=Oracle();rng=random.Random(28182);train=[]
    for i in range(args.train_size//2):
        mask=rng.choice(manifest['train'])
        k=rng.choices([0,1,2,4,8,12],[1,1,1,2,7,2])[0]
        # Short prefixes are legitimately often ambiguous; reject neither by hidden label.
        want=(i%3!=0) if k>=8 else None
        row=make_episode(mask,rng,oracle,k,want)
        train.extend([row,complement(row,oracle)])
    unique={}
    for row in train:
        key=prompt(row)
        if key in unique:assert unique[key]['answer']==row['answer']
        else:unique[key]=row
    train=list(unique.values())
    dev=eval_rows(manifest['dev'],61623,3)
    transfer=eval_rows(manifest['dev_transfer'],61624,3)
    write_jsonl(root/'train_episodes.jsonl',train);write_jsonl(root/'dev.jsonl',dev);write_jsonl(root/'dev_transfer.jsonl',transfer)
    # Shuffle complements to avoid adjacent opposite targets in batches.
    rng.shuffle(train)
    write_jsonl(root/'sft/train.jsonl',({'messages':messages(r)} for r in train))
    write_jsonl(root/'sft/valid.jsonl',({'messages':messages(r)} for r in dev[:64]))
    # An explicit hidden rule would contradict U targets. The prospective
    # format control instead reveals the correct decision, including U.
    for split, examples in [('train', train), ('valid', dev[:64])]:
        control=[]
        for row in examples:
            item=messages(row)
            item[0]['content']+=' In this training exercise, a certified answer key is supplied. Copy it exactly.'
            item[1]['content']+='\nCertified answer key: '+row['answer']
            control.append({'messages':item})
        write_jsonl(root/'format_control'/f'{split}.jsonl',control)
    report={'grammar_rules':len(rules()),'society_pairs':{k:len(v) for k,v in manifest.items() if isinstance(v,list)},'rows':{'train':len(train),'dev':len(dev),'dev_transfer':len(transfer)},'train_labels':dict(Counter(r['answer'] for r in train)),'dev_labels':dict(Counter(r['answer'] for r in dev))}
    (root/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
