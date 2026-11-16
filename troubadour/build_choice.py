"""Reproduce balanced decision-focused training and its answer-key control."""
from collections import Counter
import json
from pathlib import Path
import random
from .worlds import messages,write_jsonl


def main():
    rows=[json.loads(l) for l in Path('data/train_episodes.jsonl').read_text().splitlines()]
    groups={c:[r for r in rows if r['k']==8 and r['answer']==c] for c in 'ABU'}
    rng=random.Random(919);n=min(map(len,groups.values()));selected=[]
    for c in 'ABU':rng.shuffle(groups[c]);selected.extend(groups[c][:n])
    rng.shuffle(selected)
    write_jsonl('data/choice_episodes.jsonl',selected)
    train=[{'messages':messages(r)} for r in selected]
    valid=[json.loads(l) for l in Path('data/sft/valid.jsonl').read_text().splitlines()]
    write_jsonl('data/choice/train.jsonl',train);write_jsonl('data/choice/valid.jsonl',valid)
    for split,examples in [('train',train),('valid',valid)]:
        control=[]
        for example in examples:
            row=json.loads(json.dumps(example));answer=row['messages'][-1]['content']
            row['messages'][0]['content']+=' In this training exercise, a certified answer key is supplied. Copy it exactly.'
            row['messages'][1]['content']+='\nCertified answer key: '+answer
            control.append(row)
        write_jsonl(f'data/choice_control/{split}.jsonl',control)
    print(json.dumps({'rows':len(selected),'labels':dict(Counter(r['answer'] for r in selected)),'seed':919,'k':8},indent=2))

if __name__=='__main__':main()
