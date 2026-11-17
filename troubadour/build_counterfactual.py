"""Training groups that prevent a query-blind answer from fitting all examples."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from .worlds import Oracle,complement,messages,prompt,rules,split_manifest,support,write_jsonl


def build_groups(seed=19001,tables_per_world=3):
    rng=random.Random(seed);oracle=Oracle();families={r.mask:r.family for r in rules()}
    groups=[];excluded=[]
    for mask in split_manifest()['train']:
        if not support(mask):excluded.append(mask);continue
        for _ in range(tables_per_world):
            for attempt in range(2000):
                states=rng.sample(range(32),8);obs=[(s,(mask>>s)&1) for s in states]
                queries={c:[] for c in 'ABU'}
                for q in range(32):
                    if q not in states:
                        answer,n=oracle.answer(obs,q);queries[answer].append((q,n))
                if all(queries.values()):break
            else:raise RuntimeError('Could not construct query-dependent training group')
            group=hashlib.sha256(json.dumps([mask,obs]).encode()).hexdigest()[:20]
            rows=[]
            for c in 'ABU':
                q,n=rng.choice(queries[c])
                row={'rule_mask':mask,'observations':obs,'query':q,'answer':c,'consistent_rules':n,'k':8,'id':group+'-'+c,'society_pair':mask,'family':families[mask],'contrast_group':group+'-original'}
                reverse=complement(row,oracle);reverse['contrast_group']=group+'-reverse'
                rows.extend([row,reverse])
            groups.append(rows)
    # Keep 20% ordinary balanced groups to include examples with unique rules.
    original=[json.loads(l) for l in Path('data/choice_episodes.jsonl').read_text().splitlines()]
    pools={c:[r for r in original if r['answer']==c] for c in 'ABU'}
    mixed_count=len(groups)//4
    for c in pools:rng.shuffle(pools[c])
    for i in range(mixed_count):
        groups.append([pools[c][2*i+j] for j in range(2) for c in 'ABU'])
    rng.shuffle(groups)
    return groups,excluded,mixed_count


def main():
    groups,excluded,mixed=build_groups();rows=[r for g in groups for r in g]
    signatures=[prompt(r) for r in rows]
    assert len(signatures)==len(set(signatures)), 'Duplicate public training prompt'
    for g in groups:assert Counter(r['answer'] for r in g)==Counter({'A':2,'B':2,'U':2})
    write_jsonl('data/counterfactual_episodes.jsonl',rows)
    write_jsonl('data/counterfactual/train.jsonl',({'messages':messages(r)} for r in rows))
    valid=Path('data/choice/valid.jsonl').read_text()
    Path('data/counterfactual/valid.jsonl').write_text(valid)
    for split in ('train','valid'):
        examples=[json.loads(l) for l in Path('data/counterfactual',split+'.jsonl').read_text().splitlines()]
        for example in examples:
            answer=example['messages'][-1]['content']
            example['messages'][0]['content']+=' In this training exercise, a certified answer key is supplied. Copy it exactly.'
            example['messages'][1]['content']+='\nCertified answer key: '+answer
        write_jsonl(f'data/counterfactual_control/{split}.jsonl',examples)
    report={'rows':len(rows),'groups_of_six':len(groups),'query_dependent_groups':len(groups)-mixed,'ordinary_balanced_groups':mixed,'labels':dict(Counter(r['answer'] for r in rows)),'excluded_constant_pairs_from_query_dependent_groups':excluded,'seed':19001,'purpose':'Prepared training recipe; not evidence of improved model performance'}
    Path('data/counterfactual_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
