"""Audit available splits without generating or inspecting any missing test data."""
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
from .worlds import Oracle, independent_answer, split_manifest, FULL, prompt

SPLITS=[('train_episodes','train'),('choice_episodes','train'),('counterfactual_episodes','train'),
        ('dev','dev'),('dev_transfer','dev_transfer'),('dev_contrast','dev'),
        ('final','final'),('final_transfer','final_transfer'),('final_contrast','final')]


def audit_rows(rows, allowed):
    oracle=Oracle();ids=set();signatures=set()
    for i,r in enumerate(rows):
        assert r['rule_mask'] in allowed, 'Rule outside assigned society split'
        assert r['society_pair']==min(r['rule_mask'],FULL^r['rule_mask'])
        assert len(r['observations'])==r['k']
        assert len({s for s,v in r['observations']})==r['k']
        assert r['query'] not in [s for s,v in r['observations']]
        assert all(((r['rule_mask']>>s)&1)==v for s,v in r['observations'])
        answer,count=oracle.answer(r['observations'],r['query'])
        assert (answer,count)==(r['answer'],r['consistent_rules'])
        if i%20==0:assert independent_answer(r['observations'],r['query'])==r['answer']
        signature=prompt(r)
        assert signature not in signatures, 'Duplicate public prompt within a split'
        assert r['id'] not in ids, 'Duplicate episode ID within a split'
        signatures.add(signature);ids.add(r['id'])
    groups=defaultdict(list)
    for row in rows:
        if row.get('contrast_group'):groups[row['contrast_group']].append(row)
    for group in groups.values():
        assert len(group) in (2,3), 'Incomplete paired-query group'
        assert all(r['observations']==group[0]['observations'] and r['rule_mask']==group[0]['rule_mask'] for r in group)
        assert len({r['query'] for r in group})==len(group)
        assert {r['answer'] for r in group}==({'A','B'} if len(group)==2 else {'A','B','U'})
    return signatures, {'rows':len(rows),'unique_ids':len(ids),'duplicate_prompts':0,
                        'labels':dict(Counter(r['answer'] for r in rows))}


def main():
    manifest=split_manifest();report={};signatures={};allocations={}
    for filename,key in SPLITS:
        path=Path('data',filename+'.jsonl')
        if not path.exists():continue
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        allowed={m for p in manifest[key] for m in (p,FULL^p)}
        signatures[filename],report[filename]=audit_rows(rows,allowed)
        report[filename]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        allocations[filename]=key
    overlaps={}
    for left,right in combinations(signatures,2):
        shared=len(signatures[left]&signatures[right])
        if allocations[left]!=allocations[right]:
            assert shared==0, 'Public prompt overlap across independent society allocations'
        if shared:overlaps[left+' / '+right]=shared
    output={'splits':report,'overlapping_training_views_or_same_allocation':overlaps,
            'final_generated':'final' in report}
    Path('data/validation_report.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
