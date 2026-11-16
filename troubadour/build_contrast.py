"""Same evidence, opposite determinate queries: query-blind strategies cannot pass both."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from .worlds import Oracle,rules,support,complement,write_jsonl,split_manifest


def build(masks,seed):
    rng=random.Random(seed);oracle=Oracle();rows=[];skipped=[]
    families={r.mask:r.family for r in rules()}
    for mask in masks:
        if not support(mask):skipped.append(mask);continue
        for _ in range(2000):
            states=rng.sample(range(32),8)
            obs=[(s,(mask>>s)&1) for s in states]
            by={'A':[],'B':[]}
            for q in range(32):
                if q in states:continue
                ans,n=oracle.answer(obs,q)
                if ans in by:by[ans].append((q,n))
            if by['A'] and by['B']:break
        else:skipped.append(mask);continue
        group=hashlib.sha256(json.dumps([mask,obs]).encode()).hexdigest()[:16]
        for answer in ('A','B'):
            q,n=rng.choice(by[answer])
            row={'rule_mask':mask,'observations':obs,'query':q,'answer':answer,'consistent_rules':n,'k':8,'id':group+'-'+answer,'society_pair':mask,'family':families[mask],'contrast_group':group+'-original'}
            reverse=complement(row,oracle);reverse['contrast_group']=group+'-reversed'
            rows.extend([row,reverse])
    return rows,skipped


def main():
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
    if a.final and not Path('artifacts/selection.json').exists():raise SystemExit('Freeze selection before generating final contrast')
    manifest=split_manifest()
    if a.final:
        selection=json.loads(Path('artifacts/selection.json').read_text())
        checksum=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        if selection.get('split_sha256')!=checksum:raise SystemExit('Selection must freeze the exact split manifest hash')
    key='final' if a.final else 'dev'
    rows,skipped=build(manifest[key],98218 if a.final else 98117)
    write_jsonl(f'data/{key}_contrast.jsonl',rows)
    report={'rows':len(rows),'society_pairs':len(manifest[key])-len(skipped),'skipped_unconstructible_or_constant':skipped,'guarantee':'Every within-world pair shares observations and has opposite determinate answers'}
    Path(f'data/{key}_contrast_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
