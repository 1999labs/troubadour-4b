from collections import Counter
import json
from pathlib import Path
from .worlds import Oracle, independent_answer, split_manifest, FULL, prompt


def main():
    oracle=Oracle();manifest=split_manifest();report={};prior_prompts=set()
    for filename,key in [('train_episodes','train'),('dev','dev'),('dev_transfer','dev_transfer')]:
        rows=[json.loads(l) for l in Path('data',filename+'.jsonl').read_text().splitlines()]
        ids=set();exact=set();dupes=0
        allowed={m for p in manifest[key] for m in (p,FULL^p)}
        for i,r in enumerate(rows):
            assert r['rule_mask'] in allowed
            assert r['query'] not in [s for s,v in r['observations']]
            assert oracle.answer(r['observations'],r['query'])[0]==r['answer']
            if i%20==0:assert independent_answer(r['observations'],r['query'])==r['answer']
            signature=prompt(r)
            if signature in exact:dupes+=1
            exact.add(signature);ids.add(r['id'])
        assert not prior_prompts.intersection(exact), 'Public prompt overlap across splits'
        prior_prompts.update(exact)
        report[filename]={'rows':len(rows),'unique_ids':len(ids),'duplicate_prompts':dupes,'labels':dict(Counter(r['answer'] for r in rows))}
    Path('data/validation_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
