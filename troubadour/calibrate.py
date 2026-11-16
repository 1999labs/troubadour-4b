"""Development-only uncertainty-bias baseline; no weight training or oracle at inference."""
import argparse
import copy
import json
import math
from pathlib import Path
from .evaluate import summarize


def adjusted(rows,bias):
    out=[]
    for row in rows:
        r=dict(row)
        p=r['label_probs']
        scores={c:math.log(max(float(p[c]),1e-30))+(bias if c=='U' else 0) for c in ('A','B','U')}
        r['prediction']=max(scores,key=scores.get)
        out.append(r)
    return out


def fit(rows):
    grid=[i*.25 for i in range(-32,33)]
    results=[(summarize(adjusted(rows,b))['accuracy'],b) for b in grid]
    best=max(results,key=lambda ab:(ab[0],-abs(ab[1]),-ab[1]))
    return {'uncertainty_logit_bias':best[1],'development_accuracy':best[0],'grid':grid,'fit_on':'development only','purpose':'Detect improvements explainable by a global change in abstention threshold'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--fit-out');p.add_argument('--apply');p.add_argument('--out');a=p.parse_args()
    rows=[json.loads(l) for l in Path(a.data).read_text().splitlines()]
    if a.fit_out:
        result=fit(rows);Path(a.fit_out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    else:
        config=json.loads(Path(a.apply).read_text());result=adjusted(rows,config['uncertainty_logit_bias'])
        Path(a.out).write_text(''.join(json.dumps(r)+'\n' for r in result));print(json.dumps(summarize(result),indent=2))

if __name__=='__main__':main()
