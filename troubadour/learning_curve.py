"""Learning curves on a fixed cohort, without rewarding trivial zero-shot abstention."""
import argparse
import json
from pathlib import Path
from .evaluate import summarize


def curve(rows):
    ks=sorted({r['k'] for r in rows});full={r['episode_id']:r for r in rows if r['k']==max(ks)}
    cohort={eid for eid,r in full.items() if r['expected']!='U'}
    report=[]
    for k in ks:
        subset=[r for r in rows if r['k']==k and r['episode_id'] in cohort]
        if len(subset)!=len(cohort):raise ValueError('Each curve point must use the same query cohort')
        unknown=[r for r in subset if r['expected']=='U']
        report.append({'k':k,'cohort_size':len(subset),'oracle_resolvable_fraction':sum(r['expected']!='U' for r in subset)/len(subset),
            'justified_correct_resolution':sum(r['expected']!='U' and r['prediction']==r['expected'] for r in subset)/len(subset),
            'unjustified_certainty_rate':sum(r['prediction']!='U' for r in unknown)/len(unknown) if unknown else None,
            'answerable_accuracy':summarize(subset)['answerable_accuracy'],'joint_accuracy_including_uncertainty':summarize(subset)['accuracy']})
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    rows=[json.loads(l) for l in Path(a.data).read_text().splitlines()]
    result=curve(rows);Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
