"""Paired, society-clustered uncertainty and transparent baseline comparisons."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .evaluate import summarize


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def compare(base, candidate, repeats=10000, seed=91):
    b={(r['episode_id'],r['k'],r['style']):r for r in base}
    c={(r['episode_id'],r['k'],r['style']):r for r in candidate}
    if b.keys()!=c.keys():raise ValueError('Comparisons require exactly matching examples')
    clusters=defaultdict(list)
    for key in b:
        if b[key]['expected']!=c[key]['expected']:raise ValueError('Expected answers changed')
        clusters[b[key]['society_pair']].append(int(c[key]['prediction']==c[key]['expected'])-int(b[key]['prediction']==b[key]['expected']))
    vals=np.array([np.mean(v) for v in clusters.values()]);rng=np.random.default_rng(seed)
    boot=np.mean(rng.choice(vals,size=(repeats,len(vals)),replace=True),axis=1)
    return {'base':summarize(base),'candidate':summarize(candidate),'society_pairs':len(vals),'cluster_mean_gain':float(vals.mean()),'cluster_bootstrap_95ci':[float(x) for x in np.quantile(boot,[.025,.975])],'method':'paired bootstrap over complement-society clusters, equal cluster weighting; 10000 replicates'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=compare(read(a.base),read(a.candidate));Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
