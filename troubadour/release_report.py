"""Render measured reserved results without turning diagnostics into primary claims."""
import argparse
import json
from pathlib import Path


def percent(value):
    return 'not applicable' if value is None else f'{100*value:.1f}%'


def render(result, curves, robustness, smoke):
    primary=result['primary']; b=primary['base']; c=primary['candidate']
    passed=result['passes_primary_criteria']
    outcome=('The selected checkpoint met all six predeclared primary criteria on this reserved test.' if passed else
             'The selected checkpoint did not meet all six predeclared primary criteria. This is a research checkpoint, not a validated improvement in street smarts.')
    text=f"# Troubadour-Bench results\n\n{outcome}\n\n"
    text+='**Can post-training teach a model street smarts so that it can learn and apply unfamiliar social rules?**\n\n'
    text+='Here, street smarts means inferring fictional conventions from examples and recognizing when the evidence does not determine an answer. These results concern a finite Boolean rule system expressed in social language. They do not establish real-world cultural competence.\n\n'
    text+=f"The primary test contains {b['n']} episodes from {primary['society_pairs']} reserved complementary society pairs. Each model receives eight observations and chooses A (respect), B (breach), or U (insufficient evidence). We selected the adapter using development data and froze its checksum before generating any final encounters.\n\n"
    controls=result['nonparametric_baselines']['primary']
    methods=[('Original Qwen',b),('Qwen + development-only U bias',result['against_dev_calibrated_qwen']['base']),('Nearest example',controls['nearest_example']),('Observation majority (ignores query)',controls['majority_label']),('Always U',controls['always_U']),('Troubadour-4B',c),('Exact grammar solver',controls['reference_solver'])]
    text+='| Method | Joint accuracy | Answerable accuracy | Uncertainty recall |\n|---|---:|---:|---:|\n'
    for name,metrics in methods:
        text+=f"| {name} | {percent(metrics['accuracy'])} | {percent(metrics['answerable_accuracy'])} | {percent(metrics['uncertainty_recall'])} |\n"
    lo,hi=primary['cluster_bootstrap_95ci']
    text+=f"\nTroubadour's gain over raw Qwen is **{100*primary['cluster_mean_gain']:+.1f} percentage points**, with a paired society-clustered 95% bootstrap interval of [{100*lo:+.1f}, {100*hi:+.1f}] points (10,000 resamples). This interval reflects uncertainty over sampled societies, not over independent training seeds.\n\n"
    text+='Answerable accuracy includes abstentions as errors on determinate A/B questions. Uncertainty recall measures how often the model correctly says U on underdetermined questions. The three labels are balanced by design; this is not a natural population frequency estimate.\n\n'
    names={'raw_qwen_gain':'At least +10 points over raw Qwen','positive_cluster_interval':'Paired cluster interval entirely above zero','answerable_gain':'At least +10 points answerable accuracy','uncertainty_recall_preserved':'Uncertainty recall drops no more than 10 points','gain_over_calibrated_qwen':'At least +5 points over development-calibrated Qwen','positive_transfer_gain':'Positive gain on structural transfer'}
    text+='## Frozen success criteria\n\n| Criterion | Result |\n|---|---|\n'
    for name,ok in result['checks'].items():text+=f"| {names[name]} | {'Pass' if ok else 'Fail'} |\n"
    text+='\nAll six must pass. These thresholds were not changed after final testing. The unadjusted adapter remains the primary model.\n\n'
    text+='## Transfer and query dependence\n\n| Test | Episodes | Qwen | Troubadour | Paired gain (95% cluster interval) |\n|---|---:|---:|---:|---|\n'
    for name,key in [('Unseen rule identities','primary'),('Reserved feature combinations','transfer'),('Opposite-query stress','opposite_query_stress')]:
        r=result[key];low,high=r['cluster_bootstrap_95ci']
        text+=f"| {name} | {r['base']['n']} | {percent(r['base']['accuracy'])} | {percent(r['candidate']['accuracy'])} | {100*r['cluster_mean_gain']:+.1f} [{100*low:+.1f}, {100*high:+.1f}] points |\n"
    stress=result['opposite_query_stress']
    text+=f"\nOn the stress test, getting both opposite queries right occurs for {percent(stress['base']['contrast_both_queries_correct'])} of query pairs for Qwen and {percent(stress['candidate']['contrast_both_queries_correct'])} for Troubadour. These pairs share observations and require different answers, so observation-only guesses cannot solve both. This is a selected stress distribution, not the same distribution as the primary test.\n\n"
    text+='## Calibration diagnostics\n\n'
    cal=result['secondary_both_dev_calibrated']['primary'];bias=result['calibration_biases']
    text+=f"The U-logit biases fitted on development data were {bias['base']:+.2f} for Qwen and {bias['candidate']:+.2f} for the selected adapter. Applied unchanged to the final test, the adjusted accuracies are {percent(cal['base']['accuracy'])} and {percent(cal['candidate']['accuracy'])}. These are secondary comparisons; the deployed predictor uses the unadjusted model.\n\n"
    text+=f"When U is ignored solely to compare the relative A/B scores on answerable questions, accuracy is {percent(b['conditional_AB_accuracy'])} for Qwen and {percent(c['conditional_AB_accuracy'])} for Troubadour. U-versus-best-A/B ranking AUC is {b['uncertainty_ranking_auc']:.3f} and {c['uncertainty_ranking_auc']:.3f}, respectively. AUC 0.5 means chance ranking. These metrics cannot by themselves replace joint accuracy or safe abstention.\n\n"
    text+='## Evidence curves and rendering checks\n\n'
    text+='The fixed first 60 reserved episodes are checked at 0, 1, 2, 4 and 8 observations. The plot focuses on queries answerable at eight observations, and credits a correct resolution only when the current evidence justifies it. Perfect U accuracy at zero observations is trivial and is not called successful learning.\n\n![Evidence curve](results/figures/evidence_curve.png)\n\n'
    text+='| Rendering (first 48 reserved episodes) | Qwen | Troubadour |\n|---|---:|---:|\n'
    for style,r in robustness.items():text+=f"| {style} | {percent(r['base']['accuracy'])} | {percent(r['candidate']['accuracy'])} |\n"
    text+='\nThese small, prospectively bounded subsets are exploratory. Their scores do not establish broad robustness.\n\n'
    text+=f"The 12 simple general-instruction smoke checks passed {smoke['base']['passed']}/12 for Qwen and {smoke['candidate']['passed']}/12 for Troubadour. This is too small to claim general capabilities were preserved.\n\n"
    text+='## Scope and reproducibility\n\n'
    s=result['selection']
    text+=f"Published seed: {s['seed']}. Selected recipe: `{s['recipe']}` at microstep {s['checkpoint_microsteps']}. The exact weight, recipe, data, renderer and runtime identities are in `selection.json`. Failed and weaker trials are retained in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).\n\n"
    text+='This first release is exploratory and uses one training seed. Independent-seed replication and the prepared budget-matched answer-key training control were not completed. Therefore neither a positive result nor a negative result establishes a general conclusion about post-training, a neural learning mechanism, or other model sizes. Repeated development comparisons also make development scores unsuitable as final evidence.\n\n'
    text+='Test data is public after evaluation. Future models may train on it; subsequent claims should disclose this and use genuinely held-out societies. The exact grammar solver is a transparent reference, not part of the neural predictor.\n'
    return text,methods


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='artifacts/final_results.json');a=p.parse_args()
    result=json.loads(Path(a.results).read_text())
    curves={name:json.loads(Path(f'artifacts/{name}-final-curve.json').read_text()) for name in ['base','candidate']}
    from .analyze import compare,read
    robustness={style:compare(read(f'runs/base-final-{style}.jsonl'),read(f'runs/candidate-final-{style}.jsonl')) for style in ['prose','symbols']}
    smoke={name:json.loads(Path(f'artifacts/{name}-smoke.json').read_text()) for name in ['base','candidate']}
    text,methods=render(result,curves,robustness,smoke)
    folder=Path('results');figures=folder/'figures';figures.mkdir(parents=True,exist_ok=True)
    (folder/'final.json').write_text(json.dumps(result,indent=2)+'\n')
    (folder/'secondary.json').write_text(json.dumps({'curves':curves,'rendering_checks':robustness,'smoke':smoke},indent=2)+'\n')
    Path('RESULTS.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.5))
    bars=ax.barh([n for n,_ in methods],[100*m['accuracy'] for _,m in methods],color=['#959aa4','#b1b7c1','#b08343','#c1a77d','#c7cbd2','#41745f','#d4e2d8'])
    for bar,(_,m) in zip(bars,methods):
        value=100*m['accuracy'];ax.text(min(value+1,98),bar.get_y()+bar.get_height()/2,f'{value:.1f}%',va='center',ha='right' if value>96 else 'left')
    ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xlabel('Joint A/B/U accuracy (%)');ax.set_title('Reserved societies · eight observations',loc='left');ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(figures/'comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,4.6))
    for name,label,color in [('base','Original Qwen','#818a98'),('candidate','Troubadour-4B','#41745f')]:
        rows=curves[name];ax.plot([r['k'] for r in rows],[100*r['justified_correct_resolution'] for r in rows],marker='o',label=label,color=color)
    rows=curves['base'];ax.plot([r['k'] for r in rows],[100*r['oracle_resolvable_fraction'] for r in rows],linestyle='--',label='Evidence permits an answer',color='#b08343')
    ax.set(xlabel='Observed encounters',ylabel='Fixed query cohort correctly resolved (%)',ylim=(0,105),xticks=[0,1,2,4,8],title='Using more evidence · exploratory fixed cohort')
    ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False);fig.tight_layout();fig.savefig(figures/'evidence_curve.png',dpi=180);plt.close(fig)
    print('Wrote RESULTS.md, machine-readable results, and two figures. Inspect the images and all claims before publication.')

if __name__=='__main__':main()
