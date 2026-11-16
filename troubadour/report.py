"""Create an explicitly development-only comparison report after the trial finishes."""
import json
import argparse
from pathlib import Path
from .analyze import compare,read
from .calibrate import adjusted,fit
from .evaluate import summarize


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',default='runs/choice-dev.jsonl');args=parser.parse_args()
    base=read('runs/base-dev.jsonl');candidate=read(args.candidate)
    if len(base)!=240 or len(candidate)!=240:raise SystemExit('Wait for complete development evaluations')
    report=compare(base,candidate)
    bcal=fit(base);ccal=fit(candidate)
    report['calibrated_base']=summarize(adjusted(base,bcal['uncertainty_logit_bias']))
    report['calibrated_candidate']=summarize(adjusted(candidate,ccal['uncertainty_logit_bias']))
    report['nearest_example']=json.loads(Path('artifacts/heuristic-baseline.json').read_text())['dev']
    report['phase']='development only; no final claim'
    Path('artifacts/candidate-calibration.json').write_text(json.dumps(ccal,indent=2))
    Path('artifacts/development_report.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Original Qwen','Qwen + abstention adjustment','Nearest-example heuristic','Troubadour trial']
    values=[report['base']['accuracy'],report['calibrated_base']['accuracy'],report['nearest_example']['accuracy'],report['candidate']['accuracy']]
    fig,ax=plt.subplots(figsize=(8,4.3))
    bars=ax.barh(labels,[100*x for x in values],color=['#92979f','#bbc1c9','#c29348','#517b67'])
    ax.set_xlim(0,100);ax.set_xlabel('Accuracy (%)');ax.set_title('Development results — not the final test',loc='left')
    ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
    for bar,value in zip(bars,values):ax.text(value*100+1,bar.get_y()+bar.get_height()/2,f'{value*100:.1f}%',va='center')
    fig.tight_layout();fig.savefig('artifacts/development_comparison.png',dpi=160);plt.close(fig)
    text='# Development report\n\nThese results are for model development only. The final test remains reserved.\n\n'
    text+='| Method | Accuracy |\n|---|---:|\n'+''.join(f'| {name} | {value:.1%} |\n' for name,value in zip(labels,values))
    text+=f"\nThe trial's paired gain over raw Qwen is {report['cluster_mean_gain']:.1%}; its society-clustered 95% bootstrap interval is {report['cluster_bootstrap_95ci']}. This interval describes the development sample and is not confirmatory after model selection.\n"
    text+=f"\nAnswerable accuracy: Qwen {report['base']['answerable_accuracy']:.1%}, trial {report['candidate']['answerable_accuracy']:.1%}. Uncertainty recall: Qwen {report['base']['uncertainty_recall']:.1%}, trial {report['candidate']['uncertainty_recall']:.1%}.\n"
    text+='\nOnly one training seed has been assessed here. No broad social-intelligence or real-cultural-competence claim is supported.\n'
    Path('DEVELOPMENT_RESULTS.md').write_text(text)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
