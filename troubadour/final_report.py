"""Score reserved test predictions against the previously frozen success criteria."""
import argparse
import hashlib
import json
from pathlib import Path
from .analyze import compare,read
from .calibrate import adjusted
from .evaluate import summarize
from .template import renderer_sha256


def checked_predictions(path, data_path, adapter_sha=None):
    rows=read(path);data=read(data_path)
    expected={r['id']:r['answer'] for r in data}
    if len(rows)!=len(expected) or {r['episode_id'] for r in rows}!=expected.keys():
        raise ValueError('Final report requires each reserved episode exactly once: '+path)
    if any(r['expected']!=expected[r['episode_id']] or r['k']!=8 or r['style']!='table' for r in rows):
        raise ValueError('Wrong targets or evaluation condition: '+path)
    meta=json.loads(Path(path).with_suffix('.meta.json').read_text())
    if meta.get('protocol')!='constrained-choice-v3' or meta.get('renderer_sha256')!=renderer_sha256():
        raise ValueError('Unverified or changed prompt renderer: '+path)
    if meta['data_sha256']!=hashlib.sha256(Path(data_path).read_bytes()).hexdigest():
        raise ValueError('Dataset checksum changed: '+path)
    if adapter_sha and meta.get('adapter_sha256')!=adapter_sha:
        raise ValueError('Wrong frozen checkpoint: '+path)
    if not adapter_sha and meta.get('adapter_sha256'):
        raise ValueError('Baseline unexpectedly has an adapter: '+path)
    return rows


def main():
    p=argparse.ArgumentParser()
    for name in ['base','candidate','base-transfer','candidate-transfer','base-contrast','candidate-contrast']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--out',default='artifacts/final_results.json');a=p.parse_args()
    selection=json.loads(Path('artifacts/selection.json').read_text())
    sha=selection['adapter_sha256']
    if selection.get('renderer_sha256')!=renderer_sha256():raise ValueError('Renderer changed after model selection')
    criteria=json.loads(Path('SUCCESS_CRITERIA.json').read_text())
    if hashlib.sha256(Path('SUCCESS_CRITERIA.json').read_bytes()).hexdigest()!=selection['criteria_sha256']:
        raise ValueError('Success criteria changed after selection')
    for prediction_path in [a.base,a.candidate,a.base_transfer,a.candidate_transfer,a.base_contrast,a.candidate_contrast]:
        meta=json.loads(Path(prediction_path).with_suffix('.meta.json').read_text())
        if meta.get('runtime_versions')!=selection.get('runtime_versions'):raise ValueError('Runtime differs from frozen selection: '+prediction_path)
    data=json.loads(Path('data/split_manifest.json').read_text())
    if hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()!=selection['split_sha256']:
        raise ValueError('Society allocation changed after selection')
    base=checked_predictions(a.base,'data/final.jsonl')
    candidate=checked_predictions(a.candidate,'data/final.jsonl',sha)
    bt=checked_predictions(a.base_transfer,'data/final_transfer.jsonl')
    ct=checked_predictions(a.candidate_transfer,'data/final_transfer.jsonl',sha)
    bc=checked_predictions(a.base_contrast,'data/final_contrast.jsonl')
    cc=checked_predictions(a.candidate_contrast,'data/final_contrast.jsonl',sha)
    primary=compare(base,candidate);transfer=compare(bt,ct);contrast=compare(bc,cc)
    calibration=json.loads(Path('artifacts/base-calibration.json').read_text())
    if hashlib.sha256(Path('artifacts/base-calibration.json').read_bytes()).hexdigest()!=selection['base_calibration_sha256']:
        raise ValueError('Baseline calibration changed after selection')
    calibrated=compare(adjusted(base,calibration['uncertainty_logit_bias']),candidate)
    b,c=primary['base'],primary['candidate']
    checks={
        'raw_qwen_gain':c['accuracy']-b['accuracy']>=criteria['minimum_raw_qwen_gain'],
        'positive_cluster_interval':primary['cluster_bootstrap_95ci'][0]>criteria['paired_cluster_bootstrap_lower_bound_must_exceed'],
        'answerable_gain':c['answerable_accuracy']-b['answerable_accuracy']>=criteria['minimum_answerable_gain'],
        'uncertainty_recall_preserved':c['uncertainty_recall']>=b['uncertainty_recall']-criteria['maximum_uncertainty_recall_drop'],
        'gain_over_calibrated_qwen':calibrated['cluster_mean_gain']>=criteria['minimum_gain_over_dev_calibrated_qwen'],
        'positive_transfer_gain':transfer['cluster_mean_gain']>0,
    }
    result={'research_question':'Can post-training teach a model street smarts so that it can learn and apply unfamiliar social rules?',
            'selection':selection,'criteria':criteria,'checks':checks,'passes_primary_criteria':all(checks.values()),
            'primary':primary,'against_dev_calibrated_qwen':calibrated,'transfer':transfer,'opposite_query_stress':contrast,
            'interpretation':'Synthetic finite-grammar rule induction in social language; no claim of real-world social competence. Independent-seed and robustness results must be reported separately.'}
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({'checks':checks,'passes':all(checks.values())},indent=2))

if __name__=='__main__':main()
