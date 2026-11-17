"""Stage reviewed artifacts locally; does not publish automatically."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',required=True);p.add_argument('--results',required=True);a=p.parse_args()
    result=json.loads(Path(a.results).read_text())
    if not Path('artifacts/selection.json').exists():raise SystemExit('No frozen selection')
    selection=json.loads(Path('artifacts/selection.json').read_text())
    checksum=hashlib.sha256((Path(a.adapter)/'adapters.safetensors').read_bytes()).hexdigest()
    if checksum!=selection['adapter_sha256']:raise SystemExit('Adapter differs from frozen selection')
    if result.get('selection')!=selection:raise SystemExit('Results do not match frozen selection')
    files={'train':'train_episodes','train_choice':'choice_episodes','validation':'dev','validation_transfer':'dev_transfer','test':'final','test_transfer':'final_transfer','validation_contrast':'dev_contrast','test_contrast':'final_contrast'}
    if Path('data/counterfactual_episodes.jsonl').exists():
        files['train_counterfactual']='counterfactual_episodes'
    for f in files.values():
        if not Path('data',f+'.jsonl').exists():raise SystemExit('Missing dataset split '+f)
    from .dataset_export import to_hf_dataset
    stage=Path('release');dataset=stage/'dataset';model=stage/'model'
    for folder in (dataset/'data',model/'adapter'):folder.mkdir(parents=True,exist_ok=True)
    stats={}
    for split,filename in files.items():
        rows=[json.loads(l) for l in Path('data',filename+'.jsonl').read_text().splitlines()]
        to_hf_dataset(rows).to_parquet(str(dataset/'data'/f'{split}.parquet'))
        stats[split]=len(rows)
    card=Path('DATASET_CARD.md').read_text()
    config='configs:\n- config_name: default\n  data_files:\n'+''.join(f'  - split: {s}\n    path: data/{s}.parquet\n' for s in files)
    card=card.replace('---\nlanguage:', '---\n'+config+'language:',1)
    card+='\n## Released split sizes\n\n'+''.join(f'- {s}: {n:,} episodes\n' for s,n in stats.items())
    (dataset/'README.md').write_text(card)
    for dest in (dataset,model):
        shutil.copy2('LICENSE',dest/'LICENSE')
        shutil.copy2('NOTICE',dest/'NOTICE')
        shutil.copy2('RESEARCH_PLAN.md',dest/'RESEARCH_PLAN.md')
        shutil.copy2('EXPERIMENT_LOG.md',dest/'EXPERIMENT_LOG.md')
        for name in ['RESULTS.md','EXAMPLE.md','COMPUTE.md']:
            shutil.copy2(name,dest/name)
        shutil.copytree('results/figures',dest/'results/figures',dirs_exist_ok=True)
        shutil.copy2('results/secondary.json',dest/'SECONDARY_RESULTS.json')
        shutil.copy2('results/compute.json',dest/'COMPUTE.json')
        shutil.copy2(selection['training_config'],dest/'training_config.yaml')
        shutil.copy2('artifacts/selection.json',dest/'selection.json')
        for filename in ['base-calibration.json','candidate-calibration.json']:
            shutil.copy2(Path('artifacts')/filename,dest/filename)
        shutil.copy2('data/split_manifest.json',dest/'split_manifest.json')
        shutil.copy2('SUCCESS_CRITERIA.json',dest/'SUCCESS_CRITERIA.json')
        shutil.copy2(a.results,dest/'RESULTS.json')
    shutil.copy2('MODEL_CARD.md',model/'README.md')
    source=Path(a.adapter)
    if hashlib.sha256((source/'adapter_config.json').read_bytes()).hexdigest()!=selection['adapter_config_sha256']:
        raise SystemExit('Adapter configuration differs from frozen selection')
    for name in ['adapters.safetensors','adapter_config.json']:
        shutil.copy2(source/name,model/'adapter'/name)
    shutil.copytree('troubadour',model/'troubadour',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2('app.py',model/'app.py')
    shutil.copy2('requirements.lock',model/'requirements.lock')
    shutil.copy2('troubadour/chat_template.jinja',model/'chat_template.jinja')
    example=json.loads(Path('data/dev.jsonl').read_text().splitlines()[2])
    (model/'example_episode.json').write_text(json.dumps({k:example[k] for k in ['observations','query']},indent=2))
    (model/'requirements.txt').write_text('mlx-lm==0.32.2\nhuggingface-hub==2.4.0\n')
    from .release_integrity import write_manifest
    for folder in (dataset,model):write_manifest(folder)
    integrity={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in (dataset,model) for p in folder.rglob('*') if p.is_file()}
    (stage/'integrity.json').write_text(json.dumps(integrity,indent=2))
    print('Staged dataset/model files. Write and review model README, refresh each folder SHA256SUMS with release_integrity --write, then verify fresh loading before publishing.')

if __name__=='__main__':main()
