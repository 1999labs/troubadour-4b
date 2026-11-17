"""Freeze a development-selected checkpoint before generating reserved encounters."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone
from .evaluate import BASE, REVISION
from .template import renderer_sha256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adapter',required=True)
    p.add_argument('--checkpoint',default='adapters.safetensors')
    p.add_argument('--config',required=True)
    p.add_argument('--training-data',required=True)
    p.add_argument('--recipe',required=True)
    p.add_argument('--microsteps',type=int,required=True)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--reason',required=True)
    a=p.parse_args()
    selection=Path('artifacts/selection.json')
    if selection.exists():raise SystemExit('Selection is already frozen; do not retune against reserved tests')
    if any(Path('data',s+'.jsonl').exists() for s in ['final','final_transfer','final_contrast']):
        raise SystemExit('Reserved encounters already exist; this is no longer a prospective selection')
    criteria=json.loads(Path('SUCCESS_CRITERIA.json').read_text())
    if a.seed!=criteria['published_seed_preselected']:
        raise SystemExit('Release seed differs from the prospectively selected seed')
    source=Path(a.adapter);weights=source/a.checkpoint
    split=json.loads(Path('data/split_manifest.json').read_text())
    payload={
        'frozen_at':datetime.now(timezone.utc).isoformat(),
        'base':BASE,'base_revision':REVISION,'recipe':a.recipe,'seed':a.seed,
        'checkpoint_microsteps':a.microsteps,'source_checkpoint':str(weights),
        'adapter_sha256':sha(weights),'adapter_config_sha256':sha(source/'adapter_config.json'),
        'training_config':a.config,'training_config_sha256':sha(a.config),
        'training_data':a.training_data,'training_data_sha256':sha(a.training_data),
        'split_sha256':hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest(),
        'criteria_sha256':sha('SUCCESS_CRITERIA.json'),
        'base_calibration_sha256':sha('artifacts/base-calibration.json'),
        'renderer_sha256':renderer_sha256(),
        'runtime_versions':{n:version(n) for n in ['mlx','mlx-lm','transformers']},
        'selection_reason':a.reason,
        'final_data_seen':False,
    }
    dest=Path('artifacts/selected');dest.mkdir(exist_ok=True)
    shutil.copy2(weights,dest/'adapters.safetensors')
    shutil.copy2(source/'adapter_config.json',dest/'adapter_config.json')
    selection.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
