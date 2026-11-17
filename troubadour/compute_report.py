"""Report measured job wall time, distinguishing pauses and logical token counts."""
import json
from pathlib import Path

TRAINING = ['sft-seed17','sft-v2-seed17','sft-v3-recovery','choice-seed17','choice-slow-seed17','counterfactual-small-seed17','balanced-rate-seed17']


def main():
    hardware=json.loads(Path('artifacts/hardware.json').read_text())
    selection=json.loads(Path('artifacts/selection.json').read_text())
    config=json.loads(Path('artifacts/selected/adapter_config.json').read_text())
    pauses=[json.loads(l) for l in Path('runs/training_pauses.jsonl').read_text().splitlines()]
    jobs=[]
    for name in TRAINING:
        path=Path('runs',name+'.status.json')
        if not path.exists():continue
        s=json.loads(path.read_text())
        if s['status']=='running':raise ValueError('Finish training before reporting complete compute')
        paused=sum(r['paused_seconds'] for r in pauses if r['trainer_pid']==s.get('pid'))
        jobs.append({'name':name,'status':s['status'],'wall_seconds':s['elapsed_seconds'],'recorded_pause_seconds':paused,'wall_seconds_excluding_recorded_pauses':max(0,s['elapsed_seconds']-paused)})
    microsteps=selection['checkpoint_microsteps'];batch=config['batch_size'];accum=config['grad_accumulation_steps']
    assert microsteps%accum==0
    presentations=microsteps*batch
    report={'hardware':hardware,'training_jobs':jobs,'selected_checkpoint':{'microsteps':microsteps,'batch_size':batch,'gradient_accumulation_steps':accum,'optimizer_updates':microsteps//accum,'example_presentations':presentations,'input_tokens_per_example':320,'logical_input_token_positions':presentations*320,'supervised_decisions':presentations},'limits':'Job wall time includes loading, validation and cache I/O. Subtracted pauses are explicitly recorded SIGSTOP intervals. This is not GPU-kernel time. Untracked setup/audits are not included. No power/energy or measured peak-memory estimate is claimed.'}
    evaluation_jobs=[]
    for condition in ['final','final-transfer','final-contrast','final-curve','final-prose','final-symbols']:
        for model in ['base','candidate']:
            name=model+'-'+condition;status_path=Path('runs',name+'.status.json');predictions=Path('runs',name+'.jsonl')
            if not status_path.exists() or not predictions.exists():continue
            status=json.loads(status_path.read_text())
            if status['status']!='complete':continue
            rows=[json.loads(line) for line in predictions.read_text().splitlines()]
            evaluation_jobs.append({'name':name,'condition':condition,'model':model,'inputs':len(rows),'logical_input_tokens':sum(r['prompt_tokens'] for r in rows),'wall_seconds':status['elapsed_seconds']})
    report['evaluation_jobs']=evaluation_jobs
    report['evaluation_timing_note']='Completed reserved scoring jobs only. Base runs populate caches reused by candidate runs; these are resource records, not a fair model-speed comparison. Mixed-prefix audits and general-instruction smoke checks are excluded.'
    Path('results/compute.json').write_text(json.dumps(report,indent=2)+'\n')
    text='# Compute and training accounting\n\nAll project training and evaluation used the local Apple M2 with 16 GiB unified memory. There was no cloud GPU, paid teacher API, or reinforcement-learning rollout service. The measured environment was Python 3.12.15, MLX 0.32.4, MLX-LM 0.32.2 and Transformers 5.23.0 on macOS 26.6.2.\n\n'
    text+='| Training trial | Recorded status | Wall time | Recorded pauses | Wall time minus pauses |\n|---|---|---:|---:|---:|\n'
    for r in jobs:
        text+=f"| {r['name']} | {r['status']} | {r['wall_seconds']/60:.1f} min | {r['recorded_pause_seconds']/60:.1f} min | {r['wall_seconds_excluding_recorded_pauses']/60:.1f} min |\n"
    text+='\nThese are full attempted-trial durations, including loading, validation and cache I/O. A selected early checkpoint does not retain updates performed after it. Failed, interrupted and superseded trials still consumed time and are shown. Pauses allowed another diagnostic or training process to use the GPU sequentially; their time is removed from the suspended trial rather than counted twice as active training. These measurements are not GPU-kernel time or an all-inclusive elapsed project duration. A small adapter still requires substantial computation through the large base network.\n\n'
    text+=f"The released checkpoint retains {microsteps:,} microsteps, {microsteps//accum:,} optimizer updates and {presentations:,} example presentations. Each training input prefix has 320 tokens, giving {presentations*320:,} logical input-token positions and {presentations:,} supervised A/B/U decisions. Frozen-layer cache hits reduce physical computation; these token counts do not imply every base layer was recomputed for each position. The trainer's displayed token counter counts supervised decisions for this custom loss, not all prompt tokens.\n\n"
    text+='Training configured a 10 GiB MLX memory limit and a 256 MiB allocator-cache limit. These are configured limits, not measured peak usage. The frozen activation cache uses additional disk space. Training and scoring on another machine may have very different timings, particularly with a cold cache. No energy estimate is claimed.\n\n'
    text+='Exact configurations, pinned base revision, weight checksums and runtime versions are included with the release. Weight checkpoints contain adapter parameters; they are not complete optimizer-state resumes. An interrupted run restarted from weights would not be an exact continuation.\n'
    if evaluation_jobs:
        text+='\n## Reserved evaluation compute\n\nThese are completed scoring-job wall times, including model loading and cache I/O. **They are not a fair speed comparison:** original Qwen runs first and populates frozen-prefix caches that the adapter run reuses. Mixed-prefix audits and the general-instruction smoke checks are excluded.\n\n| Scoring job | Inputs | Wall time |\n|---|---:|---:|\n'
        for row in evaluation_jobs:
            text+=f"| {row['name']} | {row['inputs']:,} | {row['wall_seconds']/60:.1f} min |\n"
        text+=f"\nThese {len(evaluation_jobs)} completed scoring jobs total {sum(row['wall_seconds'] for row in evaluation_jobs)/60:.1f} minutes. Inference uses the original full model plus the selected adapter, with optional frozen-prefix reuse. The scorer evaluates A/B/U next-token probabilities; it does not generate a chain of thought or call an external service.\n"
    Path('COMPUTE.md').write_text(text)
    print('Wrote COMPUTE.md and results/compute.json from completed job records.')

if __name__=='__main__':main()
