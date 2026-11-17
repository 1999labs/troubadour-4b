# Compute and training accounting

All project training and evaluation used the local Apple M2 with 16 GiB unified memory. There was no cloud GPU, paid teacher API, or reinforcement-learning rollout service. The measured environment was Python 3.12.15, MLX 0.32.4, MLX-LM 0.32.2 and Transformers 5.23.0 on macOS 26.6.2.

| Training trial | Recorded status | Wall time | Recorded pauses | Wall time minus pauses |
|---|---|---:|---:|---:|
| sft-seed17 | failed | 3.5 min | 0.0 min | 3.5 min |
| sft-v2-seed17 | failed | 18.7 min | 0.0 min | 18.7 min |
| sft-v3-recovery | failed | 0.5 min | 0.0 min | 0.5 min |
| choice-seed17 | complete | 26.9 min | 0.0 min | 26.9 min |
| choice-slow-seed17 | complete | 468.4 min | 108.0 min | 360.5 min |
| counterfactual-small-seed17 | complete | 74.2 min | 0.0 min | 74.2 min |
| balanced-rate-seed17 | complete | 45.6 min | 0.0 min | 45.6 min |

These are full attempted-trial durations, including loading, validation and cache I/O. A selected early checkpoint does not retain updates performed after it. Failed, interrupted and superseded trials still consumed time and are shown. Pauses allowed another diagnostic or training process to use the GPU sequentially; their time is removed from the suspended trial rather than counted twice as active training. These measurements are not GPU-kernel time or an all-inclusive elapsed project duration. A small adapter still requires substantial computation through the large base network.

The released checkpoint retains 1,200 microsteps, 300 optimizer updates and 2,400 example presentations. Each training input prefix has 320 tokens, giving 768,000 logical input-token positions and 2,400 supervised A/B/U decisions. Frozen-layer cache hits reduce physical computation; these token counts do not imply every base layer was recomputed for each position. The trainer's displayed token counter counts supervised decisions for this custom loss, not all prompt tokens.

Training configured a 10 GiB MLX memory limit and a 256 MiB allocator-cache limit. These are configured limits, not measured peak usage. The frozen activation cache uses additional disk space. Training and scoring on another machine may have very different timings, particularly with a cold cache. No energy estimate is claimed.

Exact configurations, pinned base revision, weight checksums and runtime versions are included with the release. Weight checkpoints contain adapter parameters; they are not complete optimizer-state resumes. An interrupted run restarted from weights would not be an exact continuation.

## Reserved evaluation compute

These are completed scoring-job wall times, including model loading and cache I/O. **They are not a fair speed comparison:** original Qwen runs first and populates frozen-prefix caches that the adapter run reuses. Mixed-prefix audits and the general-instruction smoke checks are excluded.

| Scoring job | Inputs | Wall time |
|---|---:|---:|
| base-final | 306 | 13.9 min |
| candidate-final | 306 | 4.5 min |
| base-final-transfer | 150 | 7.0 min |
| candidate-final-transfer | 150 | 2.3 min |
| base-final-contrast | 204 | 9.4 min |
| candidate-final-contrast | 204 | 3.2 min |
| base-final-curve | 300 | 8.7 min |
| candidate-final-curve | 300 | 3.8 min |
| base-final-prose | 48 | 3.7 min |
| candidate-final-prose | 48 | 1.3 min |
| base-final-symbols | 48 | 2.5 min |
| candidate-final-symbols | 48 | 0.8 min |

These 12 completed scoring jobs total 61.2 minutes. Inference uses the original full model plus the selected adapter, with optional frozen-prefix reuse. The scorer evaluates A/B/U next-token probabilities; it does not generate a chain of thought or call an external service.
