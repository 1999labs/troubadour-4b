# Development record — experiment in progress

Research question: **Can post-training teach a model street smarts so that it can learn and apply unfamiliar social rules?**

No successful final model is claimed yet. The reserved final encounters have not been generated. These are development results and engineering checks, recorded on 2026-11-16; they must not be cited as final benchmark performance.

| Method | Development accuracy | Answerable accuracy | Uncertainty recall |
|---|---:|---:|---:|
| Original Qwen, identical constrained-choice prompt | 42.50% | 26.25% | 75.00% |
| Original Qwen, global uncertainty bias fitted on development | 44.58% | — | — |
| Nearest-observation heuristic | 62.50% | 78.75% | 30.00% |
| Query-blind majority-label heuristic | 48.33% | 61.25% | 22.50% |
| First 300-example choice-training pilot | 33.33% | 0.00% | 100.00% |

The full development set has240 episodes from40 complementary society pairs, with80 examples per label. The calibrated Qwen row is fitted and scored on development, so it is an optimistic tuning result; its frozen bias will later be applied to reserved test examples. The first choice pilot answered U on all240 development questions. Its raw change from Qwen is−9.17 percentage points, with paired society-cluster bootstrap95% interval[−15.42,−2.92]. This is a failure, not an improvement.

An earlier generic SFT pilot was interrupted by a Metal GPU/display-context error after85 updates. Its saved update50 checkpoint answered U on all48 exploratory pilot questions. The cached model's original template also required an explicit correction: it inserted thinking markup in supervised completions but not the generation prefix. The corrected template preserves the original evaluation prefix. Failed attempts are disclosed rather than silently discarded.

The next planned recipe starts from original Qwen, uses a much smaller learning rate and averages8 examples per optimizer update. It trains on the same balanced2,850-example choice subset for6,000 example presentations. This is still supervised LoRA training of the original model's answer logits, with no extra prediction head or reference-solver features at inference. See configs/choice-slow-seed17.yaml.

Caching activations from the frozen lower28 layers makes repeated training passes cheaper. A GPU audit on two examples with nonzero pilot adapters found exactly matching probabilities, loss, and adapter gradients between cached and full computation. This is a numerical implementation check, not evidence of task improvement. The ordinary scorer also matched full-model scalar scoring on four audit examples. Thirteen unit tests cover rule generation, oracle agreement, complement splits, input leakage, scoring, and statistical summaries.

The opposite-query diagnostic has160 development episodes: two questions share the same observations but require opposite answers, with complementary worlds included. It will test whether gains depend on using the query rather than predicting from observation-label counts alone. It is a separate stress distribution, not a replacement for the primary evaluation.

Final evaluation, independent seeds, matched training control, learning curves, rendering robustness, instruction smoke checks, and model publication remain pending. The source and dataset preview are useful for inspection and reproduction; their presence does not mean the research hypothesis has been established.
