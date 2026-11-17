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

The next planned recipe starts from original Qwen, uses a much smaller learning rate and averages8 examples per optimizer update. It trains on the same balanced2,850-example choice subset for6,000 example presentations. This is still supervised LoRA training of the original model's answer logits, with no extra prediction head or reference-solver features at inference. See configs/choice-slow-seed 17.yaml.

Caching activations from the frozen lower28 layers makes repeated training passes cheaper. A GPU audit on two examples with nonzero pilot adapters found exactly matching probabilities, loss, and adapter gradients between cached and full computation. This is a numerical implementation check, not evidence of task improvement. The ordinary scorer also matched full-model scalar scoring on four audit examples. Seventeen unit tests cover rule generation, oracle agreement, complement splits, input leakage, scoring, and statistical summaries.

The opposite-query diagnostic has160 development episodes: two questions share the same observations but require opposite answers, with complementary worlds included. It will test whether gains depend on using the query rather than predicting from observation-label counts alone. It is a separate stress distribution, not a replacement for the primary evaluation.

Final evaluation, independent seeds, matched training control, learning curves, rendering robustness, instruction smoke checks, and model publication remain pending. The source and dataset preview are useful for inspection and reproduction; their presence does not mean the research hypothesis has been established.

A counterfactual practice recipe has been prepared and independently label-checked, but has not received a performance trial yet. Its3,690 examples form615 balanced groups; most groups require different answers to different questions about exactly the same observations. A three-microstep GPU smoke test of the grouped sampler completed successfully using two adapted layers. That smoke test verifies the code path only, not learning or generalization.

## Full longer-run checkpoint 600 and counterfactual pilot

At 600 microsteps (150 optimizer updates), full 240-question development accuracy is 44.17%, compared with raw Qwen 42.5%. Answerable accuracy 56.875% versus 26.25%; uncertainty recall 18.75% versus 75%. Conditional A/B accuracy 71.875% versus 66.25%, while uncertainty-ranking AUC remains weak at 0.522 versus 0.504. This does not meet the planned success standard. No final data has been generated.

Before finishing the longer run, we temporarily preserve its optimizer state by suspending its process and test the prepared smaller counterfactual recipe: fresh base, last two blocks, rank 8, 600 microsteps/200 updates/1,200 example presentations, balanced six-example groups, seed 17. This is a different recipe, changing data, grouping and adapter depth together; it is not a clean single-factor ablation. The longer run resumes afterward. No success is claimed from training loss or compatibility checks.

## Counterfactual pilot completed; longer run resumed on AC

The two-layer counterfactual trial completed 600 microsteps/200 optimizer updates/1,200 example presentations in 74.2 minutes on the M2. At checkpoint 300, full 240-question development accuracy is 43.75%, answerable 53.75%, uncertainty recall 23.75%, conditional A/B 71.25%, and uncertainty-ranking AUC 0.5055. At checkpoint 600, accuracy is 42.917%, answerable 53.75%, uncertainty recall 21.25%, conditional A/B 70.625%, and uncertainty-ranking AUC 0.5032. Raw Qwen scores 42.5%, with 75% uncertainty recall. Neither checkpoint meets the planned standard; both paired confidence intervals include zero gain. The eight-layer longer-run 600 checkpoint remains the best joint development score so far at 44.167%.

AC power returned before selection. The original longer trainer retained its optimizer state throughout the temporary suspension and resumed after the counterfactual comparison. Planned checkpoints 1200/2100/3000 are again queued. The resource-budget early-stop branch was not used. No final data has been generated, and no successful model is claimed. The larger eight-layer counterfactual configuration remains unrun; these results do not isolate the effects of data grouping from adapter depth.

## Release source maintenance (before final data)

The generic data builder still emitted an obsolete explicit-hidden-rule control whose instructions conflicted with U targets. That unused output path has been replaced with the already planned certified-answer-key control, including U answers, at the path expected by the format-control configuration. No running experiment uses this control, no training or evaluation input was changed, and no control-training result is claimed. The choice and counterfactual answer-key controls were already prepared separately.

## Longer run: full step-1,200 development evaluation

At 1,200 microsteps (300 optimizer updates, 2,400 presentations), joint accuracy is 47.9167% versus original Qwen 42.5%. The paired gain is +5.4167 points, with a development society-clustered 95% interval of [-1.25, +12.5] points. Answerable accuracy is 61.25%, but uncertainty recall is only 21.25% versus 75% for Qwen. Conditional A/B accuracy is 71.25%, and uncertainty-ranking AUC is 0.5477. The candidate still fails the development standard. This is a development result used in model selection; final encounters remain ungenerated. The run continues to its prospectively planned 2,100 and 3,000 checkpoints.
