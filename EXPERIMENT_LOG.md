# Development record

Research question: **Can post-training teach a model street smarts so that it can learn and apply unfamiliar social rules?**

This is the development history, not the reserved-test result. All candidate comparisons below used training/development societies only. Final evaluation follows this frozen selection; its report will appear in RESULTS.md. See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the prospective design and amendments. The first release is a single-seed exploratory experiment.

## Authoritative development comparison

All rows in this table use the corrected `constrained-choice-v4-reference` scorer on the same 240 questions from 40 complementary society pairs. It scores each unpadded input using the full model output projection. There are 80 targets of each label. These are selection scores, so they must not be presented as independent evidence of improvement.

| Candidate | Microsteps | Joint accuracy | Answerable accuracy | Uncertainty recall | Paired gain (95% society-cluster interval) | All IID development criteria |
|---|---:|---:|---:|---:|---|---|
| Original Qwen | — | 42.50% | 26.25% | 75.00% | Reference | — |
| choice-slow-seed17 | 600 | 42.92% | 56.25% | 16.25% | +0.42 [-5.42, +6.25] points | Fail |
| choice-slow-seed17 | 1200 | 48.75% | 62.50% | 21.25% | +6.25 [+0.00, +12.92] points | Fail |
| choice-slow-seed17 | 2100 | 48.33% | 66.88% | 11.25% | +5.83 [-1.25, +12.92] points | Fail |
| choice-slow-seed17 | 3000 | 48.33% | 70.62% | 3.75% | +5.83 [-1.67, +13.33] points | Fail |
| counterfactual-small-seed17 | 300 | 43.75% | 53.75% | 23.75% | +1.25 [-4.58, +7.50] points | Fail |
| counterfactual-small-seed17 | 600 | 41.67% | 51.88% | 21.25% | -0.83 [-7.08, +5.83] points | Fail |
| balanced-rate-seed17 | 300 | 48.33% | 71.25% | 2.50% | +5.83 [-1.67, +13.33] points | Fail |
| balanced-rate-seed17 | 600 | 48.75% | 67.50% | 11.25% | +6.25 [-1.67, +14.17] points | Fail |

The IID checks require at least +10 points joint accuracy over raw Qwen, a positive paired interval, at least +10 points answerable accuracy, no more than a 10-point uncertainty-recall loss, and at least +5 points over development-calibrated Qwen. The full standard additionally requires positive structural-transfer gain. A candidate failing a mandatory IID check is already ineligible, so no development-transfer score is needed to establish that failure. Final transfer is still measured for the published checkpoint.

On development, the nearest-example heuristic scores 62.50%, and query-blind observation majority scores 48.33%. Original Qwen with a U-logit bias of -4.0 fitted on this same development set scores 44.58%; this is an optimistic tuning result. The frozen bias is applied unchanged to final tests. None of these controls changes the primary unadjusted neural predictor.

Selected recipe: `choice-slow-seed17`, microstep 1200, seed 17. None of the eight bounded v4 candidates met the mandatory IID development criteria. Under the recorded fallback rule, select the highest joint accuracy, then uncertainty recall, then earlier microstep. The long-run 1200 checkpoint tied balanced-rate 600 at 48.75% accuracy and won on uncertainty recall (21.25% versus 11.25%). Publish as an exploratory research checkpoint; do not claim validated improvement from development scores. No final encounters were generated or inspected. The selection record binds the actual weights and prediction file, not just a checkpoint name. It was frozen before final encounter generation. See `results/selection.json` in GitHub or `selection.json` in the Hub releases.

## Trials and changes in order

The initial generic supervised fine-tuning attempt exposed a template mismatch: the cached model's training completion inserted thinking markup that its generation prefix did not use. The corrected template preserves the original evaluation prefix. A corrected SFT run later failed with a Metal GPU/display-context error after 85 updates; its saved update-50 checkpoint answered U on all 48 pilot questions. A recovery attempt was interrupted. These attempts are not successful models and their compute is retained in [COMPUTE.md](COMPUTE.md).

A first choice-training pilot optimized only the model's A/B/U logits, with learning rate 1e-4 and single-example updates. Its 300-example checkpoint answered U on all 240 development questions: 33.33% joint accuracy and 0% answerable accuracy under the historical scorer. It was rejected. The final search did not rehabilitate that failed checkpoint.

The longer recipe started again from original Qwen: eight adapted layers, rank 8, literal MLX scale 16, peak learning rate 5e-6, batch two with four-microstep accumulation. It completed 3,000 microsteps, 750 optimizer updates and 6,000 example presentations from the same balanced 2,850-example pool. Candidate checks were bounded to microsteps 600, 1200, 2100 and 3000 before final testing. Training loss is not treated as accuracy.

The counterfactual recipe also started fresh. It used the last two adapted layers and 3,690 grouped training episodes. Most groups contain identical observations with queries requiring A, B and U, plus the complementary world; each six-example optimizer update has two targets of each label. It completed 600 microsteps, 200 updates and 1,200 presentations, with comparisons at 300 and 600. It changes data, grouping and adapter depth together, so it is not a clean causal ablation. The larger eight-layer counterfactual configuration was prepared but not run.

The longer process was temporarily suspended to permit counterfactual and numerical audits to use the GPU sequentially. Its in-memory optimizer state was retained and it resumed on AC power. The planned battery-budget truncation was not used. Recorded pauses are separated from training job wall time in the compute report.

Because no completed candidate met the IID standard, the final bounded fallback used the original 2,850 choice examples with exactly two A, two B and two U targets per optimizer update. It started fresh with eight adapted layers, peak learning rate 2e-5, batch two/accumulation three, and 600 microsteps/200 updates/1,200 presentations. Only checkpoints 300 and 600 were candidates. This changes rate, batching and schedule together; it does not isolate one mechanism.

To reproduce the selected trial from a fresh checkout after generating the data:

```bash
python -m troubadour.train_cached_choice --config configs/choice-slow-seed17.yaml
```

The complete trial runs to microstep 3000. The release uses its numbered microstep-1200 checkpoint, not the final weights. Configurations for the rejected recipes are also retained.

## Numerical correction before selection

The early optimized scorer used a shortened last-vector output projection. Historical development accuracies were 44.17% for long-run step 600, 47.92% for step 1200, and 43.75%/42.92% for counterfactual steps 300/600. These historical numbers are superseded by the v4 table above and are not mixed into selection.

A mixed-prefix audit failed the unchanged 0.001 probability tolerance. Direct diagnosis isolated a BF16 output-logit difference of 33.0 versus 33.25 between shortened and full output projections. The four-example audit's maximum probability difference was 0.007446. A later complete baseline comparison found a maximum probability difference of 0.060547, although all 240 baseline decisions stayed the same. Candidate decisions could change; this was not dismissed as harmless rounding.

Release scoring was changed to an unpadded full output projection for each input. Cache protocol 2 computes missing frozen prefixes individually so their values do not depend on other batch members. Only known unpadded 320-token entries from the earlier cache were migrated; three short diagnostic entries were excluded. All candidate selection comparisons and U-bias fits were rerun under v4 before final selection. The prompt renderer, candidate bounds, data allocation and success thresholds did not change.

Both original Qwen and the long-run step-1200 adapter matched scalar reference probabilities exactly on four mixed-prefix cases and 16 full-prefix cases each, including migrated cache entries. A separate two-example audit with nonzero adapters found identical forward loss and adapter gradients for cached and uncached training. These are finite implementation audits, not exhaustive proof. The unchanged training objective still uses its last-vector classification loss; inference uses the canonical full projection. Audit records are public in `results/audits/`.

A development watcher exited after reading a partially written status file. Replacement job status writes are atomic, and the queue tolerates the older process's transient partial reads. The trainer was not restarted by this orchestration repair.

## What was deliberately not claimed

The obsolete explicit-hidden-rule control was replaced with a certified-answer-key control that handles U targets correctly. Temporary-directory regeneration confirmed that this source repair did not alter real training/development data. The control was prepared but not trained; no control result is invented. Independent-seed replication is also absent. This bounded local release therefore cannot establish a general learning mechanism or a general conclusion about whether post-training can teach social reasoning.

The final test is used once after selection. Learning curves, rendering spot checks and general-instruction checks are small, prospectively specified secondary diagnostics. No final score selects a checkpoint, bias or training recipe. No external teacher model, human preference judge, reinforcement learning, or symbolic inference feature was used by the neural predictor.
