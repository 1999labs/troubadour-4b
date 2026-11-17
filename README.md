# Troubadour

**Can post-training teach a model street smarts so that it can learn and apply unfamiliar social rules?**

Model: **Troubadour-4B**. Dataset: **Troubadour**. Evaluation: **Troubadour-Bench**.

[Model repository](https://huggingface.co/noahmclaughlin/Troubadour-4B) · [Dataset](https://huggingface.co/datasets/noahmclaughlin/Troubadour) · [Prospective experiment plan](RESEARCH_PLAN.md)

Here, **street smarts** means noticing local conventions, applying what the evidence supports, and recognizing when you have not seen enough to know. This is a controlled, synthetic convention-inference experiment. The model sees a few encounters from an unfamiliar settlement and interprets a new encounter as respect (A), breach (B), or insufficient evidence (U). A program determines what follows from the visible evidence under an explicitly disclosed finite rule grammar. No language model judges the answers.

**Status: experiment in progress. No final performance claim or model release yet.** See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for development results and [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the prospective design and recorded amendments.

For an intuitive introduction, read [a visit to an unfamiliar settlement](EXAMPLE.md). It shows how the same observations can justify one answer while leaving another unresolved.

## What is being tested?

There are 612 unique possible conventions over five binary encounter features. Training and evaluation societies have different complete truth tables; complement worlds stay together in one split. Some feature combinations are reserved for transfer tests. Queries never simply repeat a demonstrated complete encounter.

The task is deliberately bounded. It is not evidence about real cultures, consciousness, human theory of mind, or broad social intelligence. A symbolic solver solves this known grammar exactly; the research question is whether small-model post-training improves inference from examples.

## Setup on Apple silicon

Use Python 3.11 or newer, with sufficient memory and working power for training. The development machine is an Apple M2 with 16 GiB unified memory.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -c requirements.lock -e '.[mac,dev]'
pytest -q
python -m troubadour.build --train-size 6000
python -m troubadour.audit
```

The base is `mlx-community/Qwen3-4B-Instruct-2507-4bit`, pinned to revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. The predictor and evaluator download that exact revision if it is absent.

## Evaluate the original model

```bash
python -m troubadour.evaluate_batch --data data/dev.jsonl --out runs/base-dev.jsonl
```

This uses the model's next-token probabilities restricted to A/B/U. Both base and adapted models use the exact same prompt and decoding. It is not unrestricted essay generation. The optimized batched scorer is checked against scalar full-model logits. Labels' normalized probabilities are not automatically calibrated confidence estimates.

## Train an adapter

```bash
python -m troubadour.run_job --name sft-v2-seed17 -- \
  .venv/bin/python -m troubadour.train --config configs/sft-seed17.yaml
python -m troubadour.evaluate_batch --data data/dev.jsonl \
  --out runs/candidate-dev.jsonl --adapter artifacts/sft-v2-seed17
python -m troubadour.analyze --base runs/base-dev.jsonl \
  --candidate runs/candidate-dev.jsonl --out artifacts/comparison.json
```

Use `troubadour.train`, not the raw MLX LoRA command, to apply the verified non-thinking ChatML template. The cached model's template inserts thinking markup in assistant training completions but not its generation prefix. Our explicit template keeps the inference prefix unchanged and makes the supervised target begin at the actual answer letter. Original cached model files are never edited.

`run_job` saves logs/status and stops compute on a discharging battery at or below 18%. Training saves checkpoints every 50 updates. Interrupted jobs are not claimed complete, and resuming from weights is not the same as restoring optimizer state.

## Controls and reporting

Report raw Qwen, a development-fitted scalar adjustment to its uncertainty logit, a nearest-example heuristic, the trained model, and a matched training control. All claimed final comparisons use reserved societies. The input-only baseline implementations are in `troubadour/baselines.py`. The nearest-example baseline finds every observation at minimum Hamming distance from the query and answers only when all of those neighbors agree; conflicting nearest labels yield U. The majority-label baseline ignores the query and returns U on a tied vote. The public-grammar solver is the exact reference upper bound.

Uncertainty intervals are paired and clustered by complementary society pairs rather than treating correlated examples as independent.

Zero examples make U trivially correct. Therefore learning curves emphasize justified resolution on a fixed cohort and show oracle-resolvable coverage, answerable accuracy, and unwarranted certainty separately. Do not interpret zero-shot U accuracy as successful convention learning.

Final encounters are generated only after `artifacts/selection.json` freezes the recipe and exact split hash. Final results must not be used to select new checkpoints.

## Demo

```bash
python app.py --adapter artifacts/sft-v2-seed17
```

The demo runs locally. Random demo worlds can overlap training rules and are not a substitute for the held-out benchmark. Its answer checker is used only after model inference, to display feedback.

## Related work

[MetaICL](https://aclanthology.org/2022.naacl-main.201/) motivates training across tasks so that a model can learn a new task from examples without changing its weights during inference. [Symbol Tuning](https://arxiv.org/abs/2305.08298) and [Larger language models do in-context learning differently](https://arxiv.org/abs/2303.03846) motivate controls for label meanings and reliance on examples. [Garg et al.](https://arxiv.org/abs/2208.01066) study learning function classes in context with synthetic tasks.

Here we use fictional social language, exact underdetermination labels, paired queries, and a local small-adapter experiment. The project does not claim to invent learning-to-learn or synthetic rule induction. Scores apply only to this task and protocol.

## License

Original code and synthetic data: Apache-2.0. Qwen base-model weights retain their upstream Apache-2.0 license. The adapter's actual hardware, training procedure, results, and limits will be documented in its release card.

## Current decision-focused pilot

The initial generic SFT checkpoint collapsed to U on all 48 pilot questions. A separate pilot uses balanced eight-observation episodes and trains the conditional A/B/U decision directly, with the same LoRA architecture and inference path:

```bash
python -m troubadour.build_choice
python -m troubadour.run_job --name choice-seed17 -- \
  .venv/bin/python -m troubadour.train_choice --config configs/choice-seed17.yaml
python -m troubadour.evaluate_batch --out runs/choice-dev.jsonl \
  --adapter artifacts/choice-seed17 --batch-size 2
```

This is supervised token classification, not general response SFT. It does not train end-of-answer formatting. The model still supplies the answer probabilities; the reference solver never participates in inference. The balanced subset has 2,850 episodes (950 per label), and is exported as `train_choice` if this recipe is released. Shorter observation counts are evaluated as transfer.

A macOS GPU/display contention error interrupted the initial training run. Training entry points use the process-local workaround recommended by [MLX maintainers](https://github.com/ml-explore/mlx/issues/3267); no global OS settings are changed. Failed runs and their retained checkpoints are recorded separately from successful updates.

## What “learning” means here

During post-training, Troubadour practices on many invented societies and a small adapter changes the model's behavior. During evaluation, its weights stay fixed. It receives examples from a society excluded from training and must use those examples to interpret a new encounter. The experiment asks whether practice makes that second kind of learning—learning from the current examples—better.

Sometimes the right response is “I cannot tell yet.” If two permitted customs explain every observation but disagree about the new encounter, the benchmark's correct answer is U. Guessing the hidden society's answer is not enough. This gives us an objective way to test both applying a rule and recognizing missing evidence.

Troubadour-Bench also asks opposite questions about the same observations. A model that ignores the query cannot get both right. This stress test helps distinguish actual use of the examples from simple answer-frequency shortcuts.

## Longer cached training recipe

After building the training data, the optional cached trainer reuses the unchanged lower28 layers. Run its equivalence audit first:

```bash
python -m troubadour.build_choice
python -m troubadour.verify_cached_training
python -m troubadour.run_job --name choice-slow-seed17 -- \
  .venv/bin/python -m troubadour.train_cached_choice --config configs/choice-slow-seed17.yaml
```

This recipe processes6,000 examples with8 examples averaged per optimizer update. It is still under development. Checkpoints are saved every50 microsteps; no result is promised by running this command. The cache is disposable and can occupy approximately5GB. Neither the base weights nor the model architecture change when caching is enabled.

A split by rule identity does not by itself prove that a model learned a general algorithm. It may learn a shortcut that transfers across related Boolean rules. Opposite-query checks, complementary worlds, evidence curves and alternative renderings help test those explanations; they do not establish a neural mechanism.

## Prepared counterfactual practice

An optional training recipe asks different questions about the same observations, with A/B/U all represented in each group. It keeps six examples together and averages their gradients in one update. This prevents an observation-only guess from fitting the entire group. The main held-out benchmark stays unchanged. The smaller two-layer pilot completed, but its best full development accuracy was 43.75% versus original Qwen 42.5%, with substantially worse uncertainty recall. The larger eight-layer configuration below remains unrun. No performance gain is claimed for it.

```bash
python -m troubadour.build_counterfactual
python -m troubadour.run_job --name counterfactual-seed17 -- \
  .venv/bin/python -m troubadour.train_grouped_choice --config configs/counterfactual-seed17.yaml
```

Do not shuffle individual rows before the grouped sampler: the generated JSONL preserves complete six-example groups. A smaller two-layer pilot configuration is also provided for development.

## Prepared optimization fallback

`configs/balanced-rate-seed17.yaml` and `troubadour.train_balanced_choice` prepare a moderate-rate alternative using the existing `train_choice` data. Its sampler forms updates with two examples of each label and preserves every row once per epoch. No additional dataset split is needed. It is a contingency if the longer run remains weak, not a measured improvement. Development checkpoints are fixed at 300 and 600 before running it.
