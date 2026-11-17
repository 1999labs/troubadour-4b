# Troubadour-4B

**Can post-training teach a model street smarts so that it can learn and apply unfamiliar social rules?**

[Model: Troubadour-4B](https://huggingface.co/noahmclaughlin/Troubadour-4B) · [Dataset: Troubadour](https://huggingface.co/datasets/noahmclaughlin/Troubadour) · [Evaluation: Troubadour-Bench](RESULTS.md)

Here, street smarts means noticing unfamiliar conventions, applying what the examples support, and recognizing when you have not seen enough to know. The societies are invented. A program defines the rules and checks the answers, so there is no subjective AI judge.

This research checkpoint did not meet all six predeclared primary criteria. The release documents the result; it is not a validated improvement in street smarts.

On 306 reserved questions, original Qwen scores **41.5%** and Troubadour scores **45.4%**. Answerable accuracy changes from 33.3% to 62.7%; correctly recognizing insufficient evidence changes from 57.8% to 10.8%. These outcomes must be read together. The complete criteria, paired uncertainty intervals, transfer tests, and stronger simple baselines are in [RESULTS.md](RESULTS.md).

For context, a nearest-example heuristic scores 56.9%, and simply predicting from the observation-label majority while ignoring the query scores 45.8%. These controls help distinguish useful rule learning from changes in answer frequency.

Start with [a visit to an unfamiliar settlement](EXAMPLE.md). It shows why the same observations can justify one answer while leaving another unresolved.

## What was built?

**Troubadour** is a reproducible synthetic dataset with a finite grammar of 612 conventions over five binary encounter features. Whole rule identities and their complements are separated across training, development, and final evaluation. The final questions were generated only after model selection was frozen. Training views overlap by design and are not independent datasets; their exact contents and encoding are in [DATASET_CARD.md](DATASET_CARD.md).

**Troubadour-4B** is an MLX LoRA adapter over a frozen four-bit Qwen3-4B-Instruct-2507 checkpoint. Post-training changes a small set of adapter parameters. At test time the weights stay fixed: the model must learn about the current society from its visible examples. It receives observations and a query, never the hidden rule, target answer, or solver's intermediate results.

**Troubadour-Bench** contains the scoring code here and three separate test splits in the same Hugging Face dataset: `test`, `test_transfer`, and `test_contrast`. They test unfamiliar rule identities, reserved combinations of relevant features, and opposite questions about identical evidence. There is no separate evaluation repository to download.

The answer is A (respect), B (breach), or U (insufficient evidence). U is correct when permitted rules fit all observations but disagree about the query. Evaluation scores the model's next-token A/B/U probabilities, using one unpadded full-output projection per input. No extra reasoning tokens or tool-assisted inference are used. These are normalized choice probabilities, not calibrated confidence estimates.

An exact symbolic solver can already solve this bounded rulebook. Its role is to establish an objective reference. The experiment asks whether post-training improves the language model's ability to use new examples; it does not claim that a language model is the best implementation of this particular solver.

## Run locally on Apple silicon

Use Python 3.12. The measured machine was an Apple M2 with 16 GiB unified memory. The pinned runtime requires Python 3.12 or newer.

```bash
git clone https://github.com/1999labs/troubadour-4b.git
cd troubadour-4b
python3.12 -m venv .venv
source .venv/bin/activate
pip install -c requirements.lock -e '.[mac,dev]'
hf download noahmclaughlin/Troubadour-4B --local-dir artifacts/published-model
python -m troubadour.predict --adapter artifacts/published-model/adapter \
  --episode artifacts/published-model/example_episode.json
```

The original base weights are downloaded automatically at the pinned revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. Omit `--adapter` to ask original Qwen the same question. This release is an adapter with a complete inference path, not a merged Transformers checkpoint or GGUF file. See [MODEL_CARD.md](MODEL_CARD.md) for details.

The optional local demo compares the two models as you reveal more observations:

```bash
python app.py --adapter artifacts/published-model/adapter
```

Random demo worlds may overlap training rules. The demo is not the held-out benchmark, and its checker supplies feedback only after the model has answered.

## Reproduce the primary comparison

Download the public test split, then score both models with identical inputs and decoding:

```python
from pathlib import Path
from datasets import load_dataset
Path("data").mkdir(exist_ok=True)
load_dataset("noahmclaughlin/Troubadour", split="test").to_json("data/public-test.jsonl")
```

```bash
python -m troubadour.evaluate_batch --data data/public-test.jsonl \
  --out runs/qwen-public-test.jsonl --batch-size 2 --activation-cache
python -m troubadour.evaluate_batch --data data/public-test.jsonl \
  --out runs/troubadour-public-test.jsonl --batch-size 2 --activation-cache \
  --adapter artifacts/published-model/adapter
python -m troubadour.analyze --base runs/qwen-public-test.jsonl \
  --candidate runs/troubadour-public-test.jsonl --out artifacts/public-comparison.json
```

`--batch-size` controls output chunks; protocol v4 scores each unpadded input separately. Optional prefix caching reuses only unchanged lower-layer computation. It changes compute cost, not the model's information. `results/predictions/` contains the released raw predictions and their provenance. The [public verification record](results/public_release_verification.json) pins the exact Hub revisions; use those revisions for archival reproduction instead of an evolving `main` branch.

## Training and audit trail

The released seed is 17, using `choice-slow-seed17` at microstep 1200. Exact configurations, checksums and the pre-test selection reason are in [results/selection.json](results/selection.json). A full training trial may extend beyond the selected early checkpoint. See [COMPUTE.md](COMPUTE.md) for measured trial durations and retained update/token counts.

This is supervised classification over the original model's answer-token logits, not reinforcement learning or an external classifier head. The original cached base files are never edited. Failed and weaker trials, the numerical scoring correction, and all prospective amendments remain documented in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) and [RESEARCH_PLAN.md](RESEARCH_PLAN.md). The same success thresholds were retained throughout.

```bash
pytest -q
python -m troubadour.build --train-size 6000
python -m troubadour.build_choice
python -m troubadour.build_counterfactual
python -m troubadour.audit
```

Training commands and fixed recipes are in `configs/` and the experiment log. The dataset builders are deterministic. Reserved data generation requires a frozen selection. Once public, these test splits may be used by future trainers, so later claims should disclose contamination and reserve genuinely unseen societies.

This first release has one training seed and no completed budget-matched answer-key training control. Small rendering, evidence-curve, and general-instruction diagnostics are exploratory. A split by rule identity does not establish a general learning algorithm or a neural mechanism. Results concern this finite synthetic task, not real-world cultural competence or broad social intelligence.

## Related work and licensing

The project draws on [MetaICL](https://aclanthology.org/2022.naacl-main.201/), [Symbol Tuning](https://arxiv.org/abs/2305.08298), [Larger language models do in-context learning differently](https://arxiv.org/abs/2303.03846), and [Garg et al.'s synthetic function-learning study](https://arxiv.org/abs/2208.01066). It does not claim to invent learning-to-learn or synthetic rule induction.

Original code, synthetic data, and adapter weights are released under [Apache-2.0](LICENSE). The original Qwen weights retain their upstream license and attribution; see [NOTICE](NOTICE). The dataset contains no records of real participants.
