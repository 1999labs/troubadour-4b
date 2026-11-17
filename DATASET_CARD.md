---
language:
- en
license: apache-2.0
pretty_name: Troubadour
task_categories:
- text-classification
- text-generation
tags:
- synthetic
- in-context-learning
- social-reasoning
- uncertainty
---
# Troubadour

Can post-training teach a model street smarts so that it can learn and apply unfamiliar social rules?

Troubadour is an original programmatically generated dataset of fictional conventions. Each episode contains observations, a query about a new encounter, and an objectively computed label: A (respect), B (breach), or U (insufficient evidence).

## What makes a label correct?

The public grammar contains 612 unique Boolean rules over five binary features. The reference solver retains every allowed rule consistent with the observations. It returns A/B only when all retained rules agree on the query. Otherwise it returns U, even when the hidden world has a definite answer. U means underdetermined within this documented finite grammar, not uncertainty over arbitrary imaginable customs.

The five features are recipient rank (lower/higher), setting (private/public), accepting/declining a token, presence/absence of a prior favor, and local/visitor status. These deliberately invented conventions do not describe real cultures or establish human-like theory of mind.

## Splits and leakage prevention

Whole truth tables and their complements are allocated together to disjoint train/development/test splits. Different formula expressions computing the same truth table are deduplicated. Structural-transfer splits reserve two combinations of three relevant features from training. Renaming a town is not counted as a new rule. Development and test contain paired worlds with opposite labels but identical contexts. Queries never repeat an observed complete context.

Evaluation sets are deliberately class-balanced, not representative of a natural population. Every three episodes per canonical world include one underdetermined and two determined queries; complementary worlds balance A/B. Learning curves recompute the correct label for each shorter observation prefix.

## Fields and safe use

`messages` contains the system instruction, public observations/query, and target assistant label. `answer`, `observations`, `query`, `society_pair`, `rule_mask`, `family`, `consistent_rules`, and `k` support auditing. `contrast_group` identifies paired queries sharing observations in the stress split and is null in the other splits. All splits share one Arrow schema so they load together. Hidden `rule_mask` and metadata must NEVER be included in the tested model's prompt. The benchmark runner constructs inputs from observations/query only; the published answer is used solely for checking.

All prose is rendered from deterministic templates. No external model or human cultural judgments generated the labels. The final test is generated only after model selection is frozen; the public release makes it available for independent reproduction. Once public, a new model could train on it, so future claims should disclose contamination and use fresh held-out rules where possible.

## Limitations

This is a small, bounded symbolic task in social language. Success does not demonstrate real-world cultural competence, moral judgment, or general social intelligence. Most training examples use compact tables; prose and symbol renderings are separate robustness tests. The complete symbolic solver is stronger than a language model on this known rule system by construction.

License: Apache-2.0. Original synthetic data and generator; no real participant records.

## Training recipes and benchmark location

The complete generated training split and the balanced `train_choice` subset are overlapping views, not independent data. `train_choice` contains 2,850 eight-observation episodes with 950 examples per A/B/U label. The selected model card identifies which recipe was actually used.

Troubadour-Bench source and scoring documentation: https://github.com/1999labs/troubadour-4b. The model release: https://huggingface.co/noahmclaughlin/Troubadour-4B. Evaluation data is kept in separate test/transfer splits within this dataset; no separate evaluation dataset is required.

## Counterfactual training view

The optional `train_counterfactual` recipe contains 3,690 episodes in 615 consecutive groups of six. Within each group there are two A, two B and two U targets. In 492 groups, the same observations support three different query labels, paired with a complementary world. The other 123 groups retain ordinary balanced examples. Exactly 738 prompts overlap `train_choice` and the original training split; these are training views, not independent datasets. The grouped sampler preserves each complete six-example group. The model card identifies whether this recipe was actually used for the released checkpoint.

The counterfactual training view reuses `contrast_group` for its three-query sets; ordinary rows have null values. The evaluation stress splits use this field for two opposing queries. All hidden rule metadata remains excluded from model inputs.
