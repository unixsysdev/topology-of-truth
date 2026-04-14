# Micro-World Truth Geometry Experiment Spec

## Purpose

This experiment is the clean version of the original truth-geometry idea.

The question is not whether truth has a universal scalar signature. The question is:

> In a procedurally generated linguistic micro-world with exact semantics, do true, false, and unknown statements induce distinct local representation structure near the verdict state?

The design must prevent benchmark memorization and surface-template shortcuts as much as possible.

## Core Design

Each example is generated from a latent structured world.

The pipeline is:

1. Sample a world state.
2. Generate semantic propositions against that world.
3. Evaluate each proposition exactly as `True`, `False`, or `Unknown`.
4. Render each proposition with multiple paraphrase templates.
5. Prompt a model to answer with only `True`, `False`, or `Unknown`.
6. Extract hidden states near the verdict region.
7. Analyze label-conditioned local geometry within each world.

The unit of analysis is the world, not the individual sentence.

## World State

Initial domain:

- 4-6 entities per world
- Nonce entity names, for example `dax`, `wug`, `blicket`, `zorb`
- 2 unary attributes per world
- 2 binary relations per world
- Optional one-step rules after the base generator is stable

Example structured state:

```json
{
  "world_id": "world_000001",
  "entities": ["dax", "wug", "blicket", "zorb"],
  "attributes": {
    "red": ["dax", "zorb"],
    "heavy": ["wug"]
  },
  "relations": {
    "left_of": [["dax", "wug"], ["blicket", "zorb"]],
    "touches": [["wug", "zorb"]]
  },
  "rules": [
    {"if_attr": "red", "then_attr": "heavy"}
  ]
}
```

The first implementation should make unknowns possible by using partial observability. Unknown means the proposition is not entailed and not contradicted by the given facts under the chosen closed/open world convention.

Use a documented convention:

- Closed-world attributes and relations for the first binary task.
- Add an open-world or partial-world mode for the three-label task.
- Do not mix conventions in the same split.

Recommended first serious version: partial-world three-label mode.

## Proposition Families

Start with these semantic forms:

- Unary: `attr(entity)`
- Binary: `relation(entity_a, entity_b)`
- Negation: `not proposition`
- Conjunction: `p and q`
- Disjunction: `p or q`
- Existential: `some attr thing has relation to entity`
- Universal: `all attr things have relation to entity`
- None: `no attr thing has relation to entity`
- Counting: `exactly one`, `at least two`

Keep first-pass depth at 1-2 hops. Add 3-hop reasoning only after the generator passes balance and leakage audits.

## Language Rendering

Each semantic proposition must have multiple paraphrases.

Example proposition:

```json
{
  "logical_form": {"type": "relation", "relation": "left_of", "subject": "dax", "object": "wug"},
  "label": "True"
}
```

Example paraphrases:

- `The dax is left of the wug.`
- `Dax sits to the left of wug.`
- `Relative to wug, dax is on the left.`
- `Wug has dax on its left side.`

Template families must be balanced across labels. A template must not imply a label distribution by construction.

## Prompt Format

Use short forced outputs first:

```text
World:
- dax is red.
- wug is heavy.
- dax is left of wug.
- wug touches zorb.

Statement:
The dax is left of the wug.

Answer with exactly one token: True, False, or Unknown.
```

Do not ask for chain-of-thought in the primary experiment.

Later optional condition:

```text
Answer with exactly one token, then one short reason.
```

Keep this secondary. The primary verdict-state analysis should not depend on verbose generation.

## Splits

Use several split axes:

- Train/dev/eval split by world ID.
- Held-out entity nonce vocabulary for eval.
- Held-out paraphrase template families for eval.
- Held-out relation/attribute lexicalizations for eval.
- Optional held-out world sizes, for example train on 4-5 entities and evaluate on 6.

The strongest evaluation is generated fresh from held-out templates and held-out worlds.

## Anti-Leakage Controls

Required controls:

- Balance labels within each world where possible.
- Balance proposition complexity across labels.
- Balance template family use across labels.
- Use the same surface templates with different labels across different worlds.
- Include true negations and false negations.
- Include false positives and true negatives.
- Include `False` and `Unknown` examples with similar surface wording.
- Track lexical overlap and template ID in every row.

The generator must emit an audit table with:

- Label counts by split
- Label counts by template family
- Label counts by proposition family
- Complexity counts by label
- Entity/relation/attribute frequency by label

If these audits are not balanced, do not run model inference yet.

## Dataset Schema

Write examples as JSONL.

Required fields:

```json
{
  "example_id": "world_000001_prop_000003_para_02",
  "world_id": "world_000001",
  "split": "eval",
  "world": {},
  "facts_text": [],
  "logical_form": {},
  "proposition_id": "world_000001_prop_000003",
  "paraphrase_id": "para_02",
  "template_id": "relation_left_subject_first_v2",
  "template_family": "binary_relation",
  "statement": "The dax is left of the wug.",
  "label": "True",
  "complexity": {
    "num_atoms": 1,
    "num_quantifiers": 0,
    "num_negations": 0,
    "reasoning_hops": 1
  },
  "nonce_lexicon": {
    "entities": ["dax", "wug", "blicket", "zorb"],
    "attributes": ["red", "heavy"],
    "relations": ["left_of", "touches"]
  }
}
```

## Artifact Layout

Recommended layout:

```text
artifacts/micro_world_v1/
  dataset/
    train.jsonl
    dev.jsonl
    eval.jsonl
    audit.csv
    world_summary.csv
  generations/
    Qwen__Qwen3_5_2B/
      manifest.csv
      examples/
        world_000001_prop_000003_para_02/
          sample.json
          hidden.npy
          logits_summary.npz
  analysis/
    verdict_state_features.parquet
    within_world_distances.csv
    neighborhood_purity.csv
    topology_summary.csv
    lexical_baseline.csv
    probe_baseline.csv
```

## Representation Extraction

Primary extraction sites:

- Final prompt token before generation.
- Generated verdict token.
- Mean of the final 3-5 prompt tokens.
- Optional mean of generated verdict token plus immediate following token if generation adds punctuation.

Do not start with whole-trace H0.

The verdict-token analysis should be the first representation experiment because the output is short and controlled.

## Primary Metrics

Run these in order.

Level 1: local separability

- Within-world same-label distance vs different-label distance.
- Per-world centroid distances among `True`, `False`, and `Unknown`.
- Nearest-neighbor label purity within each world.
- Retrieval test: given an example, are nearest paraphrases same label more often than chance?

Level 2: controls

- Lexical baseline using bag-of-words or template IDs.
- Complexity-only baseline.
- Linear probe on hidden states.
- Template-held-out evaluation.

Level 3: topology

- H0/Betti curves by label-conditioned clouds per world.
- Persistence diagram distances between `True`, `False`, and `Unknown` clouds.
- Radius sweeps for connected-component merging across labels.
- Local intrinsic dimension per label cloud.

Topology is not the hero unless it improves on simpler geometry and controls.

## Success Criteria

A meaningful positive result requires:

- Within-world same-label neighborhoods are consistently tighter or purer than different-label neighborhoods.
- The effect holds on held-out worlds and held-out templates.
- The effect is not explained by template ID, lexical overlap, or proposition complexity.
- Stronger models show cleaner label-conditioned organization.
- Topology adds signal beyond centroid distance or a linear probe, if making a topology claim.

## Failure Criteria

The truth-geometry hypothesis should be weakened if:

- Label separation disappears on held-out templates.
- Lexical/template baselines match hidden-state geometry.
- `False` and `Unknown` collapse together.
- Same-label paraphrases are not closer than opposite-label paraphrases within worlds.
- Topology adds nothing beyond simple distances.

That would still be a useful negative result.

## First Scripts

Implement in this order:

1. `scripts/generate_micro_world_dataset.py`
   - Generate worlds, propositions, paraphrases, labels, and audits.

2. `scripts/run_micro_world_inference.py`
   - Run short-output inference and extract final prompt/verdict states.

3. `scripts/analyze_micro_world_geometry.py`
   - Compute within-world distance, purity, lexical baselines, and first probe baselines.

Only after those pass should TDA code be added.

## Initial Scale

Start small:

- 100 eval worlds
- 4-6 entities per world
- 6-12 propositions per world
- 8 paraphrases per proposition
- Balanced labels where possible

This gives thousands of statements while keeping world-level analysis viable.

Do not scale until audit balance is clean.
