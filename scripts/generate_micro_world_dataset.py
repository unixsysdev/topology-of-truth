from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LABELS = ["True", "False", "Unknown"]

ENTITY_POOLS = {
    "train": ["dax", "wug", "blicket", "zorb", "fep", "norp", "kiv", "lume", "pavo", "sarn", "tib", "vonn"],
    "dev": ["mav", "rusk", "drem", "siv", "glorp", "tavi", "zel", "pim"],
    "eval": ["bex", "nalo", "drin", "sop", "vex", "koba", "lirk", "mep", "toma", "grel"],
}

ATTRIBUTE_POOLS = {
    "train": ["glim", "daxel", "nubic", "vorn", "selt", "pald"],
    "dev": ["kess", "rindle", "mobic", "tess"],
    "eval": ["falm", "brindle", "soric", "peln", "zantic"],
}

RELATION_POOLS = {
    "train": ["mip", "naze", "tob", "lirp", "sume", "vark"],
    "dev": ["gorp", "zane", "plick", "rume"],
    "eval": ["drel", "fosh", "keld", "nimp", "sov"],
}


@dataclass(frozen=True)
class World:
    world_id: str
    split: str
    entities: list[str]
    attributes: list[str]
    relations: list[str]
    attr_pos: dict[str, set[str]]
    attr_neg: dict[str, set[str]]
    rel_pos: dict[str, set[tuple[str, str]]]
    rel_neg: dict[str, set[tuple[str, str]]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic micro-world truth-evaluation dataset.")
    parser.add_argument("--out-dir", default="artifacts/micro_world_v1/dataset")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--train-worlds", type=int, default=100)
    parser.add_argument("--dev-worlds", type=int, default=25)
    parser.add_argument("--eval-worlds", type=int, default=100)
    parser.add_argument("--min-entities", type=int, default=4)
    parser.add_argument("--max-entities", type=int, default=6)
    parser.add_argument("--attributes-per-world", type=int, default=2)
    parser.add_argument("--relations-per-world", type=int, default=2)
    parser.add_argument("--props-per-world", type=int, default=9)
    parser.add_argument("--paraphrases-per-prop", type=int, default=8)
    args = parser.parse_args()

    if args.props_per_world < 3:
        raise ValueError("--props-per-world must be at least 3 for three-label balancing")

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_examples: list[dict[str, Any]] = []
    split_specs = [("train", args.train_worlds), ("dev", args.dev_worlds), ("eval", args.eval_worlds)]
    for split, n_worlds in split_specs:
        examples = []
        for world_index in range(n_worlds):
            world_id = f"{split}_world_{world_index:06d}"
            world = sample_world(
                rng,
                split,
                world_id,
                min_entities=args.min_entities,
                max_entities=args.max_entities,
                attributes_per_world=args.attributes_per_world,
                relations_per_world=args.relations_per_world,
            )
            props = sample_balanced_propositions(rng, world, args.props_per_world)
            facts_text = render_facts(world)
            for prop_index, proposition in enumerate(props):
                proposition_id = f"{world_id}_prop_{prop_index:04d}"
                label = evaluate(proposition, world)
                for paraphrase_index in range(args.paraphrases_per_prop):
                    template_variant = template_variant_for(split, paraphrase_index)
                    statement, template_id, template_family = render_statement(proposition, template_variant)
                    examples.append(
                        {
                            "example_id": f"{proposition_id}_para_{paraphrase_index:02d}",
                            "world_id": world.world_id,
                            "split": split,
                            "world": world_to_payload(world),
                            "facts_text": facts_text,
                            "logical_form": proposition,
                            "proposition_id": proposition_id,
                            "paraphrase_id": f"para_{paraphrase_index:02d}",
                            "template_id": template_id,
                            "template_family": template_family,
                            "statement": statement,
                            "label": label,
                            "complexity": complexity(proposition),
                            "nonce_lexicon": {
                                "entities": world.entities,
                                "attributes": world.attributes,
                                "relations": world.relations,
                            },
                            "prompt": format_prompt(facts_text, statement),
                        }
                    )
        write_jsonl(out_dir / f"{split}.jsonl", examples)
        all_examples.extend(examples)

    write_audits(out_dir, all_examples)
    write_manifest(out_dir, args, all_examples)
    print(f"Wrote {len(all_examples)} examples to {out_dir}")


def sample_world(
    rng: random.Random,
    split: str,
    world_id: str,
    min_entities: int,
    max_entities: int,
    attributes_per_world: int,
    relations_per_world: int,
) -> World:
    entities = rng.sample(ENTITY_POOLS[split], rng.randint(min_entities, max_entities))
    attributes = rng.sample(ATTRIBUTE_POOLS[split], attributes_per_world)
    relations = rng.sample(RELATION_POOLS[split], relations_per_world)

    attr_pos = {attr: set() for attr in attributes}
    attr_neg = {attr: set() for attr in attributes}
    for attr in attributes:
        for entity in entities:
            status = rng.choices(["pos", "neg", "unknown"], weights=[0.4, 0.3, 0.3], k=1)[0]
            if status == "pos":
                attr_pos[attr].add(entity)
            elif status == "neg":
                attr_neg[attr].add(entity)

    rel_pos = {rel: set() for rel in relations}
    rel_neg = {rel: set() for rel in relations}
    for rel in relations:
        for subject in entities:
            for obj in entities:
                if subject == obj:
                    continue
                status = rng.choices(["pos", "neg", "unknown"], weights=[0.25, 0.25, 0.5], k=1)[0]
                if status == "pos":
                    rel_pos[rel].add((subject, obj))
                elif status == "neg":
                    rel_neg[rel].add((subject, obj))

    return World(
        world_id=world_id,
        split=split,
        entities=entities,
        attributes=attributes,
        relations=relations,
        attr_pos=attr_pos,
        attr_neg=attr_neg,
        rel_pos=rel_pos,
        rel_neg=rel_neg,
    )


def sample_balanced_propositions(rng: random.Random, world: World, props_per_world: int) -> list[dict[str, Any]]:
    quotas = label_quotas(props_per_world)
    selected: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    seen = set()
    for _ in range(12000):
        lf = sample_logical_form(rng, world)
        key = canonical(lf)
        if key in seen:
            continue
        seen.add(key)
        label = evaluate(lf, world)
        if len(selected[label]) < quotas[label]:
            selected[label].append(lf)
        if all(len(selected[label]) >= quotas[label] for label in LABELS):
            break

    missing = {label: quotas[label] - len(selected[label]) for label in LABELS if len(selected[label]) < quotas[label]}
    if missing:
        raise RuntimeError(f"Could not balance {world.world_id}; missing={missing}")

    props = []
    for label in LABELS:
        props.extend(selected[label])
    rng.shuffle(props)
    return props


def label_quotas(props_per_world: int) -> dict[str, int]:
    base = props_per_world // 3
    remainder = props_per_world % 3
    quotas = {label: base for label in LABELS}
    for label in LABELS[:remainder]:
        quotas[label] += 1
    return quotas


def sample_logical_form(rng: random.Random, world: World) -> dict[str, Any]:
    family = rng.choice(["attr", "rel", "not", "and", "or", "some", "all", "none", "exactly_one", "at_least_two"])
    if family == "attr":
        return sample_attr(rng, world)
    if family == "rel":
        return sample_rel(rng, world)
    if family == "not":
        return {"type": "not", "child": rng.choice([sample_attr(rng, world), sample_rel(rng, world)])}
    if family in {"and", "or"}:
        return {
            "type": family,
            "left": rng.choice([sample_attr(rng, world), sample_rel(rng, world)]),
            "right": rng.choice([sample_attr(rng, world), sample_rel(rng, world)]),
        }
    if family in {"some", "all", "none", "exactly_one", "at_least_two"}:
        return {
            "type": family,
            "attribute": rng.choice(world.attributes),
            "relation": rng.choice(world.relations),
            "object": rng.choice(world.entities),
        }
    raise ValueError(family)


def sample_attr(rng: random.Random, world: World) -> dict[str, Any]:
    return {"type": "attr", "entity": rng.choice(world.entities), "attribute": rng.choice(world.attributes)}


def sample_rel(rng: random.Random, world: World) -> dict[str, Any]:
    subject, obj = rng.sample(world.entities, 2)
    return {"type": "rel", "subject": subject, "relation": rng.choice(world.relations), "object": obj}


def evaluate(lf: dict[str, Any], world: World) -> str:
    kind = lf["type"]
    if kind == "attr":
        attr = lf["attribute"]
        entity = lf["entity"]
        if entity in world.attr_pos[attr]:
            return "True"
        if entity in world.attr_neg[attr]:
            return "False"
        return "Unknown"
    if kind == "rel":
        rel = lf["relation"]
        pair = (lf["subject"], lf["object"])
        if pair in world.rel_pos[rel]:
            return "True"
        if pair in world.rel_neg[rel]:
            return "False"
        return "Unknown"
    if kind == "not":
        return negate(evaluate(lf["child"], world))
    if kind == "and":
        return truth_and(evaluate(lf["left"], world), evaluate(lf["right"], world))
    if kind == "or":
        return truth_or(evaluate(lf["left"], world), evaluate(lf["right"], world))
    if kind in {"some", "all", "none", "exactly_one", "at_least_two"}:
        return evaluate_quantifier(lf, world)
    raise ValueError(kind)


def evaluate_quantifier(lf: dict[str, Any], world: World) -> str:
    vals = []
    for entity in world.entities:
        if entity == lf["object"]:
            continue
        attr = evaluate({"type": "attr", "entity": entity, "attribute": lf["attribute"]}, world)
        rel = evaluate({"type": "rel", "subject": entity, "relation": lf["relation"], "object": lf["object"]}, world)
        vals.append(truth_and(attr, rel))

    kind = lf["type"]
    if kind == "some":
        return existential(vals)
    if kind == "none":
        return negate(existential(vals))
    if kind == "all":
        implications = []
        for entity in world.entities:
            if entity == lf["object"]:
                continue
            attr = evaluate({"type": "attr", "entity": entity, "attribute": lf["attribute"]}, world)
            rel = evaluate({"type": "rel", "subject": entity, "relation": lf["relation"], "object": lf["object"]}, world)
            implications.append(truth_or(negate(attr), rel))
        return universal(implications)
    if kind in {"exactly_one", "at_least_two"}:
        true_count = sum(v == "True" for v in vals)
        possible_count = sum(v != "False" for v in vals)
        if kind == "exactly_one":
            if true_count > 1 or possible_count < 1:
                return "False"
            if true_count == 1 and possible_count == 1:
                return "True"
            return "Unknown"
        if true_count >= 2:
            return "True"
        if possible_count < 2:
            return "False"
        return "Unknown"
    raise ValueError(kind)


def negate(label: str) -> str:
    if label == "True":
        return "False"
    if label == "False":
        return "True"
    return "Unknown"


def truth_and(left: str, right: str) -> str:
    if "False" in {left, right}:
        return "False"
    if "Unknown" in {left, right}:
        return "Unknown"
    return "True"


def truth_or(left: str, right: str) -> str:
    if "True" in {left, right}:
        return "True"
    if "Unknown" in {left, right}:
        return "Unknown"
    return "False"


def existential(values: list[str]) -> str:
    if any(value == "True" for value in values):
        return "True"
    if all(value == "False" for value in values):
        return "False"
    return "Unknown"


def universal(values: list[str]) -> str:
    if any(value == "False" for value in values):
        return "False"
    if all(value == "True" for value in values):
        return "True"
    return "Unknown"


def render_facts(world: World) -> list[str]:
    facts = []
    for attr in world.attributes:
        for entity in sorted(world.attr_pos[attr]):
            facts.append(f"{entity} is {attr}.")
        for entity in sorted(world.attr_neg[attr]):
            facts.append(f"{entity} is not {attr}.")
    for rel in world.relations:
        for subject, obj in sorted(world.rel_pos[rel]):
            facts.append(f"{subject} {verb3(rel)} {obj}.")
        for subject, obj in sorted(world.rel_neg[rel]):
            facts.append(f"{subject} does not {rel} {obj}.")
    return facts


def render_statement(lf: dict[str, Any], variant: int) -> tuple[str, str, str]:
    kind = lf["type"]
    if kind == "attr":
        entity = lf["entity"]
        attr = lf["attribute"]
        variants = [
            f"{entity} is {attr}.",
            f"The item named {entity} is {attr}.",
            f"It is true that {entity} is {attr}.",
            f"{entity} belongs to the {attr} things.",
            f"The object {entity} has the property {attr}.",
            f"Among the named items, {entity} is {attr}.",
            f"{attr} applies to {entity}.",
            f"The {attr} property holds for {entity}.",
        ]
    elif kind == "rel":
        subject = lf["subject"]
        rel = lf["relation"]
        obj = lf["object"]
        variants = [
            f"{subject} {verb3(rel)} {obj}.",
            f"The item named {subject} {verb3(rel)} the item named {obj}.",
            f"It is true that {subject} {verb3(rel)} {obj}.",
            f"{subject} stands in the {rel} relation to {obj}.",
            f"The relation {rel} holds from {subject} to {obj}.",
            f"{obj} is {rel}ed by {subject}.",
            f"For this world, {subject} {verb3(rel)} {obj}.",
            f"The ordered pair ({subject}, {obj}) has relation {rel}.",
        ]
    elif kind == "not":
        child, child_id, _ = render_statement(lf["child"], variant)
        bare = strip_period(child)
        variants = [
            f"It is not the case that {bare}.",
            f"{bare} is not true.",
            f"Not: {bare}.",
            f"Deny the claim that {bare}.",
            f"The proposition applies logical negation to this clause: {bare}.",
            f"Logical form: NOT[{bare}].",
            f"The proposition is the negation of '{bare}'.",
            f"Apply logical NOT to this sentence: {bare}.",
        ]
        return variants[variant % len(variants)], f"not_v{variant % len(variants)}_{child_id}", "negation"
    elif kind in {"and", "or"}:
        left, left_id, _ = render_statement(lf["left"], variant)
        right, right_id, _ = render_statement(lf["right"], variant + 1)
        op = "and" if kind == "and" else "or"
        left_bare = strip_period(left)
        right_bare = strip_period(right)
        variants = [
            f"{left_bare} {op} {right_bare}.",
            f"Both claims hold: {left_bare}; {right_bare}." if kind == "and" else f"At least one claim holds: {left_bare}; {right_bare}.",
            f"The claim combines these with {op}: {left_bare}; {right_bare}.",
            f"It is true that {left_bare} {op} that {right_bare}.",
            f"{left_bare}. Also, {right_bare}." if kind == "and" else f"Either {left_bare}, or {right_bare}.",
            f"The first clause is '{left_bare}' and the connector is {op} with '{right_bare}'.",
            f"Evaluate this compound claim: {left_bare} {op} {right_bare}.",
            f"The compound proposition says {left_bare} {op} {right_bare}.",
        ]
        return variants[variant % len(variants)], f"{kind}_v{variant % len(variants)}_{left_id}_{right_id}", kind
    elif kind in {"some", "all", "none", "exactly_one", "at_least_two"}:
        attr = lf["attribute"]
        rel = lf["relation"]
        obj = lf["object"]
        quantifier = {
            "some": "Some",
            "all": "Every",
            "none": "No",
            "exactly_one": "Exactly one",
            "at_least_two": "At least two",
        }[kind]
        variants = [
            f"{quantifier} {attr} thing {verb3(rel)} {obj}.",
            f"{quantifier} object with property {attr} {verb3(rel)} {obj}.",
            f"{quantifier} item that is {attr} stands in relation {rel} to {obj}.",
            f"{quantifier} {attr} entity has the {rel} relation to {obj}.",
            f"In this world, {quantifier.lower()} {attr} thing {verb3(rel)} {obj}.",
            f"The number of {attr} things that {rel} {obj} matches: {quantifier.lower()}.",
            f"Count the {attr} things that {rel} {obj}; the result is {quantifier.lower()}.",
            f"Among {attr} things, {quantifier.lower()} of them {rel} {obj}.",
        ]
    else:
        raise ValueError(kind)
    index = variant % len(variants)
    return variants[index], f"{kind}_v{index}", kind


def template_variant_for(split: str, paraphrase_index: int) -> int:
    if split == "eval":
        return 4 + (paraphrase_index % 4)
    if split == "dev":
        return 2 + (paraphrase_index % 4)
    return paraphrase_index % 4


def strip_period(text: str) -> str:
    return text[:-1] if text.endswith(".") else text


def verb3(base: str) -> str:
    if base.endswith(("s", "x", "z", "ch", "sh")):
        return base + "es"
    if base.endswith("e"):
        return base + "s"
    return base + "s"


def format_prompt(facts_text: list[str], statement: str) -> str:
    facts = "\n".join(f"- {fact}" for fact in facts_text)
    return (
        "World:\n"
        f"{facts}\n\n"
        "Statement:\n"
        f"{statement}\n\n"
        "Answer with exactly one token: True, False, or Unknown."
    )


def complexity(lf: dict[str, Any]) -> dict[str, int]:
    kind = lf["type"]
    if kind in {"attr", "rel"}:
        return {"num_atoms": 1, "num_quantifiers": 0, "num_negations": 0, "reasoning_hops": 1}
    if kind == "not":
        child = complexity(lf["child"])
        return {
            "num_atoms": child["num_atoms"],
            "num_quantifiers": child["num_quantifiers"],
            "num_negations": child["num_negations"] + 1,
            "reasoning_hops": child["reasoning_hops"],
        }
    if kind in {"and", "or"}:
        left = complexity(lf["left"])
        right = complexity(lf["right"])
        return {
            "num_atoms": left["num_atoms"] + right["num_atoms"],
            "num_quantifiers": left["num_quantifiers"] + right["num_quantifiers"],
            "num_negations": left["num_negations"] + right["num_negations"],
            "reasoning_hops": max(left["reasoning_hops"], right["reasoning_hops"]),
        }
    if kind in {"some", "all", "none", "exactly_one", "at_least_two"}:
        return {"num_atoms": 2, "num_quantifiers": 1, "num_negations": int(kind == "none"), "reasoning_hops": 2}
    raise ValueError(kind)


def world_to_payload(world: World) -> dict[str, Any]:
    return {
        "world_id": world.world_id,
        "split": world.split,
        "entities": world.entities,
        "attributes": world.attributes,
        "relations": world.relations,
        "attr_pos": {attr: sorted(values) for attr, values in world.attr_pos.items()},
        "attr_neg": {attr: sorted(values) for attr, values in world.attr_neg.items()},
        "rel_pos": {rel: sorted([list(pair) for pair in values]) for rel, values in world.rel_pos.items()},
        "rel_neg": {rel: sorted([list(pair) for pair in values]) for rel, values in world.rel_neg.items()},
        "semantics": "partial-world three-valued: explicit positive facts entail True; explicit negative facts entail False; omitted facts are Unknown",
    }


def canonical(lf: dict[str, Any]) -> str:
    return json.dumps(lf, sort_keys=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_audits(out_dir: Path, examples: list[dict[str, Any]]) -> None:
    audit_rows = []
    dimensions = {
        "split": lambda row: row["split"],
        "label": lambda row: row["label"],
        "template_family": lambda row: row["template_family"],
        "template_id": lambda row: row["template_id"],
        "lf_type": lambda row: row["logical_form"]["type"],
        "num_atoms": lambda row: str(row["complexity"]["num_atoms"]),
        "num_quantifiers": lambda row: str(row["complexity"]["num_quantifiers"]),
        "num_negations": lambda row: str(row["complexity"]["num_negations"]),
        "reasoning_hops": lambda row: str(row["complexity"]["reasoning_hops"]),
    }
    for dimension, getter in dimensions.items():
        counts = Counter((row["split"], row["label"], getter(row)) for row in examples)
        for (split, label, value), count in sorted(counts.items()):
            audit_rows.append({"dimension": dimension, "split": split, "label": label, "value": value, "count": count})
    write_csv(out_dir / "audit.csv", audit_rows, ["dimension", "split", "label", "value", "count"])

    world_rows = []
    by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        by_world[row["world_id"]].append(row)
    for world_id, rows in sorted(by_world.items()):
        labels = Counter(row["label"] for row in rows)
        props = {row["proposition_id"] for row in rows}
        world_rows.append(
            {
                "world_id": world_id,
                "split": rows[0]["split"],
                "n_examples": len(rows),
                "n_propositions": len(props),
                "n_true": labels["True"],
                "n_false": labels["False"],
                "n_unknown": labels["Unknown"],
            }
        )
    write_csv(out_dir / "world_summary.csv", world_rows, ["world_id", "split", "n_examples", "n_propositions", "n_true", "n_false", "n_unknown"])


def write_manifest(out_dir: Path, args: argparse.Namespace, examples: list[dict[str, Any]]) -> None:
    counts = Counter((row["split"], row["label"]) for row in examples)
    payload = {
        "seed": args.seed,
        "train_worlds": args.train_worlds,
        "dev_worlds": args.dev_worlds,
        "eval_worlds": args.eval_worlds,
        "props_per_world": args.props_per_world,
        "paraphrases_per_prop": args.paraphrases_per_prop,
        "label_counts": {f"{split}:{label}": count for (split, label), count in sorted(counts.items())},
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
