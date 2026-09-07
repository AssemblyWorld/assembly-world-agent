"""Translate source atomic equivalence annotations without inventing missing labels."""

import json


def parse_source_equivalence(parts, annotations):
    parts = sorted(parts, key=lambda part: part.part_id)
    mapping = {}
    for i, part in enumerate(parts):
        key = str(part.metadata.get("annotation_part_id", part.part_id))
        if key in mapping:
            raise ValueError("Duplicate annotation part ID")
        mapping[key] = i
    relation = annotations.get("geometric_equivalence_relation") or {}
    if isinstance(relation, str):
        relation = json.loads(relation)
    if not isinstance(relation, dict):
        raise ValueError("Expected source equivalence relation object")
    parent = list(range(len(parts)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    ignored = []
    if not all(isinstance(key, str) for key in relation):
        raise ValueError("Malformed source equivalence key")
    for key, values in sorted(relation.items()):
        if any(atom not in mapping for atom in key.split(",")):
            raise ValueError("Unknown source equivalence part reference")
        if values is None:
            continue
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError("Malformed source equivalence group")
        members = [key, *values]
        if any(atom not in mapping for value in members for atom in value.split(",")):
            raise ValueError("Unknown source equivalence part reference")
        if any("," in value for value in members):
            if set(members) != {key}:
                raise ValueError("Nontrivial composite equivalence is unsupported")
            ignored.append(key)
            continue
        for member in values:
            a, b = root(mapping[key]), root(mapping[member])
            parent[max(a, b)] = min(a, b)
    groups = {}
    for i in range(len(parts)):
        groups.setdefault(root(i), []).append(i)
    return {
        "groups": [[parts[i].part_id for i in group] for group in groups.values()],
        "missing": not bool(relation),
        "ignored_composite_self_groups": ignored,
    }
