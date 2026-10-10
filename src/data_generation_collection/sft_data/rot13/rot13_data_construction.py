"""Exact three-region, nested partial ROT13 (C.1.3, H.3).

Frozen conventions: Unicode ``\\w+`` spans; ASCII-letter eligibility; early
regions receive remainders; SHA256-derived anchors and nearest-first expansion,
left before right on distance ties. No text is normalized before transformation.
"""
from __future__ import annotations

import codecs
import re
from collections.abc import Mapping

from shared_procedures.identity import content_hash, derive_seed

WORD_BOUNDARY_VERSION = "unicode_word_spans_v1"
TRANSFORM_VERSION = "rot13_three_regions_v1"
_WORD = re.compile(r"\w+", re.UNICODE)
_LATIN = re.compile(r"[A-Za-z]")


def eligible_word_spans(text: str) -> list[tuple[int, int]]:
    """Return original offsets; purely numeric/non-ASCII spans are ineligible."""
    return [(m.start(), m.end()) for m in _WORD.finditer(text) if _LATIN.search(m.group())]


def rot13_masks(reasoning: str, example_id: str, seed: int = 42) -> dict:
    if seed != 42:
        raise ValueError("Protocol ROT13 global seed must be 42")
    spans = eligible_word_spans(reasoning)
    n = len(spans)
    if n < 3:
        raise ValueError("At least three eligible original words required")
    sizes = [n // 3 + (j < n % 3) for j in range(3)]
    regions, anchors, orderings, seeds = [], [], [], []
    offset = 0
    for j, size in enumerate(sizes):
        region_seed = derive_seed(seed, "rot13", example_id, j)
        # Hash modular selection avoids dependence on Python RNG versions.
        anchor = region_seed % size
        order = sorted(range(size), key=lambda i: (abs(i - anchor), i))
        regions.append(list(range(offset, offset + size)))
        anchors.append(offset + anchor)
        orderings.append([offset + i for i in order])
        seeds.append(region_seed)
        offset += size
    masks = []
    for k in range(1, 11):
        # floor(k*size/10 + 1/2), entirely in integer arithmetic.
        mask = sorted(i for size, order in zip(sizes, orderings) for i in order[:(k * size + 5) // 10])
        if abs(10 * len(mask) - k * n) > 15:
            raise AssertionError("ROT13 half-up discrepancy invariant")
        if masks and not set(masks[-1]).issubset(mask):
            raise AssertionError("ROT13 masks must nest")
        masks.append(mask)
    if masks[-1] != list(range(n)):
        raise AssertionError("Level ten must cover all eligible words")
    return {"spans": spans, "eligible_word_count": n, "region_sizes": sizes,
            "region_seeds": seeds, "anchors": anchors, "orderings": orderings,
            "masks": masks, "repeated_masks": [k + 1 for k in range(1, 10) if masks[k] == masks[k - 1]]}


def transform_rot13(reasoning: str, example_id: str, level: int, *, seed: int = 42) -> dict:
    if isinstance(level, bool) or level not in range(1, 11):
        raise ValueError("ROT13 level must be 1..10")
    plan = rot13_masks(reasoning, example_id, seed)
    selected = set(plan["masks"][level - 1])
    pieces, cursor = [], 0
    for i, (start, end) in enumerate(plan["spans"]):
        pieces.append(reasoning[cursor:start])
        word = reasoning[start:end]
        pieces.append(codecs.encode(word, "rot_13") if i in selected else word)
        cursor = end
    pieces.append(reasoning[cursor:])
    transformed = "".join(pieces)
    # ROT13 must preserve all non-ASCII alphabetic bytes exactly.
    original_protected = "".join(c for c in reasoning if not ("a" <= c <= "z" or "A" <= c <= "Z"))
    transformed_protected = "".join(c for c in transformed if not ("a" <= c <= "z" or "A" <= c <= "Z"))
    if original_protected != transformed_protected:
        raise AssertionError("ROT13 altered protected characters")
    return {"reasoning": transformed, "method": "rot13", "level": level,
            "target_fraction": level / 10, "encoded_word_count": len(selected),
            "eligible_word_count": plan["eligible_word_count"],
            "changed_word_fraction": len(selected) / plan["eligible_word_count"],
            "selected_word_indices": sorted(selected), "region_sizes": plan["region_sizes"],
            "anchors": plan["anchors"], "region_seeds": plan["region_seeds"],
            "repeated_masks": plan["repeated_masks"], "seed": seed,
            "algorithm_version": TRANSFORM_VERSION, "word_boundary_version": WORD_BOUNDARY_VERSION,
            "source_hash": content_hash(reasoning), "content_hash": content_hash(transformed),
            "protected_content_valid": True, "status": "verified"}


def construct_rot13_data(pool: Mapping, tokenizers: Mapping, output_dir, *, config_hash: str) -> dict:
    """Construct this family's twenty verified pairs through the common builder."""
    from data_generation_collection.sft_data.sft_data_preparation import build_sft_datasets
    return build_sft_datasets(pool, tokenizers, output_dir, config_hash=config_hash, families=("rot13",))
