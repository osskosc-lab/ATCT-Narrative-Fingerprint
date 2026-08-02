"""Content-preserving order interventions."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


CONTROL_NAMES = (
    "local",
    "paragraph_inner",
    "block",
    "paragraph_order",
    "section_order",
    "random",
    "reverse",
)
STRUCTURED_CONTROLS = {"paragraph_inner", "paragraph_order", "section_order"}


def _nonidentity_random(n_items: int, rng: np.random.Generator) -> np.ndarray:
    identity = np.arange(n_items)
    for _ in range(32):
        order = rng.permutation(n_items)
        if not np.array_equal(order, identity):
            return order
    return np.roll(identity, 1)


def local_swap(n_items: int, rng: np.random.Generator) -> np.ndarray:
    """Swap exactly one adjacent sentence pair."""

    if n_items < 2:
        raise ValueError("local swap requires at least two items")
    order = np.arange(n_items)
    left = int(rng.integers(0, n_items - 1))
    order[left], order[left + 1] = order[left + 1], order[left]
    return order


def block_swap(
    n_items: int,
    rng: np.random.Generator,
    *,
    block_min: int = 3,
    block_max: int = 5,
) -> np.ndarray:
    """Reorder 3-5 sentence blocks while preserving order inside each block."""

    if n_items < 2:
        raise ValueError("block swap requires at least two items")
    block_size = min(int(rng.integers(block_min, block_max + 1)), n_items)
    blocks = [
        np.arange(start, min(start + block_size, n_items))
        for start in range(0, n_items, block_size)
    ]
    if len(blocks) < 2:
        return local_swap(n_items, rng)
    block_order = _nonidentity_random(len(blocks), rng)
    return np.concatenate([blocks[index] for index in block_order])


def paragraph_swap(
    paragraph_indices: Sequence[Sequence[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Reorder paragraphs and preserve every within-paragraph sentence order."""

    blocks = [np.asarray(block, dtype=int) for block in paragraph_indices if block]
    if len(blocks) < 2:
        raise ValueError("paragraph swap requires at least two paragraphs")
    order = _nonidentity_random(len(blocks), rng)
    return np.concatenate([blocks[index] for index in order])


def within_block_shuffle(
    block_indices: Sequence[Sequence[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle sentences inside blocks while retaining the block sequence."""

    blocks = [np.asarray(block, dtype=int) for block in block_indices if block]
    if not blocks:
        raise ValueError("within-block shuffle requires at least one block")
    candidates = [index for index, block in enumerate(blocks) if len(block) >= 2]
    if not candidates:
        raise ValueError("within-block shuffle requires a block with two items")
    identity = np.concatenate(blocks)
    for _ in range(32):
        shuffled = [
            block[rng.permutation(len(block))] if len(block) >= 2 else block
            for block in blocks
        ]
        order = np.concatenate(shuffled)
        if not np.array_equal(order, identity):
            return order
    fallback = [block.copy() for block in blocks]
    target = candidates[0]
    fallback[target][0], fallback[target][1] = (
        fallback[target][1],
        fallback[target][0],
    )
    return np.concatenate(fallback)


def generate_permutations(
    control: str,
    n_items: int,
    count: int,
    rng: np.random.Generator,
    *,
    blocks: Sequence[Sequence[int]] | None = None,
) -> list[np.ndarray]:
    """Generate deterministic interventions from a caller-owned RNG."""

    if control not in CONTROL_NAMES:
        raise ValueError(f"unknown control: {control}")
    if count < 1:
        raise ValueError("count must be positive")
    if control == "reverse":
        return [np.arange(n_items)[::-1]]
    if control in STRUCTURED_CONTROLS:
        if not blocks:
            raise ValueError(f"{control} requires structured blocks")
        generator = (
            within_block_shuffle
            if control == "paragraph_inner"
            else paragraph_swap
        )
        return [generator(blocks, rng) for _ in range(count)]
    generator = {
        "local": local_swap,
        "block": block_swap,
        "random": _nonidentity_random,
    }[control]
    return [generator(n_items, rng) for _ in range(count)]


def apply_permutation(items: Sequence[str], order: Iterable[int]) -> list[str]:
    return [items[int(index)] for index in order]
