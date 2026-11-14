"""Deterministic length and position schedule for long-context Armor training."""

import random


MIX = [
    ("<=512", 0.40, 0, 512),
    ("513-1024", 0.25, 513, 1024),
    ("1025-2048", 0.15, 1025, 2048),
    ("2049-4096", 0.10, 2049, 4096),
    ("4097-6144", 0.07, 4097, 6144),
    ("6145-8192", 0.03, 6145, 8192),
]
POSITIONS = (100, 700, 1500, 3000, 5000, 7000)


def bucket_index(length: int) -> int:
    for i, (_name, _share, low, high) in enumerate(MIX):
        if low <= length <= high:
            return i
    raise ValueError(f"input length {length} is outside the training context")


def target_counts(count: int) -> list[int]:
    counts = [round(count * share) for _name, share, _low, _high in MIX]
    counts[0] += count - sum(counts)
    return counts


def assign_rows(rows, lengths: list[int], seed: int) -> list[list[int]]:
    """Preserve every >512-token row and fill the measured curriculum mix."""
    counts = target_counts(len(rows))
    groups = [[] for _ in MIX]
    short = []
    for i, length in enumerate(lengths):
        if length > 512:
            groups[bucket_index(length)].append(i)
        else:
            short.append(i)
    rng = random.Random(seed)
    rng.shuffle(short)
    cursor = 0
    for i in range(len(MIX)):
        needed = counts[i] - len(groups[i])
        if needed < 0:
            raise RuntimeError(f"source data already exceeds requested {MIX[i][0]} share")
        groups[i].extend(short[cursor:cursor + needed])
        cursor += needed
    if cursor != len(short) or [len(group) for group in groups] != counts:
        raise RuntimeError("could not assign every training row to the requested length mix")
    return groups
