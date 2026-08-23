"""
Unified, leakage-free split protocol shared across BOTH training stages.

The core guarantee: a single deterministic actor-level partition for CREMA-D.
The same partition is used by
  - Stage-1 (emotion extractor training/eval), and
  - Stage-2 (forgery discrimination training/eval),
so that any actor whose clips appear in a Stage-2 *test* pair was NEVER seen by
the frozen extractor during Stage-1 training. This removes the in-distribution
"familiarity" shortcut (genuine = seen, fake-audio = unseen) that would
otherwise let the discriminator separate classes without using emotional
consistency.

CREMA-D  -> speaker-independent (actor-disjoint) partition. More rigorous than
            the paper's random split; eliminates speaker leakage too.
MELD     -> no speaker labels exposed and no forgery stage downstream, so it
            keeps a stratified-by-emotion random split (matches the paper for
            emotion recognition). No cross-stage leakage risk for MELD.
"""
import random
from collections import defaultdict


def crema_actor(file_id: str) -> str:
    """Actor id = leading token. For a P2 splice 'A___B' the *source/visual*
    actor A is used (the genuine clip a forgery is built on), which is what must
    not leak across the split boundary."""
    return file_id.split("___")[0].split("_")[0]


def crema_actor_partition(actors, ratios=(0.8, 0.1, 0.1), seed=42):
    """Deterministically assign each CREMA actor to 'train' | 'val' | 'test'.
    Returns {actor_id: partition}. Same seed -> identical partition everywhere,
    so Stage-1 and Stage-2 agree on which actors are held out."""
    acts = sorted(set(actors))
    rng = random.Random(seed)
    rng.shuffle(acts)
    n = len(acts)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    part = {}
    for i, a in enumerate(acts):
        if i < n_tr:
            part[a] = "train"
        elif i < n_tr + n_va:
            part[a] = "val"
        else:
            part[a] = "test"
    return part


def partition_by_actor(samples, actor_fn, ratios=(0.8, 0.1, 0.1), seed=42):
    """Split samples into (train, val, test) by a shared actor partition.
    All samples of one actor land in exactly one partition."""
    part = crema_actor_partition([actor_fn(s) for s in samples], ratios, seed)
    buckets = defaultdict(list)
    for s in samples:
        buckets[part[actor_fn(s)]].append(s)
    rng = random.Random(seed)
    train, val, test = buckets["train"], buckets["val"], buckets["test"]
    rng.shuffle(train)
    return train, val, test
