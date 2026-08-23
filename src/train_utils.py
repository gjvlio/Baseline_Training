"""Shared training utilities: seeding, stratified split, early stopping."""
import random
from collections import defaultdict

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(samples, key_fn, ratios=(0.8, 0.1, 0.1), seed=42):
    """Split samples into train/val/test, stratified by key_fn(sample)."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for s in samples:
        buckets[key_fn(s)].append(s)
    train, val, test = [], [], []
    for _, items in buckets.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        train += items[:n_tr]
        val += items[n_tr:n_tr + n_va]
        test += items[n_tr + n_va:]
    rng.shuffle(train)
    return train, val, test


def group_aware_split(samples, group_fn, ratios=(0.8, 0.1, 0.1), seed=42):
    """Split so that all samples sharing a group_fn(sample) key land in the
    SAME partition. Prevents component/speaker leakage: a genuine clip and any
    forgery derived from it (or from the same actor) cannot straddle
    train/test. Groups are assigned to partitions by cumulative count to keep
    the sample-level ratios close to the targets.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for s in samples:
        groups[group_fn(s)].append(s)

    keys = list(groups.keys())
    rng.shuffle(keys)

    total = len(samples)
    n_tr = int(total * ratios[0])
    n_va = int(total * ratios[1])

    train, val, test = [], [], []
    count = 0
    for k in keys:
        items = groups[k]
        if count < n_tr:
            train += items
        elif count < n_tr + n_va:
            val += items
        else:
            test += items
        count += len(items)
    rng.shuffle(train)
    return train, val, test


class EarlyStopper:
    """Tracks best val loss; restores best state; signals stop after patience."""

    def __init__(self, patience):
        self.patience = patience
        self.best = float("inf")
        self.best_state = None
        self.counter = 0

    def step(self, val_loss, model):
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
