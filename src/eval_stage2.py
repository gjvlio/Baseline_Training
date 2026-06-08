"""
Stage-2 evaluation -> reproduces paper Table 4 (cross-modal detection by
forgery type). Re-derives the SAME seeded balanced set + 80/10/10 split used in
training, evaluates ONLY the held-out test portion with frozen ACE-Net, then
breaks results down by pairing type:

    Genuine Pairs            (label 0)
    Emotion Tampering        (label 1, P1)
    Cross-Identity Spliced   (label 1, P2)

For each subgroup reports ACC / Precision / Recall / F1 / AUC, matching the
paper's per-type table. AUC/precision/recall for a single forgery type are
computed against the genuine test set (one-vs-genuine), mirroring the paper's
pairing-type evaluation.

Example:
    python -m src.eval_stage2
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .config import TrainConfig
from .data import manifests
from .data.dataset import PairDataset, collate_pair
from .models.acenet import ACENet
from .train_utils import set_seed, group_aware_split
from . import logging_utils as L


def move(b, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def dataset_of(sample):
    root = str(sample.audio).lower()
    return "MELD" if "meld" in root else "CREMA-D"


def build_balanced_tagged(seed):
    """Same selection as train_stage2.build_balanced, but KEEP a pairing-type
    tag on each sample instead of clearing it."""
    rng = random.Random(seed)
    genuine, fake = manifests.build_stage2_samples()
    p1 = [s for s in fake if s.emotion == "crema_fake_p1"]
    p2 = [s for s in fake if s.emotion == "crema_fake_p2"]

    n_pos_each = min(len(p1), len(p2))
    rng.shuffle(p1); rng.shuffle(p2)
    fakes = p1[:n_pos_each] + p2[:n_pos_each]
    rng.shuffle(fakes)

    n = min(len(genuine), len(fakes))
    rng.shuffle(genuine)
    chosen = genuine[:n] + fakes[:n]
    for s in chosen:
        if s.label == 0:
            s.ptype = "Genuine Pairs"
        elif s.emotion == "crema_fake_p1":
            s.ptype = "Emotion Tampering"
        else:
            s.ptype = "Cross-Identity Spliced"
        s.emotion = None
    rng.shuffle(chosen)
    return chosen


@torch.no_grad()
def collect(model, samples, device, bs, nw):
    dl = DataLoader(PairDataset(samples), batch_size=bs, shuffle=False,
                    num_workers=nw, collate_fn=collate_pair)
    ys, ps = [], []
    meter = L.ProgressMeter(len(dl), 0, 0, "test", every=5)
    for bi, batch in enumerate(dl):
        batch = move(batch, device)
        logit = model(batch)
        ys.append(batch["label"].cpu()); ps.append(torch.sigmoid(logit).cpu())
        meter.update(bi)
    meter.close()
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def auc_score(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bin_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(len(y), 1)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return acc, prec, rec, f1


def main():
    ap = argparse.ArgumentParser(description="ACE-Net Stage-2 evaluation (Table 4)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    logger = L.Logger(config.CKPT_ROOT / "eval_stage2.log")
    L.run_header(logger, "ACE-Net Stage-2 EVAL  ::  forgery detection (Table 4)",
                 {"seed": cfg.seed, "split_ratio": "80/10/10 (test only)",
                  "threshold": 0.5})

    samples = build_balanced_tagged(cfg.seed)
    # IDENTICAL group-aware split to training (same seed) -> same held-out
    # test, with no actor/clip-component leakage across partitions.
    _, _, test_s = group_aware_split(
        samples, group_fn=lambda s: s.group_key,
        ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio), seed=cfg.seed)
    logger.log(f"held-out test samples: {L.bold(str(len(test_s)))}")

    ckpt = args.ckpt or (config.CKPT_ROOT / "stage2_acenet.pt")
    model = ACENet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    logger.log(f"loaded checkpoint <- {L.cyan(str(ckpt))}")

    # collect predictions over the whole test set once
    y, p = collect(model, test_s, device, args.batch_size, args.num_workers)
    types = np.array([s.ptype for s in test_s])
    dsets = np.array([dataset_of(s) for s in test_s])

    genuine_mask = y == 0

    logger.raw("")
    logger.raw(L.bold("  ===== Cross-modal Detection by Forgery Type (paper Table 4) ====="))
    header = f"  {'Dataset':<10} {'Pairing Type':<24} {'ACC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}"
    logger.raw(L.bold(header))
    logger.raw("  " + "-" * 70)

    for ds in sorted(set(dsets.tolist())):
        ds_mask = dsets == ds
        ds_gen = ds_mask & genuine_mask
        for ptype in ["Genuine Pairs", "Emotion Tampering", "Cross-Identity Spliced"]:
            sel = ds_mask & (types == ptype)
            if sel.sum() == 0:
                continue
            if ptype == "Genuine Pairs":
                yy, pp = y[sel], p[sel]
                # genuine-only: report accuracy (correctly classified as real)
                acc = float((pp < 0.5).mean())
                logger.raw(f"  {ds:<10} {ptype:<24} {acc*100:>5.1f}% "
                           f"{'-':>6} {'-':>6} {'-':>6} {'-':>6}")
            else:
                # forgery vs genuine (one-vs-genuine), as in the paper
                idx = sel | ds_gen
                yy, pp = y[idx], p[idx]
                acc, prec, rec, f1 = bin_metrics(yy, pp)
                auc = auc_score(yy, pp)
                logger.raw(f"  {ds:<10} {ptype:<24} {acc*100:>5.1f}% "
                           f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {auc:>6.3f}")

    # overall AUC across all test pairs
    overall_auc = auc_score(y, p)
    overall_acc, op, orr, of1 = bin_metrics(y, p)
    logger.raw("  " + "-" * 70)
    logger.raw(L.bold(L.green(
        f"  OVERALL    all pairs              {overall_acc*100:>5.1f}% "
        f"{op:>6.3f} {orr:>6.3f} {of1:>6.3f} {overall_auc:>6.3f}")))
    logger.close()


if __name__ == "__main__":
    main()
