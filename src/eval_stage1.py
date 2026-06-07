"""
Stage-1 evaluation -> reproduces paper Table 2 (speech-text emotion ACC) and
Table 3 (facial emotion ACC), plus confusion matrices (Figures 5/6).

Re-derives the SAME seeded 80/10/10 stratified split used in training, then
evaluates ONLY the held-out test portion with the frozen checkpoint. Reports
overall accuracy, weighted-F1, and a per-class confusion matrix, broken down
per source dataset (CREMA-D / MELD).

Examples:
    python -m src.eval_stage1 --branch visual
    python -m src.eval_stage1 --branch speech_text --splits crema_genuine_lasthalf meld
"""
import argparse
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .config import TrainConfig
from .data import manifests
from .data.dataset import EmotionDataset, collate_emotion
from .models.speech_text import SpeechTextModule
from .models.fv_litenet import FVLiteNet
from .train_utils import set_seed, stratified_split
from . import logging_utils as L


def move(b, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def dataset_of(sample):
    """Tag a sample with its source dataset for per-dataset reporting."""
    root = str(sample.audio).lower()
    if "meld" in root:
        return "MELD"
    return "CREMA-D"


@torch.no_grad()
def _collect_preds(model, loader, branch, device):
    model.eval()
    ys, preds = [], []
    meter = L.ProgressMeter(len(loader), 0, 0, "test", every=5)
    for bi, batch in enumerate(loader):
        batch = move(batch, device)
        if branch == "speech_text":
            logits, _ = model(batch["melspec"], batch["mel_lengths"],
                              batch["input_ids"], batch["attention_mask"])
        else:
            logits, _ = model(batch["frames"], batch["alpha"], batch["frame_mask"])
        ys.append(batch["emotion"].cpu())
        preds.append(logits.argmax(1).cpu())
        meter.update(bi)
    meter.close()
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def metrics(y, p, n_classes):
    acc = float((y == p).mean())
    # weighted F1 (paper's primary emotion metric)
    f1s, weights = [], []
    for c in range(n_classes):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        f1s.append(f1)
        weights.append(int((y == c).sum()))
    wf1 = float(np.average(f1s, weights=weights)) if sum(weights) else 0.0
    return acc, wf1


def confusion(y, p, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, pr in zip(y, p):
        cm[t, pr] += 1
    return cm


def print_confusion(logger, cm, names):
    logger.raw(L.dim("    confusion (rows=true, cols=pred):"))
    head = "            " + " ".join(f"{n[:4]:>5}" for n in names)
    logger.raw(head)
    for i, n in enumerate(names):
        row = " ".join(f"{cm[i, j]:>5}" for j in range(len(names)))
        logger.raw(f"      {n[:8]:<8} {row}")


def main():
    ap = argparse.ArgumentParser(description="ACE-Net Stage-1 evaluation (Tables 2/3)")
    ap.add_argument("--branch", choices=["speech_text", "visual"], required=True)
    ap.add_argument("--dataset", choices=list(config.EMOTION_DATASETS), required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    class_names = config.EMOTION_DATASETS[args.dataset]["emotions"]
    nc = len(class_names)
    tag = f"{args.branch}_{args.dataset}"

    logger = L.Logger(config.CKPT_ROOT / f"eval_stage1_{tag}.log")
    L.run_header(logger, f"ACE-Net Stage-1 EVAL  ::  {args.branch}  {args.dataset}",
                 {"dataset": args.dataset, "label_space": f"{nc}-class",
                  "seed": cfg.seed, "split_ratio": "80/10/10 (test only)"})

    samples = manifests.build_emotion_samples(args.dataset)
    # IDENTICAL split to training (same seed, key_fn, ratios) -> same test set
    _, _, test_s = stratified_split(
        samples, key_fn=lambda s: s.emotion,
        ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio), seed=cfg.seed)
    logger.log(f"held-out test samples: {L.bold(str(len(test_s)))}")

    ckpt = args.ckpt or (config.CKPT_ROOT / f"stage1_{tag}.pt")
    if args.branch == "speech_text":
        model = SpeechTextModule(n_classes=nc).to(device)
    else:
        model = FVLiteNet(n_classes=nc).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    logger.log(f"loaded checkpoint <- {L.cyan(str(ckpt))}")

    dl = DataLoader(EmotionDataset(test_s), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers,
                    collate_fn=collate_emotion)
    y, p = _collect_preds(model, dl, args.branch, device)
    acc, wf1 = metrics(y, p, nc)

    logger.raw("")
    logger.raw(L.bold("  ===== Emotion Recognition Results (paper Table 2/3) ====="))
    logger.log(L.bold(L.green(
        f"  [{args.dataset.upper()}]  ACC {acc*100:.2f}%   "
        f"Weighted-F1 {wf1:.3f}   (n={len(y)})")))
    cm = confusion(y, p, nc)
    print_confusion(logger, cm, class_names)
    logger.close()


if __name__ == "__main__":
    main()
