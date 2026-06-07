"""
Stage 2: consistency discrimination (Section 3.5).

Load frozen Stage-1 extractors, train ONLY the fusion + MLP discriminator.
  Negative class (label 0): genuine aligned audio-visual pairs.
  Positive class (label 1): forgeries, stratified 50/50 across
      P1 (emotional tampering) and P2 (cross-identity splice).
  1:1 genuine:forged balance.
Loss: BCE (BCEWithLogitsLoss). Metrics: AUC, ACC, Precision, Recall, F1.

Examples:
    python -m src.train_stage2
    python -m src.train_stage2 --batch-size 8
"""
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .config import TrainConfig
from .data import manifests
from .data.dataset import PairDataset, collate_pair
from .models.acenet import ACENet
from .train_utils import set_seed, stratified_split, EarlyStopper
from . import logging_utils as L


def move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_balanced(seed, logger):
    rng = random.Random(seed)
    genuine, fake = manifests.build_stage2_samples()
    p1 = [s for s in fake if s.emotion == "crema_fake_p1"]
    p2 = [s for s in fake if s.emotion == "crema_fake_p2"]
    logger.log(f"pool     : genuine {len(genuine)}  fake-P1(tamper) {len(p1)}  "
               f"fake-P2(splice) {len(p2)}")

    n_pos_each = min(len(p1), len(p2))
    rng.shuffle(p1); rng.shuffle(p2)
    fakes = p1[:n_pos_each] + p2[:n_pos_each]
    rng.shuffle(fakes)
    logger.log(f"positives: stratified 50/50 -> {n_pos_each} P1 + {n_pos_each} P2 "
               f"= {len(fakes)} fake")

    n = min(len(genuine), len(fakes))
    rng.shuffle(genuine)
    chosen = genuine[:n] + fakes[:n]
    for s in chosen:
        s.emotion = None
    rng.shuffle(chosen)
    logger.log(f"balanced : {L.bold(str(n))} genuine + {L.bold(str(n))} fake "
               f"= {len(chosen)} total (1:1)")
    return chosen


def binary_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(len(y), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


def auc_score(y, p):
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@torch.no_grad()
def evaluate(model, loader, device, epoch, max_epochs, phase="val"):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    losses, ys, ps = 0.0, [], []
    n = 0
    meter = L.ProgressMeter(len(loader), epoch, max_epochs, phase, every=5)
    for bi, batch in enumerate(loader):
        batch = move(batch, device)
        logit = model(batch)
        loss = crit(logit, batch["label"])
        bs = batch["label"].size(0)
        losses += loss.item() * bs; n += bs
        ys.append(batch["label"].cpu()); ps.append(torch.sigmoid(logit).cpu())
        meter.update(bi, loss=losses / n)
    meter.close()
    y = torch.cat(ys).numpy(); p = torch.cat(ps).numpy()
    m = binary_metrics(y, p)
    m["auc"] = auc_score(y, p)
    m["loss"] = losses / max(n, 1)
    return m


def main():
    ap = argparse.ArgumentParser(description="ACE-Net Stage-2 consistency discrimination")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--st-ckpt", default=None)
    ap.add_argument("--vis-ckpt", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap #samples (debug/quick smoke run)")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.epochs:      cfg.max_epochs = args.epochs
    if args.lr:          cfg.lr = args.lr
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    config.CKPT_ROOT.mkdir(exist_ok=True)
    logger = L.Logger(config.CKPT_ROOT / "stage2_acenet.log")
    L.run_header(
        logger, "ACE-Net Stage 2  ::  multimodal consistency discriminator",
        {
            "batch_size": cfg.batch_size, "lr": cfg.lr,
            "weight_decay": cfg.weight_decay, "max_epochs": cfg.max_epochs,
            "early_stop": cfg.early_stop_patience, "num_workers": cfg.num_workers,
            "seed": cfg.seed, "d_model": config.D_MODEL,
            "fusion": "concat|diff|product (4d)",
        },
    )

    model = ACENet().to(device)
    # Stage-2 forgeries are CREMA-based -> use the CREMA-trained extractors.
    st = args.st_ckpt or (config.CKPT_ROOT / "stage1_speech_text_crema.pt")
    vis = args.vis_ckpt or (config.CKPT_ROOT / "stage1_visual_crema.pt")
    if Path(st).exists():
        # strict=False: Stage-1 ckpt carries an emotion_head that Stage-2's
        # embedding-only module does not have (head=None). Backbone keys match.
        model.speech_text.load_state_dict(torch.load(st, map_location=device), strict=False)
        logger.log(f"loaded   : speech_text <- {L.cyan(str(st))}")
    else:
        logger.log(L.red(f"[warn] {st} missing -> speech_text RANDOM init"))
    if Path(vis).exists():
        model.visual.load_state_dict(torch.load(vis, map_location=device), strict=False)
        logger.log(f"loaded   : visual      <- {L.cyan(str(vis))}")
    else:
        logger.log(L.red(f"[warn] {vis} missing -> visual RANDOM init"))

    model.freeze_extractors()
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    logger.log(f"trainable: discriminator only, {disc_params/1e6:.2f}M params "
               f"(extractors frozen)")

    logger.raw("")
    logger.log("building balanced genuine/fake dataset ...")
    samples = build_balanced(cfg.seed, logger)
    if args.limit:
        samples = samples[:args.limit]
        logger.log(L.red(f"[debug] limited to {len(samples)} samples"))
    train_s, val_s, test_s = stratified_split(
        samples, key_fn=lambda s: s.label,
        ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio), seed=cfg.seed)
    logger.log(f"split    : train {L.bold(str(len(train_s)))}  "
               f"val {L.bold(str(len(val_s)))}  test {L.bold(str(len(test_s)))}")

    def make_dl(s, sh):
        return DataLoader(PairDataset(s), batch_size=cfg.batch_size, shuffle=sh,
                          num_workers=cfg.num_workers, collate_fn=collate_pair,
                          drop_last=sh, pin_memory=(device == "cuda"))
    train_dl, val_dl, test_dl = make_dl(train_s, True), make_dl(val_s, False), make_dl(test_s, False)

    optimizer = torch.optim.Adam(model.discriminator.parameters(),
                                 lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.BCEWithLogitsLoss()
    stopper = EarlyStopper(cfg.early_stop_patience)
    ckpt_path = config.CKPT_ROOT / "stage2_acenet.pt"

    logger.raw("")
    logger.log(L.bold("training ..."))
    run_t0 = datetime.now()
    epoch_durs = []
    best_auc = 0.0

    for epoch in range(1, cfg.max_epochs + 1):
        model.discriminator.train()
        meter = L.ProgressMeter(len(train_dl), epoch, cfg.max_epochs, "train", every=5)
        tot, n = 0.0, 0
        for bi, batch in enumerate(train_dl):
            batch = move(batch, device)
            logit = model(batch)
            loss = crit(logit, batch["label"])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            bs = batch["label"].size(0)
            tot += loss.item() * bs; n += bs
            meter.update(bi, loss=tot / n)
        tr_dur = meter.close()

        va = evaluate(model, val_dl, device, epoch, cfg.max_epochs, "val")
        epoch_durs.append(tr_dur)
        avg = sum(epoch_durs) / len(epoch_durs)
        eta = avg * (cfg.max_epochs - epoch)

        improved = va["auc"] > best_auc
        best_auc = max(best_auc, va["auc"])
        tag = L.green("  * best-auc") if improved else ""
        v_loss = L.yellow(f"{va['loss']:.4f}")
        v_auc = L.yellow(f"{va['auc']:.3f}")
        logger.log(
            f"epoch {epoch:03d}/{cfg.max_epochs}  train[loss {tot/max(n,1):.4f}]  "
            f"val[loss {v_loss} acc {va['acc']:.3f} auc {v_auc} "
            f"P {va['prec']:.3f} R {va['rec']:.3f} F1 {va['f1']:.3f}]  "
            f"{tr_dur:.0f}s/ep  ETA {timedelta(seconds=int(eta))}{tag}")

        stop = stopper.step(va["loss"], model.discriminator)
        if stopper.counter == 0:
            torch.save(model.state_dict(), ckpt_path)
        if stop:
            logger.log(L.red(f"early stop @ epoch {epoch} "
                             f"(best val loss {stopper.best:.4f})"))
            break

    logger.raw("")
    logger.log("restoring best-val-loss discriminator for final test ...")
    stopper.restore(model.discriminator)
    te = evaluate(model, test_dl, device, 0, cfg.max_epochs, "test")
    logger.raw("")
    logger.log(L.bold(L.green(
        f"TEST  AUC {te['auc']:.3f}  acc {te['acc']:.3f}  "
        f"P {te['prec']:.3f}  R {te['rec']:.3f}  F1 {te['f1']:.3f}")))
    logger.raw(L.dim(f"    confusion: TP {te['tp']}  TN {te['tn']}  "
                     f"FP {te['fp']}  FN {te['fn']}"))

    torch.save(model.state_dict(), ckpt_path)
    logger.raw("")
    logger.log(f"checkpoint saved -> {L.cyan(str(ckpt_path))}")
    logger.log(f"total wall time : {str(datetime.now() - run_t0).split('.')[0]}")
    logger.close()


if __name__ == "__main__":
    main()
