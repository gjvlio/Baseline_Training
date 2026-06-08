"""
Stage 1: train the unimodal emotion feature extractors on GENUINE data only
(Section 3.5). Each branch is trained independently as an emotion classifier;
the learned weights are then frozen for Stage 2.

  --branch speech_text   trains SpeechTextModule (MDCNN + cross-attn)
  --branch visual        trains FVLiteNet

Examples:
    python -m src.train_stage1 --branch speech_text
    python -m src.train_stage1 --branch visual --batch-size 8
    python -m src.train_stage1 --branch visual --splits crema_genuine_lasthalf
"""
import argparse
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .config import TrainConfig
from .data import manifests
from .data.dataset import EmotionDataset, collate_emotion
from .models.speech_text import SpeechTextModule
from .models.fv_litenet import FVLiteNet
from .train_utils import set_seed, stratified_split, EarlyStopper
from . import logging_utils as L


def move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward_logits(model, branch, b):
    if branch == "speech_text":
        return model(b["melspec"], b["mel_lengths"], b["input_ids"], b["attention_mask"])
    return model(b["frames"], b["alpha"], b["frame_mask"])


def run_epoch(model, loader, branch, device, epoch, max_epochs,
              n_classes, optimizer=None, logger=None):
    train = optimizer is not None
    model.train(train)
    crit = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    phase = "train" if train else "val"
    meter = L.ProgressMeter(len(loader), epoch, max_epochs, phase, every=5)

    for bi, batch in enumerate(loader):
        batch = move(batch, device)
        y = batch["emotion"]
        logits, _ = forward_logits(model, branch, batch)
        loss = crit(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        n += bs
        for t, p in zip(y.view(-1), pred.view(-1)):
            confusion[t.long(), p.long()] += 1
        meter.update(bi, loss=total_loss / n, acc=correct / n)

    dur = meter.close()
    return total_loss / max(n, 1), correct / max(n, 1), confusion, dur


def main():
    ap = argparse.ArgumentParser(description="ACE-Net Stage-1 unimodal training")
    ap.add_argument("--branch", choices=["speech_text", "visual"], required=True)
    ap.add_argument("--dataset", choices=list(config.EMOTION_DATASETS), required=True,
                    help="train in this corpus's native label space (paper: per-dataset)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--early-stop", type=int, default=None,
                    help="override early-stop patience (paper default 25)")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="hard wall-clock cap; stop after this many minutes")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable train-time data augmentation")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap #samples (debug/quick smoke run)")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.epochs:      cfg.max_epochs = args.epochs
    if args.early_stop is not None: cfg.early_stop_patience = args.early_stop
    if args.lr:          cfg.lr = args.lr
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    class_names = config.EMOTION_DATASETS[args.dataset]["emotions"]
    n_classes = len(class_names)

    config.CKPT_ROOT.mkdir(exist_ok=True)
    tag = f"{args.branch}_{args.dataset}"
    logger = L.Logger(config.CKPT_ROOT / f"stage1_{tag}.log")
    L.run_header(
        logger, f"ACE-Net Stage 1  ::  branch = {args.branch}  dataset = {args.dataset}",
        {
            "dataset": args.dataset,
            "label_space": f"{n_classes}-class {class_names}",
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "max_epochs": cfg.max_epochs,
            "early_stop": cfg.early_stop_patience,
            "num_workers": cfg.num_workers,
            "seed": cfg.seed,
            "d_model": config.D_MODEL,
        },
    )

    logger.log("resolving genuine emotion samples ...")
    samples = manifests.build_emotion_samples(args.dataset)
    if args.limit:
        samples = samples[:args.limit]
        logger.log(L.red(f"[debug] limited to {len(samples)} samples"))
    logger.log(f"resolved {L.bold(str(len(samples)))} usable samples")
    from collections import Counter
    dist = Counter(class_names[s.emotion] for s in samples)
    logger.raw("    label distribution: " +
               "  ".join(f"{k}={v}" for k, v in sorted(dist.items())))

    train_s, val_s, test_s = stratified_split(
        samples, key_fn=lambda s: s.emotion,
        ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio), seed=cfg.seed)
    logger.log(f"split -> train {L.bold(str(len(train_s)))}  "
               f"val {L.bold(str(len(val_s)))}  test {L.bold(str(len(test_s)))}")

    def make_dl(s, sh, aug=False):
        return DataLoader(EmotionDataset(s, augment=aug), batch_size=cfg.batch_size,
                          shuffle=sh, num_workers=cfg.num_workers,
                          collate_fn=collate_emotion, drop_last=sh,
                          pin_memory=(device == "cuda"))
    # train uses augmentation (anti-overfit); val/test do not
    train_dl = make_dl(train_s, True, aug=not args.no_augment)
    val_dl, test_dl = make_dl(val_s, False), make_dl(test_s, False)
    logger.log(f"augmentation: {'ON' if not args.no_augment else 'OFF'} (train only)")

    if args.branch == "speech_text":
        model = SpeechTextModule(n_classes=n_classes).to(device)
    else:
        model = FVLiteNet(n_classes=n_classes).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.log(f"model    : {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total params")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay)
    # cosine LR decay over the epoch budget (smoother convergence, less overfit)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs, eta_min=cfg.lr * 0.05)
    stopper = EarlyStopper(cfg.early_stop_patience)
    ckpt_path = config.CKPT_ROOT / f"stage1_{tag}.pt"

    logger.raw("")
    logger.log(L.bold("training ..."))
    run_t0 = datetime.now()
    epoch_durs = []
    best_acc = 0.0

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss, tr_acc, _, tr_dur = run_epoch(
            model, train_dl, args.branch, device, epoch, cfg.max_epochs,
            n_classes, optimizer, logger)
        with torch.no_grad():
            va_loss, va_acc, va_conf, _ = run_epoch(
                model, val_dl, args.branch, device, epoch, cfg.max_epochs,
                n_classes, None, logger)
        scheduler.step()
        epoch_durs.append(tr_dur)
        avg = sum(epoch_durs) / len(epoch_durs)
        eta_full = avg * (cfg.max_epochs - epoch)

        improved = va_acc > best_acc
        mark = L.green("  * best-acc -> saved") if improved else ""
        va_loss_s = L.yellow(f"{va_loss:.4f}")
        va_acc_s = L.yellow(f"{va_acc:.3f}")
        logger.log(
            f"epoch {epoch:03d}/{cfg.max_epochs}  "
            f"train[loss {tr_loss:.4f} acc {tr_acc:.3f}]  "
            f"val[loss {va_loss_s} acc {va_acc_s}]  "
            f"{tr_dur:.0f}s/ep  ETA {timedelta(seconds=int(eta_full))}{mark}")

        # Save the best-ACCURACY checkpoint (what Table 2/3 reports). Val loss
        # and val acc diverge under overfitting, so saving on best-loss can keep
        # a lower-accuracy epoch. Early-stop still tracks loss for a stable
        # stopping signal.
        if improved:
            best_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)
        stop = stopper.step(va_loss, model)
        # hard wall-clock cap: stop if the next epoch would blow the budget
        if args.max_minutes is not None:
            elapsed_min = (datetime.now() - run_t0).total_seconds() / 60
            if elapsed_min + (avg / 60) > args.max_minutes:
                logger.log(L.red(f"time cap: {elapsed_min:.1f}/{args.max_minutes} min "
                                 f"used, next epoch would exceed -> stopping @ epoch {epoch}"))
                break
        if stop:
            logger.log(L.red(f"early stop @ epoch {epoch} "
                             f"(best val loss {stopper.best:.4f}, patience {cfg.early_stop_patience})"))
            break

    logger.raw("")
    logger.log(f"loading best-val-ACC checkpoint for final test (val acc {best_acc:.3f}) ...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    with torch.no_grad():
        te_loss, te_acc, te_conf, _ = run_epoch(
            model, test_dl, args.branch, device, 0, cfg.max_epochs,
            n_classes, None, logger)
    logger.raw("")
    logger.log(L.bold(L.green(
        f"TEST  loss {te_loss:.4f}  acc {te_acc:.3f}")))
    L.per_class_report(logger, te_conf, class_names)

    # ckpt on disk is already the best-acc one; do not overwrite.
    total_time = datetime.now() - run_t0
    logger.raw("")
    logger.log(f"checkpoint saved -> {L.cyan(str(ckpt_path))}")
    logger.log(f"total wall time : {str(total_time).split('.')[0]}")
    logger.close()


if __name__ == "__main__":
    main()
