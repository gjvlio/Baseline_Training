"""
Inspect preprocessed outputs for a single CREMA-D sample.
Shows actual .npy contents — audio shape, decoded transcript, visual frames.

Usage:
    python -m preprocessing.tools.inspect_sample \
        --file-id 1001_DFA_ANG_XX \
        --dataset cremad

    python -m preprocessing.tools.inspect_sample \
        --file-id 1049_IEO_DIS_XX \
        --dataset cremad_deepfake

    # inspect all samples in a manifest (first N rows)
    python -m preprocessing.tools.inspect_sample \
        --manifest outputs/cremad_deepfake/cremad_deepfake_manifest_patched.csv \
        --n 5
"""
import argparse
import os
import sys
import numpy as np


def _out_dir(dataset: str) -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs")
    return os.path.join(base, dataset)


def inspect_file(file_id: str, out_dir: str, decode_text: bool = True) -> None:
    print(f"\n{'='*55}")
    print(f"  {file_id}")
    print(f"{'='*55}")

    # ── audio ────────────────────────────────────────────────
    mel_path = os.path.join(out_dir, "audio", f"{file_id}_melspec.npy")
    if os.path.exists(mel_path):
        mel = np.load(mel_path)
        print(f"[audio]  melspec shape : {mel.shape}  (expected 80 x T)")
        print(f"         min={mel.min():.2f}  max={mel.max():.2f}  mean={mel.mean():.2f}")
    else:
        print(f"[audio]  MISSING: {mel_path}")

    # ── text ─────────────────────────────────────────────────
    ids_path  = os.path.join(out_dir, "text", f"{file_id}_input_ids.npy")
    mask_path = os.path.join(out_dir, "text", f"{file_id}_attention_mask.npy")
    if os.path.exists(ids_path):
        ids  = np.load(ids_path)
        mask = np.load(mask_path) if os.path.exists(mask_path) else None
        print(f"[text]   input_ids shape: {ids.shape}  (expected 1 x 128)")
        token_len = int(mask.sum()) if mask is not None else "?"
        print(f"         active tokens  : {token_len}/128")
        if decode_text:
            try:
                from transformers import BertTokenizer
                tok = BertTokenizer.from_pretrained("bert-base-uncased")
                decoded = tok.decode(ids[0], skip_special_tokens=True)
                print(f"         decoded        : \"{decoded}\"")
            except Exception as e:
                print(f"         decode failed  : {e}")
    else:
        print(f"[text]   MISSING: {ids_path}")

    # ── visual ───────────────────────────────────────────────
    vis_dir = os.path.join(out_dir, "visual", file_id)
    if os.path.isdir(vis_dir):
        jpgs = sorted([f for f in os.listdir(vis_dir) if f.endswith(".jpg")])
        w_path = os.path.join(vis_dir, "keyframe_weights.npy")
        print(f"[visual] keyframes     : {len(jpgs)}/8  {jpgs}")
        if os.path.exists(w_path):
            weights = np.load(w_path)
            print(f"         attn weights   : {np.round(weights, 3)}")
        else:
            print(f"         keyframe_weights.npy MISSING")
    else:
        print(f"[visual] MISSING dir   : {vis_dir}")


def inspect_manifest(csv_path: str, n: int, decode_text: bool = True) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path).head(n)
    dataset = "cremad_deepfake" if "deepfake" in csv_path else "cremad"
    out_dir = _out_dir(dataset)
    for _, row in df.iterrows():
        inspect_file(str(row["file_id"]), out_dir, decode_text)
    print(f"\n{'='*55}")
    print(f"inspected {len(df)} samples from {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file-id", help="single file_id to inspect")
    group.add_argument("--manifest", help="CSV manifest path — inspect first N rows")
    parser.add_argument("--dataset", default="cremad", choices=["cremad", "cremad_deepfake", "meld", "savee"])
    parser.add_argument("--n", type=int, default=5, help="rows to inspect from manifest")
    parser.add_argument("--no-decode", action="store_true", help="skip BERT decode (faster)")
    args = parser.parse_args()

    if args.file_id:
        out_dir = _out_dir(args.dataset)
        inspect_file(args.file_id, out_dir, decode_text=not args.no_decode)
    else:
        inspect_manifest(args.manifest, args.n, decode_text=not args.no_decode)


if __name__ == "__main__":
    main()
