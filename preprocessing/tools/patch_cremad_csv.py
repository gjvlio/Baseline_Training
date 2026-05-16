"""
Patch a CREMA-D manifest CSV that has empty transcripts.

Applies the known CREMA-D fixed-sentence lookup (SentenceCode from file_id)
without reprocessing audio. Audio/text .npy files on disk are untouched.

Usage:
    python -m preprocessing.tools.patch_cremad_csv \
        --csv outputs/cremad_deepfake/cremad_deepfake_manifest_complete.csv \
        --out outputs/cremad_deepfake/cremad_deepfake_manifest_patched.csv
"""
import argparse
import pandas as pd

SENTENCE_MAP = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slippery",
    "WSI": "We'll stop in a bit",
}


def sentence_from_file_id(file_id: str) -> str:
    parts = str(file_id).split("_")
    if len(parts) >= 2:
        return SENTENCE_MAP.get(parts[1].upper(), "")
    return ""


def patch(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)

    before = df["transcript"].isna().sum() + (df["transcript"] == "").sum()

    # fill empty / NaN transcripts from sentence map
    def _fix(row):
        t = row["transcript"]
        if pd.isna(t) or str(t).strip() == "":
            return sentence_from_file_id(row["file_id"])
        return t

    df["transcript"] = df.apply(_fix, axis=1)

    after = df["transcript"].isna().sum() + (df["transcript"] == "").sum()

    df.to_csv(out_path, index=False)
    print(f"patched: {before - after} transcripts filled | {after} still empty")
    print(f"saved -> {out_path}")

    # quick summary
    print(f"\n--- manifest summary ---")
    print(f"total rows   : {len(df)}")
    print(f"emotion dist :\n{df['emotion'].value_counts().to_string()}")
    print(f"visual_ok    : {df['visual_ok'].sum() if 'visual_ok' in df.columns else 'column missing'}")
    print(f"n_faces>0    : {(df['n_faces'] > 0).sum()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    patch(args.csv, args.out)


if __name__ == "__main__":
    main()
