"""
scripts/generate_val_test_notebooks.py — Generates Clean Colab Notebooks for VAL and TEST Sets.

Target Output Folder Structure on Google Drive:
THESIS_MOTHERFILE /
└── baseline_training /
    ├── train / (Train set features from the 22 shards)
    ├── val / (Val set features: TRACKS_MELD & MOSEI_MUSTARD)
    │   ├── TRACKS_MELD / shards / shard_0001
    │   └── MOSEI_MUSTARD / shards / shard_0001
    └── test / (Test set features: TRACKS_MELD & MOSEI_MUSTARD)
        ├── TRACKS_MELD / shards / shard_0001
        └── MOSEI_MUSTARD / shards / shard_0001
"""

import json
import shutil
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks" / "colab_val_test"

if NOTEBOOKS_DIR.exists():
    shutil.rmtree(NOTEBOOKS_DIR)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

VAL_CSV = REPO_ROOT / "Manifests" / "val_manifest.csv"
TEST_CSV = REPO_ROOT / "Manifests" / "internal_test_manifest.csv"

SHARDS_ROOT = REPO_ROOT / "data" / "manifests" / "eval_shards"
SHARDS_ROOT_2 = REPO_ROOT / "Manifests" / "eval_shards"

for sr in [SHARDS_ROOT, SHARDS_ROOT_2]:
    if sr.exists():
        shutil.rmtree(sr)
    sr.mkdir(parents=True, exist_ok=True)

TRACKS_MELD_PIPELINES = {"track1", "track2", "track3", "meld_real"}
MOSEI_MUSTARD_PIPELINES = {"mosei_real", "mustard"}

def partition_by_groups(source_csv, split_name):
    with open(source_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    group_tracks_meld = [r for r in rows if r.get("source_pipeline") in TRACKS_MELD_PIPELINES]
    group_mosei_mustard = [r for r in rows if r.get("source_pipeline") in MOSEI_MUSTARD_PIPELINES]
    
    # Sort deterministically
    group_tracks_meld.sort(key=lambda r: (r.get("source_pipeline", ""), r.get("clip_id", "")))
    group_mosei_mustard.sort(key=lambda r: (r.get("source_pipeline", ""), r.get("clip_id", "")))
    
    shards_info = [
        ("TRACKS_MELD", group_tracks_meld),
        ("MOSEI_MUSTARD", group_mosei_mustard)
    ]
    
    for gname, grows in shards_info:
        for dest in [SHARDS_ROOT / split_name, SHARDS_ROOT_2 / split_name]:
            dest.mkdir(parents=True, exist_ok=True)
            p = dest / f"{gname}_manifest.csv"
            with open(p, "w", newline="", encoding="utf-8") as out_f:
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(grows)
        print(f"[{split_name}] {gname:<15}: {len(grows)} clips")

ASSIGNMENTS = [
    # --- VALIDATION SET (Target folder: baseline_training/val) ---
    {
        "split": "val",
        "group": "TRACKS_MELD",
        "name": "VAL_acc1_TRACKS_MELD.ipynb",
        "zips": ["tracks_1_2_3_4.zip", "meld_raw.zip"],
        "manifest_rel": "data/manifests/eval_shards/val/TRACKS_MELD_manifest.csv"
    },
    {
        "split": "val",
        "group": "MOSEI_MUSTARD",
        "name": "VAL_acc2_MOSEI_MUSTARD.ipynb",
        "zips": ["cmumosei.zip", "mustard.zip"],
        "manifest_rel": "data/manifests/eval_shards/val/MOSEI_MUSTARD_manifest.csv"
    },
    # --- TESTING SET (Target folder: baseline_training/test) ---
    {
        "split": "test",
        "group": "TRACKS_MELD",
        "name": "TEST_acc1_TRACKS_MELD.ipynb",
        "zips": ["tracks_1_2_3_4.zip", "meld_raw.zip"],
        "manifest_rel": "data/manifests/eval_shards/test/TRACKS_MELD_manifest.csv"
    },
    {
        "split": "test",
        "group": "MOSEI_MUSTARD",
        "name": "TEST_acc2_MOSEI_MUSTARD.ipynb",
        "zips": ["cmumosei.zip", "mustard.zip"],
        "manifest_rel": "data/manifests/eval_shards/test/MOSEI_MUSTARD_manifest.csv"
    },
]

def create_eval_notebook(item):
    split = item["split"]
    group = item["group"]
    nb_filename = item["name"]
    zips = item["zips"]
    manifest_rel = item["manifest_rel"]
    
    zip_unzip_commands = []
    for z in zips:
        zip_unzip_commands.append(f"!unzip -q -n '/content/drive/MyDrive/THESIS_MOTHERFILE/datasets/{z}' -d '/content/data/raw'")

    zip_unzip_str = "\n".join(zip_unzip_commands)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# ACE-Net Preprocessor — {split.upper()} SET [{group}]\n",
                f"### Target Evaluation Split: **{split.upper()} SET** ({group})\n",
                f"### Output Target: `Google Drive > THESIS_MOTHERFILE > baseline_training > {split} > {group} > shards > shard_0001`"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 1: Connect to GPU & Mount Google Drive"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "import os, sys, torch\n",
                "\n",
                "drive.mount('/content/drive')\n",
                "print('GPU Available:', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('Device:', torch.cuda.get_device_name(0))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 2: Clone Repository & Checkout Branch"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%cd /content\n",
                "!rm -rf Baseline_Training\n",
                "!git clone https://github.com/gjvlio/Baseline_Training.git\n",
                "%cd Baseline_Training\n",
                "!git checkout feat/baseline-preprocessing-jc\n",
                "!git pull\n",
                "!git log --oneline -1"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 3: Install Required Dependencies"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q --no-deps facenet-pytorch\n",
                "!pip install -q --no-deps git+https://github.com/openai/whisper.git\n",
                "print('Dependencies installed successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## Step 4: Fast Unzip Assigned Datasets ({', '.join(zips)}) to Local SSD"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "LOCAL_RAW = '/content/data/raw'\n",
                "os.makedirs(LOCAL_RAW, exist_ok=True)\n",
                "\n",
                f"print('Fast unzipping {zips} to local SSD (/content/data/raw)...')\n",
                f"{zip_unzip_str}\n",
                "print('Unzip complete! Local files ready.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## Step 5: Execute Preprocessing for `{split}` [{group}]"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"!python scripts/preprocess/run_shard.py \\\n",
                f"    --account 'eval_runner@gmail.com' \\\n",
                f"    --dataset '{split}/{group}' \\\n",
                f"    --shard '0001' \\\n",
                f"    --manifest '/content/Baseline_Training/{manifest_rel}' \\\n",
                f"    --raw_dir '/content/data/raw' \\\n",
                f"    --drive_root '/content/drive/MyDrive/THESIS_MOTHERFILE/baseline_training' \\\n",
                f"    --device cuda"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 6: Post-Run Integrity Check & Verification"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"import json, glob\n",
                f"from pathlib import Path\n",
                f"\n",
                f"ckpt_file = Path('/content/drive/MyDrive/THESIS_MOTHERFILE/baseline_training/{split}/{group}/checkpoints/shard_0001_checkpoint.json')\n",
                f"out_shard = Path('/content/drive/MyDrive/THESIS_MOTHERFILE/baseline_training/{split}/{group}/shards/shard_0001')\n",
                f"\n",
                f"if ckpt_file.exists():\n",
                f"    with open(ckpt_file) as f:\n",
                f"        data = json.load(f)\n",
                f"    print('=' * 60)\n",
                f"    print('SHARD STATUS:', data.get('status'))\n",
                f"    print('Completed Clips:', len(data.get('completed_ids', [])))\n",
                f"    print('Failed Clips   :', len(data.get('failed_ids', [])))\n",
                f"    print('=' * 60)\n",
                f"else:\n",
                f"    print('Checkpoint not found. Run Step 5 first.')"
            ]
        }
    ]
    
    nb_json = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    target_path = NOTEBOOKS_DIR / nb_filename
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=2)
    return nb_filename

def main():
    print("=" * 70)
    print("  PARTITIONING VAL & TEST SETS INTO baseline_training/{val, test}")
    print("=" * 70)
    
    partition_by_groups(VAL_CSV, "val")
    partition_by_groups(TEST_CSV, "test")
    
    for item in ASSIGNMENTS:
        fn = create_eval_notebook(item)
        print(f"Generated: {fn}")
        
    print("=" * 70)
    print(f"[SUCCESS] Generated 4 clean notebooks in: {NOTEBOOKS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
