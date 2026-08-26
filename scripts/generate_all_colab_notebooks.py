"""
scripts/generate_all_colab_notebooks.py — Master Generator for 36 Colab Preprocessing Notebooks.

Short, Clean Naming Convention:
Format: {MEMBER}_acc{N}_{DATASET}_s{SHARD}[_r2].ipynb
Examples:
  JC_acc1_MELD_s01.ipynb
  EL_acc2_MELD_s03.ipynb
  MATAN_acc1_MOSEI_s08_r2.ipynb
"""

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks" / "colab_shards"

# Clean old notebooks directory
if NOTEBOOKS_DIR.exists():
    shutil.rmtree(NOTEBOOKS_DIR)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# Master Assignment Matrix (Proportional: MATAN: 11, JC: 10, EL: 8, SHIKI: 7 = 36 total)
ASSIGNMENTS = [
    # --- MATAN (11 Shards: CMU-MOSEI 1 to 11) ---
    {"member": "MATAN", "acc_num": 1, "account": "exconde.matan30@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0001", "s_short": "s01", "round": 1},
    {"member": "MATAN", "acc_num": 2, "account": "johnmatters3008@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0002", "s_short": "s02", "round": 1},
    {"member": "MATAN", "acc_num": 3, "account": "mattersjohn3008@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0003", "s_short": "s03", "round": 1},
    {"member": "MATAN", "acc_num": 4, "account": "matan.exconde@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0004", "s_short": "s04", "round": 1},
    {"member": "MATAN", "acc_num": 5, "account": "baemonasa0417@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0005", "s_short": "s05", "round": 1},
    {"member": "MATAN", "acc_num": 6, "account": "baemonrora1408@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0006", "s_short": "s06", "round": 1},
    {"member": "MATAN", "acc_num": 7, "account": "baemonruka3008@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0007", "s_short": "s07", "round": 1},
    {"member": "MATAN", "acc_num": 1, "account": "exconde.matan30@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0008", "s_short": "s08", "round": 2},
    {"member": "MATAN", "acc_num": 2, "account": "johnmatters3008@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0009", "s_short": "s09", "round": 2},
    {"member": "MATAN", "acc_num": 3, "account": "mattersjohn3008@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0010", "s_short": "s10", "round": 2},
    {"member": "MATAN", "acc_num": 4, "account": "matan.exconde@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0011", "s_short": "s11", "round": 2},

    # --- JC (10 Shards: 1 of each dataset + Track 3 shards) ---
    {"member": "JC", "acc_num": 1, "account": "berlogred@gmail.com", "dataset": "MELD", "d_short": "MELD", "zip": "meld_raw.zip", "shard": "0001", "s_short": "s01", "round": 1},
    {"member": "JC", "acc_num": 2, "account": "caparasjc1025@gmail.com", "dataset": "TRACK_2", "d_short": "TRACK2", "zip": "tracks_1_2_3_4.zip", "shard": "0001", "s_short": "s01", "round": 1},
    {"member": "JC", "acc_num": 3, "account": "caparaschristine01@gmail.com", "dataset": "MUSTARD", "d_short": "MUSTARD", "zip": "mustard.zip", "shard": "0001", "s_short": "s01", "round": 1},
    {"member": "JC", "acc_num": 4, "account": "promptingacc2@gmail.com", "dataset": "TRACK_1", "d_short": "TRACK1", "zip": "tracks_1_2_3_4.zip", "shard": "0001", "s_short": "s01", "round": 1},
    {"member": "JC", "acc_num": 5, "account": "promptingacc@gmail.com", "dataset": "TRACK_2", "d_short": "TRACK2", "zip": "tracks_1_2_3_4.zip", "shard": "0002", "s_short": "s02", "round": 1},
    {"member": "JC", "acc_num": 6, "account": "johnchristan.caparas.lexmeet@gmail.com", "dataset": "TRACK_3", "d_short": "TRACK3", "zip": "tracks_1_2_3_4.zip", "shard": "0001", "s_short": "s01", "round": 1},
    {"member": "JC", "acc_num": 1, "account": "berlogred@gmail.com", "dataset": "TRACK_3", "d_short": "TRACK3", "zip": "tracks_1_2_3_4.zip", "shard": "0002", "s_short": "s02", "round": 2},
    {"member": "JC", "acc_num": 2, "account": "caparasjc1025@gmail.com", "dataset": "TRACK_3", "d_short": "TRACK3", "zip": "tracks_1_2_3_4.zip", "shard": "0003", "s_short": "s03", "round": 2},
    {"member": "JC", "acc_num": 3, "account": "caparaschristine01@gmail.com", "dataset": "TRACK_3", "d_short": "TRACK3", "zip": "tracks_1_2_3_4.zip", "shard": "0004", "s_short": "s04", "round": 2},
    {"member": "JC", "acc_num": 4, "account": "promptingacc2@gmail.com", "dataset": "TRACK_3", "d_short": "TRACK3", "zip": "tracks_1_2_3_4.zip", "shard": "0005", "s_short": "s05", "round": 2},

    # --- EL (8 Shards: MELD shards 2 to 8 + Track 2 shard 3) ---
    {"member": "EL", "acc_num": 1, "account": "elc0re143@gmail.com", "dataset": "MELD", "d_short": "MELD", "zip": "meld_raw.zip", "shard": "0002", "s_short": "s02", "round": 1},
    {"member": "EL", "acc_num": 2, "account": "micofeipao@gmail.com", "dataset": "MELD", "d_short": "MELD", "zip": "meld_raw.zip", "shard": "0003", "s_short": "s03", "round": 1},
    {"member": "EL", "acc_num": 3, "account": "gjrvlio.dev@gmail.com", "dataset": "MELD", "d_short": "MELD", "zip": "meld_raw.zip", "shard": "0004", "s_short": "s04", "round": 1},
    {"member": "EL", "acc_num": 4, "account": "geueljohn.rivera.lexmeet@gmail.com", "dataset": "MELD", "d_short": "MELD", "zip": "meld_raw.zip", "shard": "0005", "s_short": "s05", "round": 1},
    {"member": "EL", "acc_num": 5, "account": "gelj.riv@gmail.com", "dataset": "MELD", "d_short": "MELD", "zip": "meld_raw.zip", "shard": "0006", "s_short": "s06", "round": 1},
    {"member": "EL", "acc_num": 1, "account": "elc0re143@gmail.com", "dataset": "MELD", "d_short": "MELD", "zip": "meld_raw.zip", "shard": "0007", "s_short": "s07", "round": 2},
    {"member": "EL", "acc_num": 2, "account": "micofeipao@gmail.com", "dataset": "MELD", "d_short": "MELD", "zip": "meld_raw.zip", "shard": "0008", "s_short": "s08", "round": 2},
    {"member": "EL", "acc_num": 3, "account": "gjrvlio.dev@gmail.com", "dataset": "TRACK_2", "d_short": "TRACK2", "zip": "tracks_1_2_3_4.zip", "shard": "0003", "s_short": "s03", "round": 2},

    # --- SHIKI (7 Shards: Track 1 shards 2-3, Mustard shard 2, Track 2 shards 4-5, Track 3 shards 6-7) ---
    {"member": "SHIKI", "acc_num": 1, "account": "Shikina.cabral.lexmeet@gmail.com", "dataset": "TRACK_1", "d_short": "TRACK1", "zip": "tracks_1_2_3_4.zip", "shard": "0002", "s_short": "s02", "round": 1},
    {"member": "SHIKI", "acc_num": 2, "account": "cshikina18@gmail.com", "dataset": "TRACK_1", "d_short": "TRACK1", "zip": "tracks_1_2_3_4.zip", "shard": "0003", "s_short": "s03", "round": 1},
    {"member": "SHIKI", "acc_num": 3, "account": "forbaselineai@gmail.com", "dataset": "MUSTARD", "d_short": "MUSTARD", "zip": "mustard.zip", "shard": "0002", "s_short": "s02", "round": 1},
    {"member": "SHIKI", "acc_num": 4, "account": "shikinaexoexo@gmail.com", "dataset": "TRACK_2", "d_short": "TRACK2", "zip": "tracks_1_2_3_4.zip", "shard": "0004", "s_short": "s04", "round": 1},
    {"member": "SHIKI", "acc_num": 1, "account": "Shikina.cabral.lexmeet@gmail.com", "dataset": "TRACK_2", "d_short": "TRACK2", "zip": "tracks_1_2_3_4.zip", "shard": "0005", "s_short": "s05", "round": 2},
    {"member": "SHIKI", "acc_num": 2, "account": "cshikina18@gmail.com", "dataset": "TRACK_3", "d_short": "TRACK3", "zip": "tracks_1_2_3_4.zip", "shard": "0006", "s_short": "s06", "round": 2},
    {"member": "SHIKI", "acc_num": 3, "account": "forbaselineai@gmail.com", "dataset": "TRACK_3", "d_short": "TRACK3", "zip": "tracks_1_2_3_4.zip", "shard": "0007", "s_short": "s07", "round": 2},
]


def create_notebook(item):
    member = item["member"]
    acc_num = item["acc_num"]
    acc = item["account"]
    dataset = item["dataset"]
    d_short = item["d_short"]
    zip_file = item["zip"]
    shard = item["shard"]
    s_short = item["s_short"]
    round_num = item["round"]
    
    round_tag = "" if round_num == 1 else "_r2"
    # Concise naming format: {MEMBER}_acc{N}_{DATASET}_{s_short}[_r2].ipynb
    nb_filename = f"{member}_acc{acc_num}_{d_short}_{s_short}{round_tag}.ipynb"
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# ACE-Net Shard Preprocessor\n",
                f"### Assigned: **{member} (Account {acc_num})** | `{acc}`\n",
                f"### Target: **{dataset}** | **`shard_{shard}`** (Round {round_num})\n",
                f"### Output Target: `Google Drive > THESIS_MOTHERFILE > Baseline preprocessed > {dataset}`"
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
                "!git checkout feat/training-and-preprocessing-jc-turnover\n",
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
            "source": [f"## Step 4: Unzip Raw Dataset (`{zip_file}`) to Local Colab SSD"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"import os, zipfile, shutil\n",
                f"\n",
                f"DRIVE_ZIP = '/content/drive/MyDrive/THESIS_MOTHERFILE/datasets/{zip_file}'\n",
                f"LOCAL_RAW = '/content/data/raw/{dataset}'\n",
                f"\n",
                f"os.makedirs(LOCAL_RAW, exist_ok=True)\n",
                f"if not os.path.exists(DRIVE_ZIP):\n",
                f"    raise FileNotFoundError(f'Raw zip not found in Drive: {{DRIVE_ZIP}}')\n",
                f"\n",
                f"print(f'Unzipping {{DRIVE_ZIP}} to local SSD ({{LOCAL_RAW}})...')\n",
                f"with zipfile.ZipFile(DRIVE_ZIP, 'r') as z:\n",
                f"    z.extractall(LOCAL_RAW)\n",
                f"print('Unzip complete! Local files ready.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## Step 5: Execute Preprocessing for `{dataset}` [shard_{shard}]"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"!python scripts/preprocess/run_shard.py \\\n",
                f"    --account '{acc}' \\\n",
                f"    --dataset '{dataset}' \\\n",
                f"    --shard '{shard}' \\\n",
                f"    --raw_dir '/content/data/raw/{dataset}' \\\n",
                f"    --drive_root '/content/drive/MyDrive/THESIS_MOTHERFILE' \\\n",
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
                f"ckpt_file = Path('/content/drive/MyDrive/THESIS_MOTHERFILE/Baseline preprocessed/{dataset}/checkpoints/shard_{shard}_checkpoint.json')\n",
                f"out_shard = Path('/content/drive/MyDrive/THESIS_MOTHERFILE/Baseline preprocessed/{dataset}/shards/shard_{shard}')\n",
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
    print("  GENERATING CONCISE, SHORT-NAMED COLAB SHARD NOTEBOOKS")
    print("=" * 70)
    
    for item in ASSIGNMENTS:
        fn = create_notebook(item)
        print(f"[{item['member']:<5}] -> {fn:<35} (Account {item['acc_num']}: {item['account']})")
        
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Generated all {len(ASSIGNMENTS)} clean Colab notebooks in:")
    print(f"  -> {NOTEBOOKS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
