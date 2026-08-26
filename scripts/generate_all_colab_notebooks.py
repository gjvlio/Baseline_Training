"""
scripts/generate_all_colab_notebooks.py — Master Generator for EXACTLY 22 Single-Round Colab Notebooks.

One Account = One Notebook = One Shard.
No Round 2. Total Runtime: ~50 to 65 minutes across 22 concurrent accounts.
"""

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks" / "colab_shards"

if NOTEBOOKS_DIR.exists():
    shutil.rmtree(NOTEBOOKS_DIR)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# Master 22-Account Assignment Matrix
ASSIGNMENTS = [
    # --- MATAN (7 Accounts -> Entire CMU-MOSEI: 7 Shards) ---
    {"member": "MATAN", "acc_num": 1, "account": "exconde.matan30@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0001", "s_short": "s01"},
    {"member": "MATAN", "acc_num": 2, "account": "johnmatters3008@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0002", "s_short": "s02"},
    {"member": "MATAN", "acc_num": 3, "account": "mattersjohn3008@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0003", "s_short": "s03"},
    {"member": "MATAN", "acc_num": 4, "account": "matan.exconde@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0004", "s_short": "s04"},
    {"member": "MATAN", "acc_num": 5, "account": "baemonasa0417@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0005", "s_short": "s05"},
    {"member": "MATAN", "acc_num": 6, "account": "baemonrora1408@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0006", "s_short": "s06"},
    {"member": "MATAN", "acc_num": 7, "account": "baemonruka3008@gmail.com", "dataset": "CMU-MOSEI", "d_short": "MOSEI", "zip": "cmumosei.zip", "shard": "0007", "s_short": "s07"},

    # --- EL (5 Accounts -> Entire MELD: 5 Shards) ---
    {"member": "EL", "acc_num": 1, "account": "elc0re143@gmail.com", "dataset": "MELD", "d_short": "MELD", "zip": "meld_raw.zip", "shard": "0001", "s_short": "s01"},
    {"member": "EL", "acc_num": 2, "account": "micofeipao@gmail.com", "dataset": "MELD", "d_short": "MELD", "zip": "meld_raw.zip", "shard": "0002", "s_short": "s02"},
    {"member": "EL", "acc_num": 3, "account": "gjrvlio.dev@gmail.com", "dataset": "MELD", "d_short": "MELD", "zip": "meld_raw.zip", "shard": "0003", "s_short": "s03"},
    {"member": "EL", "acc_num": 4, "account": "geueljohn.rivera.lexmeet@gmail.com", "dataset": "MELD", "d_short": "MELD", "zip": "meld_raw.zip", "shard": "0004", "s_short": "s04"},
    {"member": "EL", "acc_num": 5, "account": "gelj.riv@gmail.com", "dataset": "MELD", "d_short": "MELD", "zip": "meld_raw.zip", "shard": "0005", "s_short": "s05"},

    # --- SHIKI (4 Accounts -> Track 1 [2 shards], Mustard [1 shard], Track 2 [1 shard]) ---
    {"member": "SHIKI", "acc_num": 1, "account": "Shikina.cabral.lexmeet@gmail.com", "dataset": "TRACK_1", "d_short": "TRACK1", "zip": "tracks_1_2_3_4.zip", "shard": "0001", "s_short": "s01"},
    {"member": "SHIKI", "acc_num": 2, "account": "cshikina18@gmail.com", "dataset": "TRACK_1", "d_short": "TRACK1", "zip": "tracks_1_2_3_4.zip", "shard": "0002", "s_short": "s02"},
    {"member": "SHIKI", "acc_num": 3, "account": "forbaselineai@gmail.com", "dataset": "MUSTARD", "d_short": "MUSTARD", "zip": "mustard.zip", "shard": "0001", "s_short": "s01"},
    {"member": "SHIKI", "acc_num": 4, "account": "shikinaexoexo@gmail.com", "dataset": "TRACK_2", "d_short": "TRACK2", "zip": "tracks_1_2_3_4.zip", "shard": "0001", "s_short": "s01"},

    # --- JC (6 Accounts -> Track 2 [2 shards], Entire Track 3 [4 shards]) ---
    {"member": "JC", "acc_num": 1, "account": "berlogred@gmail.com", "dataset": "TRACK_2", "d_short": "TRACK2", "zip": "tracks_1_2_3_4.zip", "shard": "0002", "s_short": "s02"},
    {"member": "JC", "acc_num": 2, "account": "caparasjc1025@gmail.com", "dataset": "TRACK_2", "d_short": "TRACK2", "zip": "tracks_1_2_3_4.zip", "shard": "0003", "s_short": "s03"},
    {"member": "JC", "acc_num": 3, "account": "caparaschristine01@gmail.com", "dataset": "TRACK_3", "d_short": "TRACK3", "zip": "tracks_1_2_3_4.zip", "shard": "0001", "s_short": "s01"},
    {"member": "JC", "acc_num": 4, "account": "promptingacc2@gmail.com", "dataset": "TRACK_3", "d_short": "TRACK3", "zip": "tracks_1_2_3_4.zip", "shard": "0002", "s_short": "s02"},
    {"member": "JC", "acc_num": 5, "account": "promptingacc@gmail.com", "dataset": "TRACK_3", "d_short": "TRACK3", "zip": "tracks_1_2_3_4.zip", "shard": "0003", "s_short": "s03"},
    {"member": "JC", "acc_num": 6, "account": "johnchristan.caparas.lexmeet@gmail.com", "dataset": "TRACK_3", "d_short": "TRACK3", "zip": "tracks_1_2_3_4.zip", "shard": "0004", "s_short": "s04"},
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
    
    nb_filename = f"{member}_acc{acc_num}_{d_short}_{s_short}.ipynb"
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# ACE-Net Universal Shard Preprocessor (One-Pass Suite)\n",
                f"### Assigned Member: **{member} (Account {acc_num})**\n",
                f"### Target Account: `{acc}`\n",
                f"### Target Dataset: **{dataset}** | Shard: **`shard_{shard}`**\n",
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
            "source": [f"## Step 4: Fast Unzip Raw Dataset (`{zip_file}`) to Local Colab SSD"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"import os\n",
                f"\n",
                f"DRIVE_ZIP = '/content/drive/MyDrive/THESIS_MOTHERFILE/datasets/{zip_file}'\n",
                f"LOCAL_RAW = '/content/data/raw/{dataset}'\n",
                f"\n",
                f"os.makedirs(LOCAL_RAW, exist_ok=True)\n",
                f"if not os.path.exists(DRIVE_ZIP):\n",
                f"    raise FileNotFoundError(f'Raw zip not found in Drive: {{DRIVE_ZIP}}')\n",
                f"\n",
                f"print(f'Fast unzipping {{DRIVE_ZIP}} to local SSD ({{LOCAL_RAW}})...')\n",
                f"!unzip -q -n '{{DRIVE_ZIP}}' -d '{{LOCAL_RAW}}'\n",
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
    print("  GENERATING EXACTLY 22 ONE-PASS PRODUCTION NOTEBOOKS")
    print("=" * 70)
    
    for item in ASSIGNMENTS:
        fn = create_notebook(item)
        print(f"[{item['member']:<5}] -> {fn:<30} (Account {item['acc_num']}: {item['account']})")
        
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Generated all {len(ASSIGNMENTS)} production notebooks in:")
    print(f"  -> {NOTEBOOKS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
