"""
scripts/generate_all_colab_notebooks.py — Master Generator for 36 Colab Preprocessing Notebooks.

Generates pre-configured, ready-to-run Jupyter notebooks for each of the 36 shards,
embedded with the assigned Gmail accounts, dataset names, and output configurations.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks" / "colab_shards"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# Master Assignment Matrix (Proportional: MATAN: 11, JC: 10, EL: 8, SHIKI: 7 = 36 total)
ASSIGNMENTS = [
    # --- MATAN (11 Shards: CMU-MOSEI 1 to 11) ---
    {"member": "MATAN", "account": "exconde.matan30@gmail.com", "dataset": "CMU-MOSEI", "zip": "cmumosei.zip", "shard": "0001", "round": 1},
    {"member": "MATAN", "account": "johnmatters3008@gmail.com", "dataset": "CMU-MOSEI", "zip": "cmumosei.zip", "shard": "0002", "round": 1},
    {"member": "MATAN", "account": "mattersjohn3008@gmail.com", "dataset": "CMU-MOSEI", "zip": "cmumosei.zip", "shard": "0003", "round": 1},
    {"member": "MATAN", "account": "matan.exconde@gmail.com", "dataset": "CMU-MOSEI", "zip": "cmumosei.zip", "shard": "0004", "round": 1},
    {"member": "MATAN", "account": "baemonasa0417@gmail.com", "dataset": "CMU-MOSEI", "zip": "cmumosei.zip", "shard": "0005", "round": 1},
    {"member": "MATAN", "account": "baemonrora1408@gmail.com", "dataset": "CMU-MOSEI", "zip": "cmumosei.zip", "shard": "0006", "round": 1},
    {"member": "MATAN", "account": "baemonruka3008@gmail.com", "dataset": "CMU-MOSEI", "zip": "cmumosei.zip", "shard": "0007", "round": 1},
    {"member": "MATAN", "account": "exconde.matan30@gmail.com", "dataset": "CMU-MOSEI", "zip": "cmumosei.zip", "shard": "0008", "round": 2},
    {"member": "MATAN", "account": "johnmatters3008@gmail.com", "dataset": "CMU-MOSEI", "zip": "cmumosei.zip", "shard": "0009", "round": 2},
    {"member": "MATAN", "account": "mattersjohn3008@gmail.com", "dataset": "CMU-MOSEI", "zip": "cmumosei.zip", "shard": "0010", "round": 2},
    {"member": "MATAN", "account": "matan.exconde@gmail.com", "dataset": "CMU-MOSEI", "zip": "cmumosei.zip", "shard": "0011", "round": 2},

    # --- JC (10 Shards: 1 of each dataset + Track 3 shards) ---
    {"member": "JC", "account": "berlogred@gmail.com", "dataset": "MELD", "zip": "meld_raw.zip", "shard": "0001", "round": 1},
    {"member": "JC", "account": "caparasjc1025@gmail.com", "dataset": "TRACK_2", "zip": "tracks_1_2_3_4.zip", "shard": "0001", "round": 1},
    {"member": "JC", "account": "caparaschristine01@gmail.com", "dataset": "MUSTARD", "zip": "mustard.zip", "shard": "0001", "round": 1},
    {"member": "JC", "account": "promptingacc2@gmail.com", "dataset": "TRACK_1", "zip": "tracks_1_2_3_4.zip", "shard": "0001", "round": 1},
    {"member": "JC", "account": "promptingacc@gmail.com", "dataset": "TRACK_2", "zip": "tracks_1_2_3_4.zip", "shard": "0002", "round": 1},
    {"member": "JC", "account": "johnchristan.caparas.lexmeet@gmail.com", "dataset": "TRACK_3", "zip": "tracks_1_2_3_4.zip", "shard": "0001", "round": 1},
    {"member": "JC", "account": "berlogred@gmail.com", "dataset": "TRACK_3", "zip": "tracks_1_2_3_4.zip", "shard": "0002", "round": 2},
    {"member": "JC", "account": "caparasjc1025@gmail.com", "dataset": "TRACK_3", "zip": "tracks_1_2_3_4.zip", "shard": "0003", "round": 2},
    {"member": "JC", "account": "caparaschristine01@gmail.com", "dataset": "TRACK_3", "zip": "tracks_1_2_3_4.zip", "shard": "0004", "round": 2},
    {"member": "JC", "account": "promptingacc2@gmail.com", "dataset": "TRACK_3", "zip": "tracks_1_2_3_4.zip", "shard": "0005", "round": 2},

    # --- EL (8 Shards: MELD shards 2 to 8 + Track 2 shard 3) ---
    {"member": "EL", "account": "elc0re143@gmail.com", "dataset": "MELD", "zip": "meld_raw.zip", "shard": "0002", "round": 1},
    {"member": "EL", "account": "micofeipao@gmail.com", "dataset": "MELD", "zip": "meld_raw.zip", "shard": "0003", "round": 1},
    {"member": "EL", "account": "gjrvlio.dev@gmail.com", "dataset": "MELD", "zip": "meld_raw.zip", "shard": "0004", "round": 1},
    {"member": "EL", "account": "geueljohn.rivera.lexmeet@gmail.com", "dataset": "MELD", "zip": "meld_raw.zip", "shard": "0005", "round": 1},
    {"member": "EL", "account": "gelj.riv@gmail.com", "dataset": "MELD", "zip": "meld_raw.zip", "shard": "0006", "round": 1},
    {"member": "EL", "account": "elc0re143@gmail.com", "dataset": "MELD", "zip": "meld_raw.zip", "shard": "0007", "round": 2},
    {"member": "EL", "account": "micofeipao@gmail.com", "dataset": "MELD", "zip": "meld_raw.zip", "shard": "0008", "round": 2},
    {"member": "EL", "account": "gjrvlio.dev@gmail.com", "dataset": "TRACK_2", "zip": "tracks_1_2_3_4.zip", "shard": "0003", "round": 2},

    # --- SHIKI (7 Shards: Track 1 shards 2-3, Mustard shard 2, Track 2 shards 4-5, Track 3 shards 6-7) ---
    {"member": "SHIKI", "account": "Shikina.cabral.lexmeet@gmail.com", "dataset": "TRACK_1", "zip": "tracks_1_2_3_4.zip", "shard": "0002", "round": 1},
    {"member": "SHIKI", "account": "cshikina18@gmail.com", "dataset": "TRACK_1", "zip": "tracks_1_2_3_4.zip", "shard": "0003", "round": 1},
    {"member": "SHIKI", "account": "forbaselineai@gmail.com", "dataset": "MUSTARD", "zip": "mustard.zip", "shard": "0002", "round": 1},
    {"member": "SHIKI", "account": "shikinaexoexo@gmail.com", "dataset": "TRACK_2", "zip": "tracks_1_2_3_4.zip", "shard": "0004", "round": 1},
    {"member": "SHIKI", "account": "Shikina.cabral.lexmeet@gmail.com", "dataset": "TRACK_2", "zip": "tracks_1_2_3_4.zip", "shard": "0005", "round": 2},
    {"member": "SHIKI", "account": "cshikina18@gmail.com", "dataset": "TRACK_3", "zip": "tracks_1_2_3_4.zip", "shard": "0006", "round": 2},
    {"member": "SHIKI", "account": "forbaselineai@gmail.com", "dataset": "TRACK_3", "zip": "tracks_1_2_3_4.zip", "shard": "0007", "round": 2},
]


def create_notebook(item):
    member = item["member"]
    acc = item["account"]
    dataset = item["dataset"]
    zip_file = item["zip"]
    shard = item["shard"]
    round_num = item["round"]
    
    round_tag = "" if round_num == 1 else "_ROUND2"
    clean_email = acc.split("@")[0].replace(".", "_")
    nb_filename = f"{member}_{clean_email}_{dataset.replace('-', '_')}_shard_{shard}{round_tag}.ipynb"
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# ACE-Net Universal Shard Preprocessor\n",
                f"### Assigned Member: **{member}**\n",
                f"### Target Account: `{acc}`\n",
                f"### Assigned Dataset: **{dataset}** | Shard: **`shard_{shard}`** (Round {round_num})\n",
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
                "!git checkout feat/training-and-preprocessing\n",
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
                "!pip -q install openai-whisper transformers facenet-pytorch librosa opencv-python tqdm pandas\n",
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
    print("      GENERATING 36 PRE-CONFIGURED COLAB SHARD NOTEBOOKS")
    print("=" * 70)
    
    for item in ASSIGNMENTS:
        fn = create_notebook(item)
        print(f"[{item['member']:<5}] Round {item['round']} | {item['dataset']:<10} | Shard {item['shard']} -> {fn}")
        
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Generated all {len(ASSIGNMENTS)} Colab notebooks in:")
    print(f"  -> {NOTEBOOKS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
