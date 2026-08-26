import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
test_dir = repo_root / "notebooks" / "colab_tests"
test_dir.mkdir(parents=True, exist_ok=True)

test_configs = [
    {"dataset": "MELD", "zip": "meld_raw.zip", "shard": "0001", "name": "TEST_MELD_smoke_test.ipynb"},
    {"dataset": "CMU-MOSEI", "zip": "cmumosei.zip", "shard": "0001", "name": "TEST_MOSEI_smoke_test.ipynb"},
    {"dataset": "MUSTARD", "zip": "mustard.zip", "shard": "0001", "name": "TEST_MUSTARD_smoke_test.ipynb"},
    {"dataset": "TRACK_1", "zip": "tracks_1_2_3_4.zip", "shard": "0001", "name": "TEST_TRACK1_smoke_test.ipynb"},
    {"dataset": "TRACK_2", "zip": "tracks_1_2_3_4.zip", "shard": "0001", "name": "TEST_TRACK2_smoke_test.ipynb"},
    {"dataset": "TRACK_3", "zip": "tracks_1_2_3_4.zip", "shard": "0001", "name": "TEST_TRACK3_smoke_test.ipynb"},
]

for cfg in test_configs:
    d = cfg["dataset"]
    z = cfg["zip"]
    s = cfg["shard"]
    fn = cfg["name"]
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# ACE-Net Quick Smoke Test (5 Samples)\n",
                f"### Target Dataset: **{d}**\n",
                f"### Purpose: Rapid pipeline & GPU validation in ~30 seconds before full run\n",
                f"### Output: `Google Drive > THESIS_MOTHERFILE > Baseline preprocessed > _TEST_RUNS > {d}`"
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
                "!pip -q install openai-whisper transformers facenet-pytorch librosa opencv-python tqdm pandas\n",
                "print('Dependencies installed successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## Step 4: Unzip Raw Dataset (`{z}`) to Local Colab SSD"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"import os, zipfile\n",
                f"\n",
                f"DRIVE_ZIP = '/content/drive/MyDrive/THESIS_MOTHERFILE/datasets/{z}'\n",
                f"LOCAL_RAW = '/content/data/raw/{d}'\n",
                f"\n",
                f"os.makedirs(LOCAL_RAW, exist_ok=True)\n",
                f"if not os.path.exists(DRIVE_ZIP):\n",
                f"    raise FileNotFoundError(f'Raw zip not found in Drive: {{DRIVE_ZIP}}')\n",
                f"\n",
                f"print(f'Unzipping {{DRIVE_ZIP}} to local SSD ({{LOCAL_RAW}})...')\n",
                f"with zipfile.ZipFile(DRIVE_ZIP, 'r') as z_file:\n",
                f"    z_file.extractall(LOCAL_RAW)\n",
                f"print('Unzip complete! Local files ready.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## Step 5: Execute 5-Sample Smoke Test on `{d}`"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"!python scripts/preprocess/run_shard.py \\\n",
                f"    --account 'test_runner@gmail.com' \\\n",
                f"    --dataset '{d}' \\\n",
                f"    --shard '{s}' \\\n",
                f"    --raw_dir '/content/data/raw/{d}' \\\n",
                f"    --drive_root '/content/drive/MyDrive/THESIS_MOTHERFILE/_TEST_RUNS' \\\n",
                f"    --limit 5 \\\n",
                f"    --device cuda"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 6: Verify Generated Feature Tensors (.npy & .jpg)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"import glob, numpy as np\n",
                f"from pathlib import Path\n",
                f"\n",
                f"test_out = Path('/content/drive/MyDrive/THESIS_MOTHERFILE/_TEST_RUNS/Baseline preprocessed/{d}/shards/shard_{s}')\n",
                f"mels = list(test_out.glob('audio/*.npy'))\n",
                f"texts = list(test_out.glob('text/*_input_ids.npy'))\n",
                f"vis = list(test_out.glob('visual/*'))\n",
                f"\n",
                f"print('=' * 60)\n",
                f"print('SMOKE TEST AUDIT: {d}')\n",
                f"print(f'Audio Melspecs Found : {{len(mels)}} / 5')\n",
                f"print(f'Text Token Sets Found: {{len(texts)}} / 5')\n",
                f"print(f'Visual Folders Found : {{len(vis)}} / 5')\n",
                f"if len(mels) > 0:\n",
                f"    sample_mel = np.load(mels[0])\n",
                f"    print(f'Sample Mel Shape (Target: [80, T]): {{sample_mel.shape}}')\n",
                f"print('=' * 60)\n",
                f"if len(mels) >= 1 and len(texts) >= 1:\n",
                f"    print('[SUCCESS] Pipeline verified working for {d}!')\n",
                f"else:\n",
                f"    print('[NOTICE] Check logs above if files were skipped.')"
            ]
        }
    ]
    
    nb = {
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
    
    with open(test_dir / fn, "w", encoding="utf-8") as out_nb:
        json.dump(nb, out_nb, indent=2)
    print(f"Generated Test Notebook: {fn}")

print("All 6 Smoke Test Notebooks generated successfully in notebooks/colab_tests/")
