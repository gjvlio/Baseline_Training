# Baseline Preprocessing — ACENet Replication

This repository contains the preprocessing scripts used in our replication of ACENet (Audio-Visual Cross-modal Emotion Network) as a baseline model for our undergraduate thesis on multimodal deepfake detection.

## Overview

Our thesis proposes a deepfake detection system that identifies cross-modal emotional inconsistencies across audio, visual, and text modalities. As part of the methodology, we replicate ACENet as a baseline to benchmark our proposed system against.

This repo covers the data preprocessing pipeline for the following datasets:

- **CREMA-D** — audio-visual emotion dataset used for cross-identity spliced forgery augmentation (Paradigm 2)
- **MELD** — multimodal emotion dataset sourced from Friends TV series
- **SAVEE** — audio-visual dataset for emotion recognition

## Contents

```
pre-processing/
    preprocess_meld.py      # Preprocessing pipeline for MELD dataset
    preprocess_savee.py     # Preprocessing pipeline for SAVEE dataset
    test_imports.py         # Dependency check script
requirements.txt            # Python dependencies
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/JJEEYSSEE/Baseline.git
cd Baseline
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Notes

- Dataset folders are not included in this repository due to size constraints. Download the datasets separately and place them in the appropriate directories.
- Preprocessing includes face extraction via MTCNN, audio feature extraction via Whisper, and tokenization via BERT tokenizer.
