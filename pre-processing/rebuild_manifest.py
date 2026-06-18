import pandas as pd
import json
import os
import numpy as np
from transformers import BertTokenizer

OUTPUT_DIR = r"D:\Baseline\outputs\meld"
SPLITS_CSV = {
    'train': r"D:\Baseline\data\meld\MELD-RAW\MELD.Raw\train\train_sent_emo.csv",
    'dev':   r"D:\Baseline\data\meld\MELD-RAW\MELD.Raw\dev_sent_emo.csv",
    'test':  r"D:\Baseline\data\meld\MELD-RAW\MELD.Raw\test_sent_emo.csv"
}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

records = []
for split, csv_path in SPLITS_CSV.items():
    df = pd.read_csv(csv_path)
    progress_path = os.path.join(OUTPUT_DIR, split, f'progress_{split}.json')
    if not os.path.exists(progress_path):
        print(f"no progress file for {split}, skipping")
        continue
    with open(progress_path) as f:
        done_ids = json.load(f)
    print(f"{split}: {len(done_ids)} entries in progress file")

    for file_id in done_ids:
        parts = file_id.split('_')
        # skip non-standard entries (e.g. "final" from final_videos_test.mp4)
        if len(parts) < 2 or not parts[0].startswith('dia') or not parts[1].startswith('utt'):
            print(f"  skipping unexpected entry: {file_id}")
            continue

        dia_id = int(parts[0].replace('dia', ''))
        utt_id = int(parts[1].replace('utt', ''))
        row = df[(df['Dialogue_ID'] == dia_id) & (df['Utterance_ID'] == utt_id)]
        emotion = row.iloc[0]['Emotion'] if not row.empty else 'unknown'

        transcript = ''
        ids_path = os.path.join(OUTPUT_DIR, split, 'text', f'{file_id}_input_ids.npy')
        if os.path.exists(ids_path):
            ids = np.load(ids_path).flatten().tolist()
            ids = [i for i in ids if i not in (0, 101, 102)]
            transcript = tokenizer.decode(ids, skip_special_tokens=True)

        frame_dir = os.path.join(OUTPUT_DIR, split, 'visual', file_id)
        n_kf = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')]) if os.path.exists(frame_dir) else 0

        records.append({
            'file_id':     file_id,
            'split':       split,
            'emotion':     emotion,
            'transcript':  transcript,
            'n_keyframes': n_kf,
            'visual_ok':   n_kf >= 4
        })

df_out = pd.DataFrame(records)
df_out.to_csv(os.path.join(OUTPUT_DIR, 'meld_manifest_rebuilt.csv'), index=False)
print(f"\nTotal records: {len(df_out)}")
print(df_out['split'].value_counts())
print(df_out['visual_ok'].value_counts())