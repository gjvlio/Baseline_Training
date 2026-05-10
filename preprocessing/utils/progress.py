import json
import os
import pandas as pd


def load_done(progress_file: str) -> set:
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return set(json.load(f))
    return set()


def mark_done(file_id: str, done_set: set, progress_file: str) -> None:
    done_set.add(file_id)
    with open(progress_file, "w") as f:
        json.dump(list(done_set), f)


def save_manifest(records: list, out_path: str) -> None:
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"manifest -> {out_path}  ({len(records)} records)")
