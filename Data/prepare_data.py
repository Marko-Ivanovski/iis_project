"""Build frozen train/val/test sets for the Hateful Meme fusion comparison.

Source: limjiayi/hateful_memes_expanded (the ONLY dataset used). The repo ships
text + labels in per-split jsonl files and the images as individual LFS pngs in
img/. We sample from the challenge's own disjoint splits, so there is no leakage
between train, val, and test, then download only the images we actually sampled.

Output layout (written to OUT_DIR):

    full-samples/              100-samples/      (100 total, for a quick Drive test)
      train.jsonl  (500)         train.jsonl  (70)
      val.jsonl    (200)         val.jsonl    (10)
      test.jsonl   (1000)        test.jsonl   (20)
      images/<id>.png            images/<id>.png

Each jsonl row keeps the project's existing shape so the notebooks' loaders work
unchanged: {"id", "img": "images/<id>.png", "label", "text"}. Load an image with
Image.open(f"{split_dir}/{row['img']}").

Run in a Colab cell (writes to Drive, keeps both notebooks consistent):

    from google.colab import drive
    drive.mount('/content/drive')
    !pip -q install huggingface_hub
    !python prepare_data.py --out "/content/drive/MyDrive/IIS Project/Data"

Or locally (slower, then upload the folders to the same Drive path):

    python prepare_data.py --out ./Data
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO = "limjiayi/hateful_memes_expanded"
SEED = 42

# our set name -> source jsonl in the repo (the challenge's canonical splits)
SOURCE = {"train": "train.jsonl", "val": "dev_seen.jsonl", "test": "test_seen.jsonl"}
FULL_SIZES = {"train": 500, "val": 200, "test": 1000}
SMALL_SIZES = {"train": 70, "val": 10, "test": 20}  # 100 total, quick Drive test

FULL_DIR = "full-samples"
SMALL_DIR = "100-samples"


@dataclass
class Row:
    id: str
    img: str  # "img/<id>.png" as stored in the source
    label: int
    text: str


def load_split(name: str) -> list[Row]:
    """Download and parse one source jsonl from the dataset repo."""
    path = hf_hub_download(REPO, SOURCE[name], repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return [Row(**json.loads(line)) for line in f]


def stratified_sample(rows: list[Row], n: int, seed: int) -> list[Row]:
    """Sample n rows keeping the original hateful/not ratio. Caps at len(rows)."""
    n = min(n, len(rows))
    rng = random.Random(seed)
    pos = [r for r in rows if r.label == 1]
    neg = [r for r in rows if r.label == 0]
    n_pos = round(n * len(pos) / len(rows))
    n_pos = min(n_pos, len(pos))
    n_neg = min(n - n_pos, len(neg))
    picked = rng.sample(pos, n_pos) + rng.sample(neg, n_neg)
    rng.shuffle(picked)
    return picked


def write_set(rows: list[Row], split_dir: str, src_image_dir: str | None) -> None:
    """Write <split>.jsonl and copy/download its images into split_dir/images/.

    If src_image_dir is given, images are copied from there (used for the smoke
    set, whose ids are a subset of full). Otherwise each image is pulled from the
    dataset repo once and cached by the hub.
    """
    images_dir = os.path.join(split_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    jsonl_path = split_dir + ".jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as out:
        for r in tqdm(rows, desc=os.path.basename(split_dir), leave=False):
            fname = os.path.basename(r.img)  # ids can be non-numeric, keep as-is
            dst = os.path.join(images_dir, fname)
            if not os.path.exists(dst):
                if src_image_dir:
                    shutil.copyfile(os.path.join(src_image_dir, fname), dst)
                else:
                    cached = hf_hub_download(REPO, r.img, repo_type="dataset")
                    shutil.copyfile(cached, dst)
            out.write(json.dumps({
                "id": r.id,
                "img": f"images/{fname}",
                "label": r.label,
                "text": r.text,
            }) + "\n")


def summarize(name: str, rows: list[Row]) -> None:
    pos = sum(r.label == 1 for r in rows)
    print(f"  {name:5s}: {len(rows):4d} rows  |  {pos} hateful / {len(rows) - pos} not")


def build(out_dir: str) -> None:
    pools = {name: load_split(name) for name in SOURCE}

    # Dataset quirk: ~100 dev_seen ids are also present in train.jsonl (verified
    # against this release). Treat val and test as authoritative and strip any of
    # their ids out of the train pool before sampling, so training never sees an
    # eval meme. Log the count rather than dropping silently.
    val_test_ids = {r.id for r in pools["val"]} | {r.id for r in pools["test"]}
    before = len(pools["train"])
    pools["train"] = [r for r in pools["train"] if r.id not in val_test_ids]
    dropped = before - len(pools["train"])
    if dropped:
        print(f"Cleaned {dropped} train memes that also appear in val/test (dataset overlap).")
    test_ids = {r.id for r in pools["test"]}
    pools["val"] = [r for r in pools["val"] if r.id not in test_ids]

    # 1. sample the three now-disjoint sets from the real dataset
    full: dict[str, list[Row]] = {
        name: stratified_sample(pools[name], FULL_SIZES[name], SEED) for name in SOURCE
    }

    # 2. hard guarantee: no id shared across train/val/test
    ids = {name: {r.id for r in rows} for name, rows in full.items()}
    assert not (ids["train"] & ids["val"]), "leak: train/val overlap"
    assert not (ids["train"] & ids["test"]), "leak: train/test overlap"
    assert not (ids["val"] & ids["test"]), "leak: val/test overlap"

    print(f"{FULL_DIR}:")
    for name in SOURCE:
        summarize(name, full[name])

    for name, rows in full.items():
        write_set(rows, os.path.join(out_dir, FULL_DIR, name), src_image_dir=None)

    # 3. 100-sample set: first N of each full split, reusing already-downloaded images
    print(f"{SMALL_DIR}:")
    for name, rows in full.items():
        small = rows[: SMALL_SIZES[name]]
        summarize(name, small)
        write_set(
            small,
            os.path.join(out_dir, SMALL_DIR, name),
            src_image_dir=os.path.join(out_dir, FULL_DIR, name, "images"),
        )

    print(f"\nDone. Data written under: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output directory (Drive path or local)")
    args = parser.parse_args()
    build(args.out)
