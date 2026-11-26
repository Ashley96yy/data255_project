"""Find exact and near-duplicate images within and across train/val/test.

Proposal Section C promises a check for duplicate or closely related images.
Each image gets an MD5 of its file bytes (exact copies) and a 64-bit perceptual
hash (pHash, via ``imagehash``). All 7,349 x 7,349 pairs are compared with a
vectorised Hamming distance; pairs at or below ``--threshold`` bits are saved.

Usage (from the repository root, needs the downloaded dataset):

    python -m scripts.check_duplicates --data-root data
    # writes results/data_audit/duplicate_pairs.csv and duplicate_summary.json

Pairs are candidates, not confirmed duplicates: review the CSV (and, if useful,
``--save-grid``) before deciding anything. Cross-split pairs (train-val,
train-test, val-test) matter most because they can leak information into model
selection or the final test score. The split itself is not changed here.
"""

import argparse
from collections import Counter
import hashlib
from itertools import combinations_with_replacement
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import OxfordIIITPet

from src.training import save_json


def image_records(data_root, split_csv, download=False):
    """One row per image: id, official split, project split, file path."""
    split_df = pd.read_csv(split_csv).set_index("dataset_index")
    rows = []
    for official in ("trainval", "test"):
        dataset = OxfordIIITPet(
            root=str(data_root), split=official, target_types="segmentation",
            download=download,
        )
        for index, path in enumerate(dataset._images):
            split = "test" if official == "test" else split_df.loc[index, "split"]
            rows.append({
                "image_id": f"{official}:{index}",
                "split": split,
                "file": Path(path).name,
                "path": str(path),
            })
    return pd.DataFrame(rows)


def hash_images(paths):
    phashes, md5s = [], []
    for path in paths:
        data = Path(path).read_bytes()
        md5s.append(hashlib.md5(data).hexdigest())
        with Image.open(path) as image:
            bits = imagehash.phash(image.convert("RGB")).hash.flatten()
        phashes.append(int("".join("1" if bit else "0" for bit in bits), 2))
    return np.array(phashes, dtype=np.uint64), md5s


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(values):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    return _POPCOUNT[values.view(np.uint8)].reshape(*values.shape, 8).sum(-1)


def close_pairs(hashes, threshold, chunk=512):
    """Return (i, j, distance) for i < j with Hamming distance <= threshold."""
    hashes = np.asarray(hashes, dtype=np.uint64)
    found = []
    for start in range(0, len(hashes), chunk):
        block = hashes[start:start + chunk]
        distances = popcount64(block[:, None] ^ hashes[None, :])
        rows, cols = np.nonzero(distances <= threshold)
        rows = rows + start
        keep = rows < cols
        found.extend(zip(rows[keep].tolist(), cols[keep].tolist(),
                         distances[rows[keep] - start, cols[keep]].tolist()))
    return found


def build_pairs(records, hashes, md5s, threshold):
    rows = []
    for i, j, distance in close_pairs(hashes, threshold):
        a, b = records.iloc[i], records.iloc[j]
        rows.append({
            "image_id_a": a.image_id, "split_a": a.split, "file_a": a.file,
            "image_id_b": b.image_id, "split_b": b.split, "file_b": b.file,
            "split_pair": "-".join(sorted((a.split, b.split))),
            "phash_distance": int(distance),
            "exact_file_match": md5s[i] == md5s[j],
        })
    columns = ["image_id_a", "split_a", "file_a", "image_id_b", "split_b", "file_b",
               "split_pair", "phash_distance", "exact_file_match"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["phash_distance", "split_pair"], kind="stable"
    )


def summarize(pairs, records, threshold):
    splits = sorted(records["split"].unique())
    by_pair = Counter(pairs["split_pair"])
    return {
        "method": "64-bit pHash (imagehash) + MD5 of file bytes",
        "phash_threshold_bits": threshold,
        "images_by_split": records["split"].value_counts().sort_index().to_dict(),
        "candidate_pairs": len(pairs),
        "exact_file_matches": int(pairs["exact_file_match"].sum()),
        "pairs_by_split_pair": {
            "-".join(p): int(by_pair.get("-".join(p), 0))
            for p in combinations_with_replacement(splits, 2)
        },
        "cross_split_pairs": int((pairs["split_a"] != pairs["split_b"]).sum()),
        "note": "Candidates need visual review; the split is not modified by this script.",
    }


def save_grid(pairs, records, output, limit=24):
    import matplotlib.pyplot as plt

    shown = pairs[pairs["split_a"] != pairs["split_b"]].head(limit)
    if shown.empty:
        shown = pairs.head(limit)
    if shown.empty:
        return
    paths = records.set_index("image_id")["path"]
    fig, axes = plt.subplots(len(shown), 2, figsize=(5, 2.4 * len(shown)), squeeze=False)
    for row, pair in zip(axes, shown.itertuples()):
        for ax, image_id, split in ((row[0], pair.image_id_a, pair.split_a),
                                    (row[1], pair.image_id_b, pair.split_b)):
            with Image.open(paths[image_id]) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"{split} {image_id} (d={pair.phash_distance})", fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-csv", type=Path, default=Path("configs/train_val_split_clean.csv"))
    parser.add_argument("--threshold", type=int, default=6,
                        help="Max pHash Hamming distance (bits of 64) to report.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/data_audit"))
    parser.add_argument("--save-grid", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    records = image_records(args.data_root, args.split_csv, args.download)
    print(f"Hashing {len(records)} images ...", flush=True)
    hashes, md5s = hash_images(records["path"])
    pairs = build_pairs(records, hashes, md5s, args.threshold)
    summary = summarize(pairs, records, args.threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "duplicate_pairs.csv", index=False)
    save_json(args.output_dir / "duplicate_summary.json", summary)
    if args.save_grid:
        save_grid(pairs, records, args.output_dir / "duplicate_examples.png")

    print(f"Candidate pairs (pHash distance <= {args.threshold}): {len(pairs)}")
    print(f"Exact file matches: {summary['exact_file_matches']}")
    for name, count in summary["pairs_by_split_pair"].items():
        print(f"  {name:12s} {count}")
    print(f"Saved: {args.output_dir / 'duplicate_pairs.csv'}")


if __name__ == "__main__":
    main()
