"""Check duplicate detection on synthetic images (no dataset download)."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from scripts.check_duplicates import build_pairs, close_pairs, hash_images, summarize


class DuplicateTest(unittest.TestCase):
    def test_close_pairs_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        hashes = rng.integers(0, 2 ** 63, size=300, dtype=np.uint64)
        hashes[10] = hashes[3] ^ np.uint64(0b101)  # distance 2
        found = close_pairs(hashes, threshold=2, chunk=64)
        brute = [(i, j, bin(int(hashes[i]) ^ int(hashes[j])).count("1"))
                 for i in range(300) for j in range(i + 1, 300)
                 if bin(int(hashes[i]) ^ int(hashes[j])).count("1") <= 2]
        self.assertEqual(sorted(found), sorted(brute))
        self.assertIn((3, 10, 2), found)

    def test_copy_and_resized_copy_are_flagged_across_splits(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).resize((256, 256))
            base = base.filter(ImageFilter.GaussianBlur(4))
            base.save(root / "a.jpg")
            (root / "a_copy.jpg").write_bytes((root / "a.jpg").read_bytes())
            base.resize((200, 180)).save(root / "a_small.jpg")
            Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(root / "other.jpg")
            records = pd.DataFrame({
                "image_id": ["trainval:0", "test:0", "trainval:1", "trainval:2"],
                "split": ["train", "test", "val", "train"],
                "file": ["a.jpg", "a_copy.jpg", "a_small.jpg", "other.jpg"],
                "path": [str(root / n) for n in ("a.jpg", "a_copy.jpg", "a_small.jpg", "other.jpg")],
            })
            hashes, md5s = hash_images(records["path"])
            pairs = build_pairs(records, hashes, md5s, threshold=6)
            summary = summarize(pairs, records, 6)
        found = {frozenset((r.file_a, r.file_b)) for r in pairs.itertuples()}
        self.assertIn(frozenset(("a.jpg", "a_copy.jpg")), found)
        self.assertIn(frozenset(("a.jpg", "a_small.jpg")), found)
        self.assertFalse(any("other.jpg" in pair for pair in found))
        self.assertEqual(summary["exact_file_matches"], 1)
        self.assertEqual(summary["pairs_by_split_pair"]["test-train"], 1)
        self.assertEqual(summary["cross_split_pairs"], 3)


if __name__ == "__main__":
    unittest.main()
