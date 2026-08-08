from pathlib import Path
import tempfile
import unittest

import numpy as np

from v6.baselines.clip_dissect_bank import (
    BANK_VERSION,
    DOMAIN_ANCHORS,
    build_domain_bank,
)


class TextDissectBankTests(unittest.TestCase):
    def test_domain_selection_is_deterministic_and_filters_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            output = root / "selected.npz"
            names = np.asarray(["personality", "123 bad", "social behavior", "tool"], object)
            emb = np.asarray([[1, 0], [1, 0], [0.8, 0.2], [-1, 0]], np.float32)
            np.savez(source, emb=emb, names=names)

            def encoder(texts):
                self.assertEqual(tuple(texts), DOMAIN_ANCHORS)
                return np.tile(np.asarray([[1.0, 0.0]]), (len(texts), 1))

            first = build_domain_bank(source, output, encoder, size=2, chunk_size=2)
            second = build_domain_bank(source, output, encoder, size=2, chunk_size=2)
            self.assertEqual(first, second)
            with np.load(output, allow_pickle=True) as payload:
                self.assertEqual(str(payload["selection_version"].item()), BANK_VERSION)
                self.assertEqual([str(x) for x in payload["names"]], [
                    "personality", "social behavior"
                ])
                self.assertNotIn("123 bad", [str(x) for x in payload["names"]])


if __name__ == "__main__":
    unittest.main()
