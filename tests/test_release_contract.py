import json
import os
import tempfile
import unittest

import torch

from pipeline_L3_v1 import lora
from pipeline_v4 import l2_modules as LM
from pipeline_v4 import release


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ReleaseContractTests(unittest.TestCase):
    def test_repository_release_is_v4_1(self):
        with open(os.path.join(ROOT, "VERSION"), encoding="utf-8") as handle:
            self.assertEqual(handle.read().strip(), "4.1")
        self.assertEqual(release.RELEASE_VERSION, "4.1")
        self.assertEqual(release.RELEASE_TAG, "v4.1")
        self.assertEqual(release.ALL13_TASK2_RESULTS_NAME,
                         "v4_1_task2_all13_api_free.json")

    def test_release_and_component_schemas_are_distinct(self):
        self.assertEqual(LM.CHECKPOINT_VERSION, release.L2_CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(lora.CHECKPOINT_VERSION, release.L3_CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(lora.DICTIONARY_VERSION, release.L3_DICTIONARY_SCHEMA_VERSION)
        self.assertIsInstance(LM.CHECKPOINT_VERSION, int)
        self.assertIsInstance(lora.CHECKPOINT_VERSION, int)

    def test_l2_schema_v4_round_trip_preserves_state_and_release_metadata(self):
        module = LM.WeightNet()
        with torch.no_grad():
            module.net[-1].bias.copy_(torch.arange(len(LM.TERMS)) / 100.0)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "l2.pt")
            LM.save(module, path, "mlp", metadata={"release_version": "4.1"})
            loaded = LM.load(path)
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key])
            payload = torch.load(path, map_location="cpu")
            self.assertEqual(payload["metadata"]["release_version"], "4.1")

    def test_release_manifest_contract_without_hashing_large_artifacts(self):
        manifest = release.load_manifest(ROOT)
        self.assertEqual(manifest["component_schemas"], {
            "l3_checkpoint": 3, "l3_dictionary": 3, "l2_checkpoint": 4})
        self.assertEqual(manifest["training"]["solver_steps"], 120)
        self.assertEqual(manifest["training"]["truncation_steps"], 60)
        expected_paths = {
            "l3_checkpoint": "outputs/l3_lora_rel.pt",
            "l3_dictionary": "outputs/concept_bank_l3_rel.npz",
            "l2_checkpoint": "outputs/l2_mlp_v4_1.pt",
            "task2_all13_api_free": "outputs/v4_1_task2_all13_api_free.json",
        }
        for role, path in expected_paths.items():
            entry = release.artifact(manifest, role)
            self.assertEqual(entry["path"], path)
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
        task2 = manifest["evaluation"]["task2"]
        self.assertEqual(task2["dataset_count"], 13)
        self.assertEqual(task2["folds"], list(range(5)))
        self.assertEqual(task2["record_count"], task2["latent_count"] * len(task2["folds"]))
        self.assertFalse(task2["judge_called"])
        self.assertFalse(task2["api_free_accuracy_metric_available"])
        task2_artifact = release.artifact(manifest, "task2_all13_api_free")
        self.assertEqual(task2_artifact["size_bytes"], 169146)
        self.assertEqual(
            task2_artifact["sha256"],
            "e0b1f26476ce5da7fc86949531aa1f1b9c6deadcd126a281c9ffa794de2df596")
        self.assertEqual(task2_artifact["record_count"], 450)
        source = manifest["source"]
        self.assertEqual(source["release_branch"], "xuran_v4")
        self.assertEqual(
            source["initial_release_commit"],
            "83879b4714004bdb05dd2eaa6ce63f326791f590")
        self.assertTrue(source["commit_created"])
        self.assertFalse(source["git_tag_created"])
        self.assertEqual(source["push_target"], "origin/xuran_v4")

    def test_wrong_release_manifest_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, release.MANIFEST_NAME)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"manifest_schema": 1, "release_version": "4.0"}, handle)
            with self.assertRaisesRegex(RuntimeError, "Incompatible release manifest"):
                release.load_manifest(directory)


if __name__ == "__main__":
    unittest.main()
