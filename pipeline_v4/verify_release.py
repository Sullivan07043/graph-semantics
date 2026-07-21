"""Verify the frozen v4.1 sidecar, component schemas, and artifact binding chain."""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pipeline_L3_v1 import lora
from pipeline_v4 import l2_modules as LM
from pipeline_v4 import release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-dictionary-sha", action="store_true",
        help="also hash the 2.23 GB concept bank (size and internal binding are always checked)")
    args = parser.parse_args()

    manifest = release.load_manifest(HERE)
    for entry in manifest["artifacts"]:
        role = entry["role"]
        path = os.path.join(HERE, entry["path"])
        release.verify_artifact(
            manifest, HERE, role, path,
            verify_sha256=role != "l3_dictionary" or args.full_dictionary_sha)
        source = entry.get("legacy_source_path")
        if source:
            source_path = os.path.join(HERE, source)
            if release.sha256_file(source_path) != entry["sha256"]:
                raise RuntimeError(
                    f"Formal {role} is not byte-identical to legacy source {source!r}.")

    l3_entry = release.artifact(manifest, "l3_checkpoint")
    l3_path = os.path.join(HERE, l3_entry["path"])
    l3_sha = release.sha256_file(l3_path)
    lora._validated_checkpoint(l3_path)

    dictionary_entry = release.artifact(manifest, "l3_dictionary")
    dictionary_path = os.path.join(HERE, dictionary_entry["path"])
    dictionary = np.load(dictionary_path, allow_pickle=True)
    try:
        if str(dictionary["format"]) != lora.DICTIONARY_FORMAT \
                or int(dictionary["version"]) != lora.DICTIONARY_VERSION:
            raise RuntimeError("v4.1 dictionary component schema does not match the code.")
        if str(dictionary["lora_checkpoint_sha256"]) != l3_sha:
            raise RuntimeError("v4.1 dictionary is bound to a different L3 checkpoint.")
        if tuple(dictionary["emb"].shape) != tuple(dictionary_entry["shape"]):
            raise RuntimeError("v4.1 dictionary embedding shape does not match the manifest.")
    finally:
        dictionary.close()

    l2_entry = release.artifact(manifest, "l2_checkpoint")
    l2_path = os.path.join(HERE, l2_entry["path"])
    LM.load(l2_path, expected_l3_sha256=l3_sha)
    payload = torch.load(l2_path, map_location="cpu")
    if payload.get("version") != release.L2_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("v4.1 L2 component schema does not match the release contract.")
    legacy_l2 = torch.load(
        os.path.join(HERE, l2_entry["legacy_source_path"]), map_location="cpu")
    if payload["state"].keys() != legacy_l2["state"].keys() \
            or not all(torch.equal(value, legacy_l2["state"][key])
                       for key, value in payload["state"].items()):
        raise RuntimeError("v4.1 L2 state tensors differ from the validated final candidate.")

    print(f"v{release.RELEASE_VERSION} verified")
    print(f"  L3 sha256: {l3_sha}")
    print(f"  L2 sha256: {l2_entry['sha256']}")
    print(f"  dictionary sha256: {dictionary_entry['sha256']}"
          f" ({'verified' if args.full_dictionary_sha else 'manifest only; size/binding verified'})")
    print("  formal copies: byte-identical; L2 state tensors equal")
    print("  component binding: dictionary -> L3 <- L2")


if __name__ == "__main__":
    main()
