"""Release contract for the adopted graph-semantics v4.1 main line.

The repository release is intentionally distinct from component serialization schemas:
L3 LoRA and its dictionary use schema 3, while L2 WeightNet uses schema 4.
"""
import hashlib
import json
import os

RELEASE_VERSION = "4.1"
RELEASE_TAG = "v4.1"
RELEASE_SLUG = "v4_1"
RELEASE_DATE = "2026-07-21"
MANIFEST_SCHEMA_VERSION = 1

L3_CHECKPOINT_SCHEMA_VERSION = 3
L3_DICTIONARY_SCHEMA_VERSION = 3
L2_CHECKPOINT_SCHEMA_VERSION = 4
SOLVER_STEPS = 120
TRUNCATION_STEPS = 60

# The 2.23 GB dictionary remains at its immutable, L3-SHA-bound component path. The final L2
# checkpoint and Task 1 records receive release-level names as inexpensive byte copies; the
# Task 2 record is produced by the subsequent final release run.
L3_CHECKPOINT_NAME = "l3_lora_rel.pt"
L3_DICTIONARY_NAME = "concept_bank_l3_rel.npz"
L2_CHECKPOINT_TEMPLATE = "l2_{arm}_v4_1.pt"
L2_TRAINLOG_TEMPLATE = "l2_{arm}_v4_1_trainlog.json"
TARGETED_RESULTS_NAME = "v4_1_targeted_task1.json"
ALL13_RESULTS_NAME = "v4_1_task1_all13_api_free.json"
ALL13_TASK2_RESULTS_NAME = "v4_1_task2_all13_api_free.json"
MANIFEST_NAME = "release_v4_1.json"


def l2_checkpoint_name(arm="mlp"):
    return L2_CHECKPOINT_TEMPLATE.format(arm=arm)


def l2_trainlog_name(arm="mlp"):
    return L2_TRAINLOG_TEMPLATE.format(arm=arm)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_manifest(repo_root):
    """Load the tracked release sidecar and reject a different/legacy release."""
    path = os.path.join(repo_root, MANIFEST_NAME)
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load v4.1 release manifest {path!r}: {exc}") from exc
    if manifest.get("manifest_schema") != MANIFEST_SCHEMA_VERSION \
            or manifest.get("release_version") != RELEASE_VERSION:
        raise RuntimeError(
            f"Incompatible release manifest {path!r}: got schema/release "
            f"{manifest.get('manifest_schema')!r}/{manifest.get('release_version')!r}, "
            f"expected {MANIFEST_SCHEMA_VERSION}/{RELEASE_VERSION!r}.")
    return manifest


def artifact(manifest, role):
    matches = [entry for entry in manifest.get("artifacts", [])
               if entry.get("role") == role]
    if len(matches) != 1:
        raise RuntimeError(
            f"Release manifest must contain exactly one {role!r} artifact, got {len(matches)}.")
    return matches[0]


def verify_artifact(manifest, repo_root, role, actual_path, verify_sha256=True):
    """Verify that a runtime file is the artifact frozen by the v4.1 sidecar."""
    entry = artifact(manifest, role)
    expected_path = os.path.normcase(os.path.abspath(os.path.join(repo_root, entry["path"])))
    actual_path = os.path.normcase(os.path.abspath(actual_path))
    if actual_path != expected_path:
        raise RuntimeError(
            f"{role} path {actual_path!r} is not the v4.1 artifact {expected_path!r}. "
            "Explicit experimental overrides are not identified as the formal release.")
    if not os.path.isfile(actual_path):
        raise RuntimeError(f"Missing v4.1 {role} artifact: {actual_path!r}")
    actual_size = os.path.getsize(actual_path)
    if actual_size != entry.get("size_bytes"):
        raise RuntimeError(
            f"v4.1 {role} size mismatch: got {actual_size}, expected "
            f"{entry.get('size_bytes')} bytes.")
    if verify_sha256:
        actual_sha = sha256_file(actual_path)
        if actual_sha != entry.get("sha256"):
            raise RuntimeError(
                f"v4.1 {role} SHA-256 mismatch: got {actual_sha}, expected "
                f"{entry.get('sha256')}.")
    return entry
