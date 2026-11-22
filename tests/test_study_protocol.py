"""Data provenance checks without accessing real dataset arrays."""

import hashlib

from src.study_protocol import SMD_MACHINE_ID, dataset_provenance


def test_provenance_fingerprints_only_the_selected_machine_without_modifying_files(tmp_path):
    folder = tmp_path / "SMD"
    folder.mkdir()
    contents = {"train": b"normal training", "test": b"held-out observations", "labels": b"ground truth"}
    for split, content in contents.items():
        (folder / f"{SMD_MACHINE_ID}_{split}.npy").write_bytes(content)
    other = folder / "machine-2-1_train.npy"
    other.write_bytes(b"another machine")

    first = dataset_provenance("SMD", tmp_path)
    assert first["machine_id"] == SMD_MACHINE_ID
    assert first["data_fingerprints"] == {
        split: hashlib.sha256(content).hexdigest() for split, content in contents.items()
    }
    other.write_bytes(b"changed unrelated machine")
    assert dataset_provenance("SMD", tmp_path) == first
    test_path = folder / f"{SMD_MACHINE_ID}_test.npy"
    test_path.write_bytes(b"different preprocessing")
    changed = dataset_provenance("SMD", tmp_path)
    assert changed["data_fingerprints"]["test"] != first["data_fingerprints"]["test"]
    assert changed["data_fingerprints"]["train"] == first["data_fingerprints"]["train"]
    assert test_path.read_bytes() == b"different preprocessing"


def test_other_datasets_do_not_read_smd_files(tmp_path):
    assert dataset_provenance("synthetic", tmp_path) == {}
