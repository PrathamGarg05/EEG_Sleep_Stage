
"""
EEG Sleep Stage Classification

Module: dataset.py

Purpose:
Process multiple Sleep-EDF subjects and save
their extracted EEG epochs and labels.
"""

import numpy as np
from pathlib import Path

from src.preprocessing import process_subject


def process_all_subjects(paths, output_dir, subject_ids=None):
    """
    Process multiple subjects and save each subject separately.

    Parameters
    ----------
    paths : list
        List of (PSG path, hypnogram path) pairs.

    output_dir : str
        Directory where processed data will be saved.

    subject_ids : list, optional
        Subject IDs corresponding to paths.

    Returns
    -------
    saved_files : list
        Paths of saved subject datasets.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if subject_ids is None:
        subject_ids = list(range(len(paths)))

    if len(paths) != len(subject_ids):
        raise ValueError("Number of paths and subject IDs must match.")

    saved_files = []

    for (psg_path, hyp_path), subject_id in zip(paths, subject_ids):

        output_file = output_dir / f"subject_{subject_id:02d}.npz"

        # Skip if already processed
        if output_file.exists():
            print(f"Skipping Subject {subject_id}: already processed")
            saved_files.append(str(output_file))
            continue

        # Process one subject
        X, y, groups = process_subject(
            psg_path,
            hyp_path,
            subject_id
        )

        # Save processed data
        output_file = output_dir / f"subject_{subject_id:02d}.npz"

        np.savez_compressed(
            output_file,
            X=X,
            y=y,
            groups=groups
        )

        print(f"Saved: {output_file}")

        saved_files.append(str(output_file))

        # Release memory before processing next subject
        del X, y, groups

    print("\nAll subjects processed successfully!")
    print("Total subjects:", len(saved_files))

    return saved_files



def create_subject_splits(processed_dir, output_dir):
    """
    Create subject-wise train, validation and test datasets.

    No subject appears in more than one split.
    """

    from collections import Counter

    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed subject-wise split
    splits = {
        "train": [1, 2, 3, 4, 5, 8, 9, 11],
        "validation": [6, 7],
        "test": [0, 10]
    }

    # Verify no subject leakage
    train_ids = set(splits["train"])
    val_ids = set(splits["validation"])
    test_ids = set(splits["test"])

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)

    saved_files = {}

    for split_name, subject_ids in splits.items():

        X_list = []
        y_list = []
        groups_list = []

        print(f"\nCreating {split_name} dataset...")

        for sid in subject_ids:

            file_path = processed_dir / f"subject_{sid:02d}.npz"

            with np.load(file_path) as data:
                X = data["X"]
                y = data["y"]
                groups = data["groups"]

            assert np.all(groups == sid)

            X_list.append(X)
            y_list.append(y)
            groups_list.append(groups)

        # Combine subjects in this split
        X_split = np.concatenate(X_list, axis=0)
        y_split = np.concatenate(y_list)
        groups_split = np.concatenate(groups_list)

        # Save split
        output_file = output_dir / f"{split_name}.npz"

        np.savez_compressed(
            output_file,
            X=X_split,
            y=y_split,
            groups=groups_split,
            subjects=np.array(subject_ids)
        )

        print("Subjects:", subject_ids)
        print("X shape:", X_split.shape)
        print("y shape:", y_split.shape)
        print("Data type:", X_split.dtype)
        print("Stage distribution:", dict(Counter(y_split)))

        saved_files[split_name] = str(output_file)

        del X_split, y_split, groups_split

    print("\nAll splits created successfully!")

    return saved_files
