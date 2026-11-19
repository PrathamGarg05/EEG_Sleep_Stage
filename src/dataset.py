
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
