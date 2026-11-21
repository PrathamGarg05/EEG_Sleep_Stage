
"""
EEG Sleep Stage Classification

Module: dataset.py

Purpose:
Process all Sleep-EDF recordings and save each
recording's extracted EEG epochs separately.
"""

import gc
import re
import numpy as np

from pathlib import Path
from src.preprocessing import process_subject


def get_recording_info(psg_path):
    """
    Extract subject ID and recording number from
    a Sleep-EDF PSG filename.

    Example:
    SC4001E0-PSG.edf -> subject 0, recording 1
    SC4012E0-PSG.edf -> subject 1, recording 2
    """

    filename = Path(psg_path).name

    match = re.fullmatch(
        r"SC4(\d{2})(\d)E0-PSG\.edf",
        filename
    )

    if match is None:
        raise ValueError(
            f"Unexpected PSG filename format: {filename}"
        )

    subject_id = int(match.group(1))
    recording_id = int(match.group(2))

    return subject_id, recording_id


def is_valid_processed_file(file_path, subject_id, recording_id):
    """
    Check whether an existing processed file is valid
    and belongs to the expected recording.
    """

    try:
        with np.load(file_path, allow_pickle=False) as data:

            X = data["X"]
            y = data["y"]
            groups = data["groups"]

            saved_subject = int(data["subject_id"].item())
            saved_recording = int(data["recording_id"].item())

            if saved_subject != subject_id:
                return False

            if saved_recording != recording_id:
                return False

            if X.ndim != 3 or X.shape[1:] != (2, 3000):
                return False

            if len(X) == 0 or len(X) != len(y):
                return False

            if len(X) != len(groups):
                return False

            if not np.all(groups == subject_id):
                return False

            if not np.isfinite(X).all():
                return False

            return True

    except Exception:
        return False


def process_all_recordings(paths, output_dir):
    """
    Process all PSG-hypnogram pairs.

    Each recording is saved separately.

    Existing valid files are skipped, allowing
    processing to resume after a Colab disconnect.

    Parameters
    ----------
    paths : list
        List of (PSG path, hypnogram path) pairs.

    output_dir : str
        Directory where processed files are saved.

    Returns
    -------
    saved_files : list
        Paths of successfully processed recordings.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not paths:
        raise ValueError("No recording paths provided.")

    saved_files = []
    seen_recordings = set()

    print("Total recordings received:", len(paths))

    for index, (psg_path, hyp_path) in enumerate(paths, start=1):

        subject_id, recording_id = get_recording_info(psg_path)

        recording_key = (subject_id, recording_id)

        if recording_key in seen_recordings:
            raise ValueError(
                f"Duplicate recording detected: {recording_key}"
            )

        seen_recordings.add(recording_key)

        output_file = output_dir / (
            f"subject_{subject_id:02d}_"
            f"recording_{recording_id}.npz"
        )

        print("\n" + "=" * 60)
        print(
            f"[{index}/{len(paths)}] "
            f"Subject {subject_id}, Recording {recording_id}"
        )

        # Skip an already completed recording
        if output_file.exists():

            if is_valid_processed_file(
                output_file,
                subject_id,
                recording_id
            ):
                print("Already processed and validated. Skipping.")
                saved_files.append(str(output_file))
                continue

            print("Existing file is invalid. Reprocessing.")

            output_file.unlink()

        # Process one recording
        X, y, groups = process_subject(
            psg_path=psg_path,
            hypnogram_path=hyp_path,
            subject_id=subject_id
        )

        # Validate extracted data
        if X.ndim != 3 or X.shape[1:] != (2, 3000):
            raise ValueError(f"Unexpected X shape: {X.shape}")

        if len(X) == 0 or len(X) != len(y):
            raise ValueError("Invalid epoch or label count.")

        if len(X) != len(groups):
            raise ValueError("Groups and epochs have different lengths.")

        if not np.isfinite(X).all():
            raise ValueError("Non-finite values found in EEG data.")

        if not np.all(groups == subject_id):
            raise ValueError("Incorrect subject IDs in groups.")

        # Save to a temporary file first.
        # This prevents an incomplete final file if saving is interrupted.
        temp_file = output_file.with_name(
            output_file.stem + ".tmp"
        )

        try:
            with open(temp_file, "wb") as f:
                np.savez_compressed(
                    f,
                    X=X,
                    y=y,
                    groups=groups,
                    subject_id=np.array(subject_id),
                    recording_id=np.array(recording_id),
                    psg_name=np.array(Path(psg_path).name),
                    hypnogram_name=np.array(Path(hyp_path).name)
                )

            # Verify the saved file before finalizing it
            if not is_valid_processed_file(
                temp_file,
                subject_id,
                recording_id
            ):
                raise ValueError("Saved file failed validation.")

            temp_file.replace(output_file)

        finally:
            if temp_file.exists():
                temp_file.unlink()

        print("Saved:", output_file)
        print("Epochs:", len(X))
        print("Shape:", X.shape)

        saved_files.append(str(output_file))

        # Release memory before the next recording
        del X, y, groups
        gc.collect()

    print("\n" + "=" * 60)
    print("Processing completed!")
    print("Total recordings:", len(saved_files))
    print("Output directory:", output_dir)

    return saved_files
