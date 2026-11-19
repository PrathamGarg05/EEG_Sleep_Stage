
"""
EEG Sleep Stage Classification

Module: preprocessing.py

Purpose:
Load Sleep-EDF PSG recordings and their
corresponding hypnogram annotations.
"""

import numpy as np
import mne


def load_recording(psg_path, hypnogram_path):
    """
    Load PSG and hypnogram EDF files.

    Parameters
    ----------
    psg_path : str
        Path to the PSG EDF file.
    hypnogram_path : str
        Path to the hypnogram EDF file.

    Returns
    -------
    raw : mne.io.Raw
        Loaded EEG recording.
    annotations : mne.Annotations
        Sleep-stage annotations.
    """

    # Load EEG recording
    raw = mne.io.read_raw_edf(
        psg_path,
        preload=True,
        verbose=False
    )

    # Load sleep-stage annotations
    annotations = mne.read_annotations(hypnogram_path)

    print("Recording loaded successfully!")
    print("Channels:", raw.ch_names)
    print("Sampling frequency:", raw.info["sfreq"], "Hz")
    print("Duration:", round(raw.times[-1] / 3600, 2), "hours")

    return raw, annotations



def preprocess_eeg(raw):
    """
    Select EEG channels and apply bandpass filtering.

    Parameters
    ----------
    raw : mne.io.Raw
        Loaded PSG recording.

    Returns
    -------
    raw_eeg : mne.io.Raw
        Filtered EEG recording containing only two EEG channels.
    """

    # Select EEG channels
    eeg_channels = ["EEG Fpz-Cz", "EEG Pz-Oz"]

    raw_eeg = raw.copy().pick(eeg_channels)

    # Apply bandpass filter
    raw_eeg.filter(
        l_freq=0.3,
        h_freq=35.0,
        picks="all",
        verbose=False
    )

    print("EEG preprocessing completed!")
    print("Selected channels:", raw_eeg.ch_names)
    print("Sampling frequency:", raw_eeg.info["sfreq"], "Hz")

    return raw_eeg



# Mapping Sleep-EDF annotation names to our 5 classes
STAGE_MAPPING = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM"
}


def extract_epochs(raw_eeg, annotations, epoch_duration=30):
    """
    Divide filtered EEG into 30-second epochs and assign labels.

    Returns
    -------
    X : numpy array
        Shape: (number_of_epochs, 2, 3000)
    y : numpy array
        Sleep-stage labels for each epoch.
    """

    sfreq = raw_eeg.info["sfreq"]

    # Number of samples in one epoch
    epoch_samples = int(epoch_duration * sfreq)

    epochs = []
    labels = []

    skipped_annotations = 0

    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description
    ):

        # Ignore unknown or unmapped sleep stages
        if description not in STAGE_MAPPING:
            continue

        # Convert time into sample indices
        start_sample = int(round(onset * sfreq))
        duration_samples = int(round(duration * sfreq))

        # Ensure annotation begins on a 30-second boundary
        if start_sample % epoch_samples != 0:
            skipped_annotations += 1
            continue

        # Number of complete epochs in this annotation
        n_epochs = duration_samples // epoch_samples

        # Do not read beyond the EEG recording
        available_epochs = (
            raw_eeg.n_times - start_sample
        ) // epoch_samples

        n_epochs = min(n_epochs, available_epochs)

        if n_epochs <= 0:
            skipped_annotations += 1
            continue

        stage = STAGE_MAPPING[description]

        for i in range(n_epochs):

            epoch_start = start_sample + i * epoch_samples
            epoch_end = epoch_start + epoch_samples

            # Extract EEG data: (2, 3000)
            epoch_data = raw_eeg.get_data(
                start=epoch_start,
                stop=epoch_end
            )

            epochs.append(epoch_data)
            labels.append(stage)

    if len(epochs) == 0:
        raise ValueError("No valid epochs extracted!")

    X = np.stack(epochs).astype(np.float32)
    y = np.array(labels)

    print("Epoch extraction completed!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Skipped annotations:", skipped_annotations)

    return X, y



def process_subject(psg_path, hypnogram_path, subject_id):
    """
    Complete preprocessing pipeline for one subject.

    Steps:
    1. Load PSG and hypnogram.
    2. Select EEG channels and filter.
    3. Extract 30-second epochs.
    4. Assign sleep-stage labels.
    5. Store subject ID for every epoch.
    """

    print(f"\nProcessing Subject {subject_id}...")

    # Load recording
    raw, annotations = load_recording(
        psg_path,
        hypnogram_path
    )

    # Select and filter EEG channels
    raw_eeg = preprocess_eeg(raw)

    # Extract epochs and labels
    X, y = extract_epochs(
        raw_eeg,
        annotations
    )

    # Identify which subject each epoch belongs to
    groups = np.full(
        len(y),
        subject_id,
        dtype=np.int16
    )

    print(f"Subject {subject_id} completed!")
    print("Data shape:", X.shape)
    print("Labels shape:", y.shape)

    return X, y, groups
