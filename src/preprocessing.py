
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
