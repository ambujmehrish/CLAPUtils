import random
import numpy as np
from CLAPForge.core.transforms_interface import BaseWaveformTransform

class SpecAugment(BaseWaveformTransform):
    def __init__(self, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3, p=1.0):
        """
        Initialize the SpecAugment transformation.

        Parameters:
            freq_masking_max_percentage (float): Maximum percentage of frequency bins to mask.
            time_masking_max_percentage (float): Maximum percentage of time frames to mask.
            p (float): Probability of applying this augmentation.
        """
        super().__init__(p=p)
        self.freq_masking_max_percentage = freq_masking_max_percentage
        self.time_masking_max_percentage = time_masking_max_percentage

    def apply(self, input_melspec, sample_rate):
        """
        Apply SpecAugment to the input mel-spectrogram.

        Parameters:
            input_melspec (np.ndarray): Input mel-spectrogram of shape (time_frames, frequency_bins).
            sample_rate (int): Sample rate (not used in this transform, but kept for interface consistency).

        Returns:
            np.ndarray: Augmented mel-spectrogram.
        """
        augmented_melspec = input_melspec.copy()
        num_frames, num_freqs = augmented_melspec.shape

        # Frequency masking
        freq_percentage = random.uniform(0.0, self.freq_masking_max_percentage)
        num_freqs_to_mask = int(freq_percentage * num_freqs)
        if num_freqs_to_mask > 0:
            f0 = int(np.random.uniform(0.0, max(1, num_freqs - num_freqs_to_mask)))
            augmented_melspec[:, f0:(f0 + num_freqs_to_mask)] = 0

        # Time masking
        time_percentage = random.uniform(0.0, self.time_masking_max_percentage)
        num_frames_to_mask = int(time_percentage * num_frames)
        if num_frames_to_mask > 0:
            t0 = int(np.random.uniform(0.0, max(1, num_frames - num_frames_to_mask)))
            augmented_melspec[t0:(t0 + num_frames_to_mask), :] = 0

        return augmented_melspec
