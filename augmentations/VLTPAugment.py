"""
VTLPAugment

This augmentation applies Vocal Tract Length Perturbation (VTLP) to a mel-spectrogram.
VTLP warps the frequency axis of a mel-spectrogram to simulate variations in vocal tract
lengths among speakers. This can improve model robustness by introducing speaker variability.

The transformation uses the following mapping for each frequency bin f:
    f_warp = f_max * (f / f_max)^alpha,
where f_max is the maximum frequency bin index (num_mel - 1) and alpha is sampled uniformly
from [alpha_min, alpha_max].

The transform interpolates along the frequency axis for each time frame using linear interpolation.
"""

import numpy as np
from CLAPForge.core.transforms_interface import BaseWaveformTransform

class VTLPAugment(BaseWaveformTransform):
    def __init__(self, alpha_min=0.9, alpha_max=1.1, p=1.0):
        """
        Initialize VTLPAugment.

        Parameters:
            alpha_min (float): Minimum warp factor (e.g., 0.9).
            alpha_max (float): Maximum warp factor (e.g., 1.1).
            p (float): Probability of applying this augmentation.
        """
        super().__init__(p=p)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def apply(self, input_melspec, sample_rate=None):
        """
        Apply VTLP to a mel-spectrogram.

        Parameters:
            input_melspec (np.ndarray): Input mel-spectrogram of shape (time, num_mel).
            sample_rate (int, optional): Sample rate of the original audio (not used here).

        Returns:
            np.ndarray: The VTLP-warped mel-spectrogram with the same shape as input.
        """
        # Get the dimensions of the mel-spectrogram.
        time_steps, num_mel = input_melspec.shape
        
        # Sample a random warp factor from the range [alpha_min, alpha_max].
        alpha = np.random.uniform(self.alpha_min, self.alpha_max)
        
        # Define the maximum frequency index.
        f_max = num_mel - 1
        
        # Original frequency bin indices.
        freq_bins = np.arange(num_mel)
        # Compute warped frequency indices using the VTLP formula.
        warped_freq = f_max * (freq_bins / f_max) ** alpha
        
        # Initialize the output array.
        warped_melspec = np.empty_like(input_melspec)
        # For each time frame, interpolate the mel-spectrogram along the frequency axis.
        for t in range(time_steps):
            warped_melspec[t, :] = np.interp(freq_bins, warped_freq, input_melspec[t, :])
        
        return warped_melspec
