import numpy as np
from CLAPForge.core.transforms_interface import BaseWaveformTransform

class SpectralInversionAugment(BaseWaveformTransform):
    def __init__(self, method="invert", p=1.0):
        """
        Initialize Spectral Inversion augmentation.
        
        Parameters:
            method (str): "invert" to flip the frequency axis or "shuffle" to randomly reorder frequency bins.
            p (float): Probability of applying this augmentation.
        """
        super().__init__(p=p)
        self.method = method

    def apply(self, input_melspec, sample_rate=None):
        """
        Apply spectral inversion or frequency shuffling to a mel-spectrogram.
        
        Parameters:
            input_melspec (np.ndarray): A 2D mel-spectrogram with shape (time, frequency).
            sample_rate (int): Not used (included for interface consistency).
        
        Returns:
            np.ndarray: The augmented mel-spectrogram.
        """
        if self.method == "invert":
            return np.flip(input_melspec, axis=1)
        elif self.method == "shuffle":
            shuffled = np.copy(input_melspec)
            num_bins = shuffled.shape[1]
            indices = np.arange(num_bins)
            np.random.shuffle(indices)
            return shuffled[:, indices]
        else:
            return input_melspec
