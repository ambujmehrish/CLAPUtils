import numpy as np
from CLAPForge.core.transforms_interface import BaseWaveformTransform

class DynamicRangeAugment(BaseWaveformTransform):
    def __init__(self, threshold=0.5, ratio=2.0, mode='compress', p=1.0):
        """
        Initialize dynamic range augmentation.
        
        Parameters:
            threshold (float): The amplitude threshold above which dynamic range is adjusted.
            ratio (float): Compression (ratio > 1) or expansion factor.
            mode (str): Either 'compress' or 'expand'.
            p (float): Probability of applying this augmentation.
        """
        super().__init__(p=p)
        self.threshold = threshold
        self.ratio = ratio
        self.mode = mode

    def apply(self, input_samples, sample_rate):
        """
        Apply dynamic range compression or expansion.
        
        Parameters:
            input_samples (np.ndarray): Input audio waveform.
            sample_rate (int): Sample rate (unused, included for interface consistency).
        
        Returns:
            np.ndarray: Audio after dynamic range processing.
        """
        output = np.copy(input_samples)
        # Process sample-wise (a simplified approach)
        for i in range(len(output)):
            abs_val = abs(output[i])
            if abs_val > self.threshold:
                excess = abs_val - self.threshold
                if self.mode == 'compress':
                    output[i] = np.sign(output[i]) * (self.threshold + excess / self.ratio)
                elif self.mode == 'expand':
                    output[i] = np.sign(output[i]) * (self.threshold + excess * self.ratio)
        # Clip to [-1, 1] range
        output = np.clip(output, -1.0, 1.0)
        return output
