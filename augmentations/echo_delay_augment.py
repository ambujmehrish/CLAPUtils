import numpy as np
from CLAPForge.core.transforms_interface import BaseWaveformTransform

class EchoDelayAugment(BaseWaveformTransform):
    def __init__(self, delay_time=0.2, decay=0.5, p=1.0):
        """
        Initialize the Echo/Delay augmentation.
        
        Parameters:
            delay_time (float): Delay time in seconds.
            decay (float): The decay factor for the echo amplitude.
            p (float): Probability of applying this augmentation.
        """
        super().__init__(p=p)
        self.delay_time = delay_time
        self.decay = decay

    def apply(self, input_samples, sample_rate):
        """
        Apply echo/delay to the input audio.
        
        Parameters:
            input_samples (np.ndarray): The input audio waveform.
            sample_rate (int): The sample rate of the audio.
        
        Returns:
            np.ndarray: The audio signal with echo added.
        """
        delay_samples = int(self.delay_time * sample_rate)
        # Create an echo signal by padding the original signal
        echo_signal = np.pad(input_samples, (delay_samples, 0), mode="constant")[:len(input_samples)]
        # Combine original signal with the echo (with decay)
        output = input_samples + self.decay * echo_signal
        # Normalize to prevent clipping if necessary
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        return output
