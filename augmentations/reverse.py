import numpy as np
from numpy.typing import NDArray

from CLAPForge.core.transforms_interface import BaseWaveformTransform


class ReverseAugment(BaseWaveformTransform):
    """
    ReverseAugment the audio, also known as time inversion. Inversion of an audio track along its time axis is
    analogous to the random flip of an image, an augmentation technique widely used in the visual domain.
    This can be relevant in the context of audio classification. It was successfully applied in the paper
    AudioCLIP: Extending CLIP to Image, Text and Audio
    https://arxiv.org/pdf/2106.13043.pdf
    """

    supports_multichannel = True

    def __init__(self, p: float = 0.5):
        """
        :param p: The probability of applying this transform
        """
        super().__init__(p)

    def apply(self, input_samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        if len(input_samples.shape) > 1:
            return np.fliplr(input_samples)
        else:
            return np.flipud(input_samples)
