import numpy as np
from numpy.typing import NDArray

from CLAPForge.core.transforms_interface import BaseWaveformTransform
from CLAPForge.core.utils import get_max_abs_amplitude


class NormalizeAugment(BaseWaveformTransform):
    """
    Apply a constant amount of gain, so that highest signal level present in the sound becomes
    0 dBFS, i.e. the loudest level allowed if all input_samples must be between -1 and 1. Also known
    as peak normalization.
    """

    supports_multichannel = True

    def __init__(self, apply_to: str = "all", p: float = 0.5):
        super().__init__(p)
        assert apply_to in ("all", "only_too_loud_sounds")
        self.apply_to = apply_to

    def randomize_parameters(self, input_samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(input_samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["max_amplitude"] = get_max_abs_amplitude(input_samples)

    def apply(self, input_samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        if (
            self.apply_to == "only_too_loud_sounds"
            and self.parameters["max_amplitude"] < 1.0
        ):
            return input_samples

        if self.parameters["max_amplitude"] > 0:
            return input_samples / self.parameters["max_amplitude"]
        else:
            return input_samples
