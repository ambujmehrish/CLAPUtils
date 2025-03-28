import random

import numpy as np
from numpy.typing import NDArray

from CLAPForge.core.transforms_interface import BaseWaveformTransform


class ClippingDistortionAugment(BaseWaveformTransform):
    """Distort signal by clipping a random percentage of points

    The percentage of points that will be clipped is drawn from a uniform distribution between
    the two input parameters min_percentile_threshold and max_percentile_threshold. If for instance
    30% is drawn, the input_samples are clipped if they're below the 15th or above the 85th percentile.
    """

    supports_multichannel = True

    def __init__(
        self,
        min_percentile_threshold: int = 0,
        max_percentile_threshold: int = 40,
        p: float = 0.5,
    ):
        """
        :param min_percentile_threshold: int, A lower bound on the total percent of input_samples that
            will be clipped
        :param max_percentile_threshold: int, An upper bound on the total percent of input_samples that
            will be clipped
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_percentile_threshold <= max_percentile_threshold
        assert 0 <= min_percentile_threshold <= 100
        assert 0 <= max_percentile_threshold <= 100
        self.min_percentile_threshold = min_percentile_threshold
        self.max_percentile_threshold = max_percentile_threshold

    def randomize_parameters(self, input_samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(input_samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["percentile_threshold"] = random.randint(
                self.min_percentile_threshold, self.max_percentile_threshold
            )

    def apply(self, input_samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        lower_percentile_threshold = int(self.parameters["percentile_threshold"] / 2)
        lower_threshold, upper_threshold = np.percentile(
            input_samples, [lower_percentile_threshold, 100 - lower_percentile_threshold]
        )
        input_samples = np.clip(input_samples, lower_threshold, upper_threshold)
        return input_samples
