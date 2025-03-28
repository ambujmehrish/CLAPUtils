from typing import Callable

import numpy as np
from numpy.typing import NDArray

from CLAPForge.core.transforms_interface import BaseWaveformTransform


class Lambda(BaseWaveformTransform):
    """
    Apply a user-defined transform (callable) to the signal.
    """

    supports_multichannel = True

    def __init__(self, transform: Callable, p: float = 0.5, **kwargs):
        """
        :param transform: A callable to be applied over input_samples. It should input
            input_samples (ndarray), sample_rate (int) and optionally some user-defined
            keyword arguments.
        :param p: The probability of applying this transform
        :param **kwargs: Any extra keyword arguments to be passed to the transform.
        """
        super().__init__(p=p)
        self.transform = transform
        self.kwargs = kwargs

    def apply(self, input_samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        return self.transform(input_samples, sample_rate, **self.kwargs)
