import numpy as np
from numpy.typing import NDArray

from CLAPForge.core.transforms_interface import BaseWaveformTransform


class DurationAdjustAugment(BaseWaveformTransform):
    """
    TrimAugment or pad the audio to the specified length/duration in input_samples or seconds. If the
    input sound is longer than the target duration, pick a random offset and crop the
    sound to the target duration. If the input sound is shorter than the target
    duration, pad the sound so the duration matches the target duration.
    """

    supports_multichannel = True

    def __init__(
        self,
        duration_samples: int = None,
        duration_seconds: float = None,
        padding_mode: str = "silence",
        padding_position: str = "end",
        p: float = 0.5,
    ):
        """
        :param duration_samples: Target duration in number of input_samples
        :param duration_seconds: Target duration in seconds
        :param padding_mode: PaddingAugment mode. Must be "silence", "wrap" or "reflect". Only
            used when audio input is shorter than the target duration.
        :param padding_position: The position of the inserted/added padding. Must be
            "start" or "end". Only used when audio input is shorter than the target duration.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert padding_mode in ("silence", "wrap", "reflect")
        if padding_mode == "silence":
            padding_mode = "constant"  # for numpy.pad compatibility
        self.padding_mode = padding_mode

        assert padding_position in ("start", "end")
        self.padding_position = padding_position
        
        assert duration_samples is not None or duration_seconds is not None
        if duration_samples is not None and duration_seconds is not None:
            raise ValueError(
                "should have duration_samples or duration_seconds, but not both"
            )
        elif duration_seconds:
            assert duration_seconds > 0
            self.get_target_samples = lambda sr: int(duration_seconds * sr)
        elif duration_samples:
            assert duration_samples > 0
            self.get_target_samples = lambda sr: duration_samples
            
        self.duration_samples = duration_samples
        self.duration_seconds = duration_seconds

    def apply(self, input_samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        target_samples = self.get_target_samples(sample_rate)
        sample_length = input_samples.shape[-1]

        if sample_length == target_samples:
            return input_samples

        elif sample_length > target_samples:
            start = np.random.randint(0, sample_length - target_samples)
            return input_samples[..., start : start + target_samples]

        elif sample_length < target_samples:
            padding_length = target_samples - sample_length
            if input_samples.ndim == 1:
                if self.padding_position == "start":
                    pad_width = (padding_length, 0)
                else:
                    pad_width = (0, padding_length)
            else:
                if self.padding_position == "start":
                    pad_width = ((0, 0), (padding_length, 0))
                else:
                    pad_width = ((0, 0), (0, padding_length))
            return np.pad(input_samples, pad_width, self.padding_mode)
