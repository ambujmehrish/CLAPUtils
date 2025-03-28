import numpy as np
from CLAPForge.augmentations.vtlp_augment import VTLPAugment

# Assume we have a mel-spectrogram of shape (time, num_mel)
# For example, create a dummy mel-spectrogram:
time_steps = 100
num_mel = 64
mel_spectrogram = np.random.rand(time_steps, num_mel).astype(np.float32)

# Instantiate VTLPAugment (e.g., with default alpha range [0.9, 1.1])
vtlp_transform = VTLPAugment(alpha_min=0.9, alpha_max=1.1, p=1.0)

# Apply VTLPAugment to the mel-spectrogram
warped_melspec = vtlp_transform(mel_spectrogram, sample_rate=16000)

print("Original mel-spectrogram shape:", mel_spectrogram.shape)
print("Warped mel-spectrogram shape:", warped_melspec.shape)
