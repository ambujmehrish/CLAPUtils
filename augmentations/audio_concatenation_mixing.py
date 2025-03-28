"""
Audio Concatenation & Mixing Augmentation

This augmentation randomly combines different audio sources (speech, sound effects, music)
by introducing structured overlays. Depending on the randomly chosen scenario, it overlays:
- Speech + Sound Effects: Overlay speech with environmental sounds.
- Speech + Music: Mix voice data with music at varying intensities.
- Sound Effects + Music: Blend sound effects with background music.

Decibel-based loudness adjustments ensure that one audio source is dominant over the other.
"""

import os
import random
import numpy as np
from CLAPForge.core.transforms_interface import BaseWaveformTransform
from CLAPForge.core.audio_loading_utils import load_sound_file

class AudioConcatMixAugment(BaseWaveformTransform):
    def __init__(self,
                 speech_dir="audio/speech",
                 sound_effects_dir="audio/sound_effects",
                 music_dir="audio/music",
                 p=0.5,
                 mix_method="overlay"):
        """
        Initialize the Audio Concatenation & Mixing Augmentation.

        Parameters:
            speech_dir (str): Directory containing speech audio files.
            sound_effects_dir (str): Directory containing sound effects audio files.
            music_dir (str): Directory containing music audio files.
            p (float): Probability of applying the augmentation.
            mix_method (str): 'overlay' to mix audio sources, 'concatenate' to join sequentially.
        """
        super().__init__(p=p)
        self.speech_dir = speech_dir
        self.sound_effects_dir = sound_effects_dir
        self.music_dir = music_dir
        self.mix_method = mix_method

    def get_random_audio(self, directory):
        """
        Load a random audio file from the specified directory.
        
        Returns:
            samples (np.ndarray): The audio samples.
            sample_rate (int): The sample rate.
        """
        if not os.path.isdir(directory):
            return None, None
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.wav', '.flac', '.ogg', '.mp3'))]
        if not files:
            return None, None
        file_path = os.path.join(directory, random.choice(files))
        samples, sr = load_sound_file(file_path)
        return samples, sr

    def apply(self, input_samples, sample_rate):
        """
        Apply the augmentation to the input audio.
        
        Parameters:
            input_samples (np.ndarray): Input audio samples.
            sample_rate (int): Sample rate of the input audio.
        
        Returns:
            np.ndarray: The mixed audio samples.
        """
        # Choose a random mixing scenario.
        scenarios = ["speech_sound_effects", "speech_music", "sound_effects_music"]
        scenario = random.choice(scenarios)

        if scenario == "speech_sound_effects":
            primary_dir = self.speech_dir
            secondary_dir = self.sound_effects_dir
        elif scenario == "speech_music":
            primary_dir = self.speech_dir
            secondary_dir = self.music_dir
        else:  # "sound_effects_music"
            primary_dir = self.sound_effects_dir
            secondary_dir = self.music_dir

        primary_audio, sr_primary = self.get_random_audio(primary_dir)
        secondary_audio, sr_secondary = self.get_random_audio(secondary_dir)

        # If either audio is missing, return the original input.
        if primary_audio is None or secondary_audio is None:
            return input_samples

        # Assume sample rates match; in a full implementation, you might resample if needed.
        # Adjust lengths: use the length of the primary audio.
        len_primary = len(primary_audio)
        len_secondary = len(secondary_audio)
        if len_secondary < len_primary:
            pad_width = len_primary - len_secondary
            secondary_audio = np.pad(secondary_audio, (0, pad_width), mode='constant')
        elif len_secondary > len_primary:
            secondary_audio = secondary_audio[:len_primary]

        # Apply decibel-based loudness adjustment.
        # Make primary dominant and reduce secondary by a random value between 6dB and 15dB.
        reduction_db = random.uniform(6, 15)
        secondary_gain = 10 ** (-reduction_db / 20.0)
        secondary_audio = secondary_audio * secondary_gain

        # Mix audio based on the chosen method.
        if self.mix_method == "overlay":
            mixed_audio = primary_audio + secondary_audio
            # Normalize if necessary.
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 1.0:
                mixed_audio = mixed_audio / max_val
        elif self.mix_method == "concatenate":
            mixed_audio = np.concatenate([primary_audio, secondary_audio])
        else:
            mixed_audio = primary_audio

        return mixed_audio
