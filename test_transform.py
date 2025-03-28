import argparse
import json
import importlib
import soundfile as sf
from CLAPForge.core.audio_loading_utils import load_sound_file

# Mapping from short augmentation identifiers to a tuple (class_name, module_name)
AUG_MAPPING = {
    "background_noise": ("BackgroundNoiseAugment", "background_noise"),
    "color_noise": ("ColorNoiseAugment", "add_color_noise"),
    "gaussian_noise": ("GaussianNoiseAugment", "add_gaussian_noise"),
    "gaussian_snr": ("GaussianSNRAugment", "add_gaussian_snr"),
    "short_noises": ("ShortNoisesAugment", "add_short_noises"),
    "duration_adjust": ("DurationAdjustAugment", "adjust_duration"),
    "air_absorption": ("AirAbsorptionAugment", "air_absorption"),
    "aliasing": ("AliasingAugment", "aliasing"),
    "impulse_response": ("ImpulseResponseAugment", "apply_impulse_response"),
    "band_pass_filter": ("BandPassFilterAugment", "band_pass_filter"),
    "band_stop_filter": ("BandStopFilterAugment", "band_stop_filter"),
    "bit_crush": ("BitCrushAugment", "bit_crush"),
    "clip": ("ClipAugment", "clip"),
    "clipping_distortion": ("ClippingDistortionAugment", "clipping_distortion"),
    "gain": ("GainAugment", "gain"),
    "gain_transition": ("GainTransitionAugment", "gain_transition"),
    "high_pass_filter": ("HighPassFilterAugment", "high_pass_filter"),
    "high_shelf_filter": ("HighShelfFilterAugment", "high_shelf_filter"),
    "lambda_transform": ("LambdaTransformAugment", "lambda_transform"),
    "limiter": ("LimiterAugment", "limiter"),
    "loudness_norm": ("LoudnessNormAugment", "loudness_normalization"),
    "low_pass_filter": ("LowPassFilterAugment", "low_pass_filter"),
    "low_shelf_filter": ("LowShelfFilterAugment", "low_shelf_filter"),
    "mp3_compression": ("MP3CompressionAugment", "mp3_compression"),
    "normalize": ("NormalizeAugment", "normalize"),
    "padding": ("PaddingAugment", "padding"),
    "peaking_filter": ("PeakingFilterAugment", "peaking_filter"),
    "pitch_shift": ("PitchShiftAugment", "pitch_shift"),
    "polarity_invert": ("PolarityInvertAugment", "polarity_inversion"),
    "repeat_part": ("RepeatPartAugment", "repeat_part"),
    "resample": ("ResampleAugment", "resample"),
    "reverse": ("ReverseAugment", "reverse"),
    "room_simulate": ("RoomSimulateAugment", "room_simulator"),
    "seven_band_eq": ("SevenBandEQAugment", "seven_band_parametric_eq"),
    "shift": ("ShiftAugment", "shift"),
    "tanh_distortion": ("TanhDistortionAugment", "tanh_distortion"),
    "time_mask": ("TimeMaskAugment", "time_mask"),
    "time_stretch": ("TimeStretchAugment", "time_stretch"),
    "trim": ("TrimAugment", "trim"),
    "audio_concat_mix": ("AudioConcatMixAugment", "audio_concatenation_mixing"),
    "procedural_captioning": ("ProceduralCaptioningAugment", "procedural_captioning"),
    "mixup": ("MixupAugment", "mixup"),
    "spec_augment": ("SpecAugment", "spec_augment"),
    "vtlp": ("VTLPAugment", "vtlp_augment"),
    "echo_delay": ("EchoDelayAugment", "echo_delay_augment"),
    "dynamic_range": ("DynamicRangeAugment", "dynamic_range_augment"),
    "spectral_inversion": ("SpectralInversionAugment", "spectral_inversion_augment"),
    "harmonic_percussive": ("HarmonicPercussiveAugment", "harmonic_percussive_augment")
}

def convert_value(value_str):
    """
    Attempt to convert a string value to int, float, or leave it as a string.
    """
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            # Remove surrounding quotes if present.
            if (value_str.startswith('"') and value_str.endswith('"')) or \
               (value_str.startswith("'") and value_str.endswith("'")):
                return value_str[1:-1]
            return value_str

def parse_extra_args(arg_list):
    """
    Convert a list of command-line key-value pairs into a dictionary.
    For example, ["--delay_time", "0.3", "--decay", "0.7"] becomes:
    {"delay_time": 0.3, "decay": 0.7}
    """
    extra_params = {}
    if len(arg_list) % 2 != 0:
        raise ValueError("Override arguments must be provided in key-value pairs.")
    for i in range(0, len(arg_list), 2):
        key = arg_list[i].lstrip("-")
        value = convert_value(arg_list[i + 1])
        extra_params[key] = value
    return extra_params

def main(args):
    # Load the JSON configuration (an array of augmentation configurations).
    with open("clapforge_hyperparameters.json", "r") as f:
        aug_configs = json.load(f)
    # Build a dictionary mapping from the short identifier to its configuration.
    aug_param_dict = {config["aug_name"]: config for config in aug_configs}
    
    if args.aug_type not in aug_param_dict:
        print(f"Augmentation type '{args.aug_type}' not found in hyperparameter configuration.")
        print("Available augmentation types:")
        for key in sorted(aug_param_dict.keys()):
            print(f"  {key}")
        return

    # Retrieve the default hyperparameters for the chosen augmentation.
    aug_params = aug_param_dict[args.aug_type].copy()
    # Remove the aug_name key (not needed for instantiation).
    aug_params.pop("aug_name", None)
    
    # If overrides were provided, update the defaults.
    if args.override:
        extra_params = parse_extra_args(args.override)
        aug_params.update(extra_params)
    
    # Check that the provided augmentation identifier exists in our module mapping.
    if args.aug_type not in AUG_MAPPING:
        print(f"Augmentation type '{args.aug_type}' not recognized in mapping. Available options:")
        for key in sorted(AUG_MAPPING.keys()):
            print(f"  {key}")
        return
    
    class_name, module_name = AUG_MAPPING[args.aug_type]
    
    # Dynamically import the module.
    try:
        mod = importlib.import_module(f"CLAPForge.augmentations.{module_name}")
    except ImportError as e:
        print(f"Could not import module for '{class_name}': {e}")
        return
    
    # Retrieve the augmentation class from the module.
    try:
        aug_class = getattr(mod, class_name)
    except AttributeError as e:
        print(f"Module '{module_name}' does not contain class '{class_name}': {e}")
        return
    
    # Instantiate the augmentation transform with the updated hyperparameters.
    transform = aug_class(**aug_params)
    
    # Load the input audio file.
    input_samples, sample_rate = load_sound_file(args.input_audio)
    
    # Apply the transformation.
    output = transform(input_samples, sample_rate)
    
    # If the transform returns a dictionary (e.g., for captioning), extract the audio.
    if isinstance(output, dict):
        transformed_audio = output.get("audio", None)
        caption = output.get("caption", None)
        if caption:
            print("Generated Caption:", caption)
    else:
        transformed_audio = output
    
    # Write the transformed audio to the output file.
    sf.write(args.output_audio, transformed_audio, sample_rate)
    print(f"Augmentation '{args.aug_type}' applied with parameters: {aug_params}")
    print(f"Output saved to: {args.output_audio}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a CLAPForge augmentation on an input audio file using hyperparameters from JSON. "
                    "Optional overrides can be provided with the --override flag as key-value pairs."
    )
    parser.add_argument("--aug_type", type=str, required=True,
                        help="Augmentation type identifier (e.g., echo_delay, vtlp, dynamic_range, etc.)")
    parser.add_argument("--input_audio", type=str, required=True,
                        help="Input audio file path (e.g., data/sample.wav)")
    parser.add_argument("--output_audio", type=str, required=True,
                        help="Output audio file path (e.g., output.wav)")
    parser.add_argument("--override", nargs="*", default=None,
                        help="Optional key-value pairs to override hyperparameters (e.g., --override --delay_time 0.3 --decay 0.7)")
    
    args = parser.parse_args()
    main(args)
