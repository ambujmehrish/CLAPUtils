import zipfile
import re

# List of directories (relative to audiomentations‑main) to exclude.
excluded_dirs = {
    ".circleci",
    ".github",
    "demo",
    "docs",
    "overrides",
    "tests",
}

# List of individual files (in the root of audiomentations‑main) to exclude.
excluded_files = {
    ".codecov.yml",
    ".coveragerc",
    ".editorconfig",
    ".gitignore",
    "mkdocs.yml",
    "packaging.md",
    "pytest.ini"
}

# Mapping of original class names to new class names for files in augmentations.
class_name_map = {
    "AddBackgroundNoise": "BackgroundNoiseAugment",
    "AddColorNoise": "ColorNoiseAugment",
    "AddGaussianNoise": "GaussianNoiseAugment",
    "AddGaussianSNR": "GaussianSNRAugment",
    "AddShortNoises": "ShortNoisesAugment",
    "AdjustDuration": "DurationAdjustAugment",
    "AirAbsorption": "AirAbsorptionAugment",
    "Aliasing": "AliasingAugment",
    "ApplyImpulseResponse": "ImpulseResponseAugment",
    "BandPassFilter": "BandPassFilterAugment",
    "BandStopFilter": "BandStopFilterAugment",
    "BaseButterwordFilter": "ButterworthFilterBase",
    "BitCrush": "BitCrushAugment",
    "Clip": "ClipAugment",
    "ClippingDistortion": "ClippingDistortionAugment",
    "Gain": "GainAugment",
    "GainTransition": "GainTransitionAugment",
    "HighPassFilter": "HighPassFilterAugment",
    "HighShelfFilter": "HighShelfFilterAugment",
    "LambdaTransform": "LambdaTransformAugment",
    "Limiter": "LimiterAugment",
    "LoudnessNormalization": "LoudnessNormAugment",
    "LowPassFilter": "LowPassFilterAugment",
    "LowShelfFilter": "LowShelfFilterAugment",
    "Mp3Compression": "MP3CompressionAugment",
    "Normalize": "NormalizeAugment",
    "Padding": "PaddingAugment",
    "PeakingFilter": "PeakingFilterAugment",
    "PitchShift": "PitchShiftAugment",
    "PolarityInversion": "PolarityInvertAugment",
    "RepeatPart": "RepeatPartAugment",
    "Resample": "ResampleAugment",
    "Reverse": "ReverseAugment",
    "RoomSimulator": "RoomSimulateAugment",
    "SevenBandParametricEQ": "SevenBandEQAugment",
    "Shift": "ShiftAugment",
    "TanhDistortion": "TanhDistortionAugment",
    "TimeMask": "TimeMaskAugment",
    "TimeStretch": "TimeStretchAugment",
    "Trim": "TrimAugment",
}

def should_exclude(filepath):
    """
    Determine if a file (given its path in the zip) should be excluded.
    """
    parts = filepath.split('/')
    # Skip if the file is not inside the top-level folder "audiomentations-main"
    if not parts or parts[0] != "audiomentations-main":
        return True
    # Check if any directory (except the top-level and the file itself) is in excluded_dirs.
    for part in parts[1:-1]:
        if part in excluded_dirs:
            return True
    # Check if the file (in the root) is in excluded_files.
    if len(parts) == 2 and parts[-1] in excluded_files:
        return True
    return False

def is_text_file(filename):
    text_extensions = ('.py', '.md', '.yml', '.ini', '.txt')
    return filename.endswith(text_extensions)

def process_file_content(filepath, content):
    """
    Process text file content:
     - For files under the augmentations folder, apply class renaming and variable renaming.
     - For all text files, replace "audiomentations" with "CLAPForge".
    """
    # If the file is in the augmentations folder, do class and variable substitutions.
    if re.search(r"audiomentations/augmentations/", filepath):
        for old_class, new_class in class_name_map.items():
            content = re.sub(rf"\b{old_class}\b", new_class, content)
        content = re.sub(r"\bsamples\b", "input_samples", content)
    # Replace the package name everywhere.
    content = re.sub(r"\baudiomentations\b", "CLAPForge", content)
    return content

# Create a new zip file for the refactored project.
with zipfile.ZipFile("CLAPForge.zip", "w", zipfile.ZIP_DEFLATED) as new_zip:
    with zipfile.ZipFile("audiomentations-main.zip", "r") as old_zip:
        for item in old_zip.infolist():
            # Skip directories.
            if item.is_dir():
                continue
            orig_path = item.filename  # e.g., "audiomentations-main/..."
            if should_exclude(orig_path):
                continue

            # Read file data.
            data = old_zip.read(item)
            # If this is a text file, decode, process substitutions, then re-encode.
            if is_text_file(orig_path):
                try:
                    text = data.decode('utf-8')
                    text = process_file_content(orig_path, text)
                    data = text.encode('utf-8')
                except UnicodeDecodeError:
                    # Not a text file; leave binary data as-is.
                    pass

            # If the file is under the package folder "audiomentations", change it to "CLAPForge"
            new_path = orig_path
            if new_path.startswith("audiomentations-main/audiomentations/"):
                new_path = new_path.replace("audiomentations-main/audiomentations/", "audiomentations-main/CLAPForge/", 1)
            new_zip.writestr(new_path, data)

print("Created CLAPForge.zip with the refactored files and removed unnecessary files.")
