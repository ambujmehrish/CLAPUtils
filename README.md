# 🌀 Hugging Face Audio Dataset Downloader

`download.py` is a Python script for downloading, extracting, converting, and organizing audio datasets from Hugging Face that are stored in `.tar` or `.zip` formats. The script converts audio files to `.flac` format and generates a CSV file containing metadata and file paths.

---

## ✅ Features

- 🔽 Downloads `.tar` or `.zip` files from Hugging Face datasets
- 📂 Extracts archive contents
- 🔊 Converts `.wav`, `.mp3`, `.ogg` audio files to `.flac`
- 📜 Reads metadata from JSON files
- 📊 Creates a consolidated CSV file with audio file paths and corresponding text

---

## 📦 Requirements

- Python 3.7+
- [ffmpeg](https://ffmpeg.org/download.html) (must be installed and accessible from the command line)

Install Python dependencies:

```bash
pip install pandas tqdm requests
```

---

## 📁 Dataset File Format

Prepare a text file (e.g., datasets.txt) listing datasets and their archive types (either tar or zip):

```
<dataset_name> <file_type>

speechcolab/GigaSpeech tar
superb/asr zip
```

---

## 🚀 Usage

```bash
python download.py --datasets_file config/datasets.txt --output_dir ./output
```

- `--datasets_file`: Path to your text file with dataset names and file types.
- `--output_dir`: Folder where the extracted files and CSVs will be saved.

---

## 📂 Output Structure

```
output/
  └── <dataset_name>/
      ├── audios/
      │   └── train/
      │       ├── *.flac
      │       └── *.json
      └── csv/
          └── <dataset_name>_data.csv
```

Each CSV contains:

- `file_path`: Path to the .flac file
- `metadata`: Transcription or text from the corresponding .json
- `dataset`: Name of the dataset

---

## 🧪 Augmentation

Augmentation support is currently under development. This will include optional steps for applying audio augmentations (e.g., noise addition, time stretching, pitch shifting) during the preprocessing pipeline.

