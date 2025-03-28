import argparse
import requests
import os
import tarfile
import zipfile
import json
import csv
import shutil
import pandas as pd
import subprocess
from tqdm import tqdm

# --------------------------
# Audio Conversion Function
# --------------------------
def convert_to_flac(audio_file):
    """
    Converts an audio file to FLAC format if it is not already in FLAC format.
    Returns the path to the FLAC file.
    """
    base, ext = os.path.splitext(audio_file)
    if ext.lower() == ".flac":
        return audio_file

    flac_file = base + ".flac"
    # If already converted, return it
    if os.path.exists(flac_file):
        return flac_file

    try:
        # Convert using ffmpeg; ensure ffmpeg is installed on your system.
        command = ["ffmpeg", "-y", "-i", audio_file, "-acodec", "flac", flac_file]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Converted {audio_file} to {flac_file}")
        return flac_file
    except Exception as e:
        print(f"Error converting {audio_file} to FLAC: {e}")
        return audio_file

# --------------------------
# Tar File Handling Functions
# --------------------------
def fetch_all_tar_file_urls(dataset_name, directory="main"):
    """
    Recursively fetches .tar file URLs from the Hugging Face dataset API.
    """
    tar_urls = []
    try:
        api_directory = directory if directory.startswith("main") else "main/" + directory
        url = f"https://huggingface.co/api/datasets/{dataset_name}/tree/{api_directory}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch dataset metadata for {dataset_name} at {api_directory}: {response.status_code}")
            return []
        data = response.json()
        for item in data:
            if item.get("type") == "directory":
                new_directory = item["path"]
                if not new_directory.startswith("main/"):
                    new_directory = "main/" + new_directory
                tar_urls.extend(fetch_all_tar_file_urls(dataset_name, new_directory))
            else:
                if item["path"].endswith(".tar"):
                    file_path = item["path"]
                    if not file_path.startswith("main/"):
                        file_path = "main/" + file_path
                    relative_file_path = file_path[len("main/"):]
                    tar_urls.append(f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{relative_file_path}")
    except Exception as e:
        print(f"Error fetching tar file URLs for {dataset_name} at {directory}: {e}")
    return tar_urls

def download_tar_files(tar_file_urls, output_dir):
    """
    Downloads .tar files with a progress bar and saves them in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for tar_url in tar_file_urls:
        file_name = tar_url.split("/")[-1]
        file_path = os.path.join(output_dir, file_name)
        try:
            if os.path.exists(file_path):
                print(f"File {file_name} already exists in {output_dir}. Skipping download.")
                continue
            print(f"Downloading {file_name} to {output_dir}...")
            response = requests.get(tar_url, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                with open(file_path, "wb") as f, tqdm(
                    desc=file_name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                        bar.update(len(chunk))
                print(f"Downloaded {file_name}")
            else:
                print(f"Failed to download {file_name}: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {file_name}: {e}")
            continue

def extract_tar_files_and_create_csv(dataset_name, output_dir):
    """
    Extracts all .tar files in the output directory, processes JSON metadata,
    converts audio files to FLAC, and creates a CSV.
    """
    dataset_identifier = dataset_name.split("/")[-1]
    dataset_dir = os.path.join(output_dir, "audios", "train")
    os.makedirs(dataset_dir, exist_ok=True)
    csv_data = []

    for file_name in os.listdir(output_dir):
        if file_name.endswith(".tar"):
            tar_path = os.path.join(output_dir, file_name)
            try:
                print(f"Extracting {file_name}...")
                with tarfile.open(tar_path) as tar:
                    for member in tar.getmembers():
                        try:
                            member.name = os.path.basename(member.name)  # Remove directory structure
                            tar.extract(member, path=dataset_dir)
                        except Exception as e:
                            print(f"Error extracting member {member.name} from {file_name}: {e}")
                os.remove(tar_path)
            except Exception as e:
                print(f"Error processing tar file {file_name}: {e}")
                continue

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, "r") as json_file:
                        metadata = json.load(json_file)
                except Exception as e:
                    print(f"Error reading JSON file {json_path}: {e}")
                    continue

                json_base_name = os.path.splitext(file)[0]
                audio_file = None
                try:
                    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
                        potential_audio_file = os.path.join(root, json_base_name + ext)
                        if os.path.exists(potential_audio_file):
                            audio_file = potential_audio_file
                            break
                except Exception as e:
                    print(f"Error finding audio file for {json_path}: {e}")
                    continue

                if audio_file:
                    flac_audio = convert_to_flac(audio_file)
                    csv_data.append({
                        "file_path": flac_audio,
                        "metadata": metadata.get("text", ""),
                        "dataset": dataset_identifier
                    })

    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{dataset_identifier}_data.csv")
    try:
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["file_path", "metadata", "dataset"])
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"CSV file saved to {csv_path}")
    except Exception as e:
        print(f"Error writing CSV file {csv_path}: {e}")

# --------------------------
# ZIP File Handling Functions
# --------------------------
def fetch_all_zip_file_urls(dataset_name, directory="main"):
    """
    Recursively fetches .zip file URLs from the Hugging Face dataset API.
    """
    zip_urls = []
    try:
        api_directory = directory if directory.startswith("main") else "main/" + directory
        url = f"https://huggingface.co/api/datasets/{dataset_name}/tree/{api_directory}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch dataset metadata for {dataset_name} at {api_directory}: {response.status_code}")
            return []
        data = response.json()
        for item in data:
            if item.get("type") == "directory":
                new_directory = item["path"]
                if not new_directory.startswith("main/"):
                    new_directory = "main/" + new_directory
                zip_urls.extend(fetch_all_zip_file_urls(dataset_name, new_directory))
            else:
                if item["path"].endswith(".zip"):
                    file_path = item["path"]
                    if not file_path.startswith("main/"):
                        file_path = "main/" + file_path
                    relative_file_path = file_path[len("main/"):]
                    zip_urls.append(f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{relative_file_path}")
    except Exception as e:
        print(f"Error fetching zip file URLs for {dataset_name} at {directory}: {e}")
    return zip_urls

def download_zip_files(zip_file_urls, output_dir):
    """
    Downloads .zip files with a progress bar and saves them in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for zip_url in zip_file_urls:
        file_name = zip_url.split("/")[-1]
        file_path = os.path.join(output_dir, file_name)
        try:
            if os.path.exists(file_path):
                print(f"File {file_name} already exists in {output_dir}. Skipping download.")
                continue
            print(f"Downloading {file_name} to {output_dir}...")
            response = requests.get(zip_url, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                with open(file_path, "wb") as f, tqdm(
                    desc=file_name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                        bar.update(len(chunk))
                print(f"Downloaded {file_name}")
            else:
                print(f"Failed to download {file_name}: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {file_name}: {e}")
            continue

def extract_zip_files_and_create_csv(dataset_name, output_dir):
    """
    Extracts all .zip files in the output directory, processes JSON metadata
    (if available), converts audio files to FLAC, and creates a CSV.
    If a JSON file is not present, it skips processing without error.
    """
    dataset_identifier = dataset_name.split("/")[-1]
    dataset_dir = os.path.join(output_dir, "audios", "train")
    os.makedirs(dataset_dir, exist_ok=True)
    csv_data = []

    for file_name in os.listdir(output_dir):
        if file_name.endswith(".zip"):
            zip_path = os.path.join(output_dir, file_name)
            try:
                print(f"Extracting {file_name}...")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    for member in z.infolist():
                        try:
                            member_filename = os.path.basename(member.filename)
                            if not member_filename:
                                continue
                            z.extract(member, path=dataset_dir)
                        except Exception as e:
                            print(f"Error extracting member {member.filename} from {file_name}: {e}")
                os.remove(zip_path)
            except Exception as e:
                print(f"Error processing zip file {file_name}: {e}")
                continue

    # Process JSON files in the extracted directory
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, "r") as json_file:
                        metadata = json.load(json_file)
                except Exception as e:
                    print(f"Error reading JSON file {json_path}: {e}")
                    continue

                json_base_name = os.path.splitext(file)[0]
                audio_file = None
                try:
                    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
                        potential_audio_file = os.path.join(root, json_base_name + ext)
                        if os.path.exists(potential_audio_file):
                            audio_file = potential_audio_file
                            break
                except Exception as e:
                    print(f"Error finding audio file for {json_path}: {e}")
                    continue

                if audio_file:
                    flac_audio = convert_to_flac(audio_file)
                    csv_data.append({
                        "file_path": flac_audio,
                        "metadata": metadata.get("text", ""),
                        "dataset": dataset_identifier
                    })
                else:
                    # Skip silently if corresponding audio file is not found
                    print(f"No audio file found corresponding to {json_path}. Skipping.")
                    continue

    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{dataset_identifier}_zip_data.csv")
    try:
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["file_path", "metadata", "dataset"])
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"CSV file saved to {csv_path}")
    except Exception as e:
        print(f"Error writing CSV file {csv_path}: {e}")

# --------------------------
# Main Processing Function
# --------------------------
def main(datasets_file, output_dir):
    """
    Reads a datasets file where each line contains two entries:
      - dataset name
      - file type ("tar", "parquet", or "zip")
    
    Based on the file type, it either processes tar files, loads a parquet dataset
    using the Hugging Face dataloader, or processes zip files.
    """
    try:
        with open(datasets_file, "r") as f:
            dataset_entries = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading datasets file {datasets_file}: {e}")
        return

    for entry in dataset_entries:
        parts = entry.split(" ")
        if len(parts) != 2:
            print(f"Invalid format in line: {entry}")
            continue

        dataset_name = parts[0].strip()
        file_type = parts[1].strip().lower()

        print(f"\nProcessing dataset: {dataset_name} with file type: {file_type}")
        sanitized_name = dataset_name.replace("/", "_")
        dataset_output_dir = os.path.join(output_dir, sanitized_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        if file_type == "tar":
            try:
                tar_file_urls = fetch_all_tar_file_urls(dataset_name)
                tar_file_urls = tar_file_urls[0:3]
                if tar_file_urls:
                    download_tar_files(tar_file_urls, dataset_output_dir)
                    extract_tar_files_and_create_csv(dataset_name, dataset_output_dir)
                else:
                    print(f"No .tar files found for {dataset_name}")
            except Exception as e:
                print(f"Error processing tar files for {dataset_name}: {e}")
        elif file_type == "parquet":
            try:
                from datasets import load_dataset
                print(f"Downloading parquet dataset {dataset_name} using Hugging Face dataloader...")
                hf_dataset = load_dataset(dataset_name, split="train")
                parquet_output_dir = os.path.join(dataset_output_dir, "audios", "parquet_extracted")
                os.makedirs(parquet_output_dir, exist_ok=True)
                parquet_csv_data = []

                for row in hf_dataset:
                    try:
                        audio_field = row.get("audio")
                        if audio_field is None:
                            print("No audio column found in row, skipping.")
                            continue
                        if isinstance(audio_field, dict) and "path" in audio_field:
                            audio_path = audio_field["path"]
                        else:
                            audio_path = audio_field

                        if os.path.exists(audio_path):
                            flac_audio = convert_to_flac(audio_path)
                            destination = os.path.join(parquet_output_dir, os.path.basename(flac_audio))
                            shutil.copy(flac_audio, destination)
                            print(f"Extracted and converted: {flac_audio}")
                        else:
                            print(f"Audio file not found: {audio_path}")
                            continue

                        caption = ""
                        for col in ['caption', 'metadata', 'text']:
                            if col in row and row[col]:
                                caption = row[col]
                                break
                        parquet_csv_data.append({
                            "file_path": destination,
                            "metadata": caption,
                            "dataset": os.path.basename(dataset_output_dir)
                        })
                    except Exception as e:
                        print(f"Error processing a row in dataset {dataset_name}: {e}")
                        continue

                csv_dir = os.path.join(dataset_output_dir, "csv")
                os.makedirs(csv_dir, exist_ok=True)
                csv_path = os.path.join(csv_dir, "parquet_data.csv")
                try:
                    with open(csv_path, "w", newline="") as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=["file_path", "metadata", "dataset"])
                        writer.writeheader()
                        writer.writerows(parquet_csv_data)
                    print(f"CSV file for parquet data saved to {csv_path}")
                except Exception as e:
                    print(f"Error writing parquet CSV file {csv_path}: {e}")
            except Exception as e:
                print(f"Error loading parquet dataset {dataset_name} with Hugging Face dataloader: {e}")
        elif file_type == "zip":
            try:
                zip_file_urls = fetch_all_zip_file_urls(dataset_name)
                zip_file_urls = zip_file_urls[0:3]
                if zip_file_urls:
                    download_zip_files(zip_file_urls, dataset_output_dir)
                    extract_zip_files_and_create_csv(dataset_name, dataset_output_dir)
                else:
                    print(f"No .zip files found for {dataset_name}")
            except Exception as e:
                print(f"Error processing zip files for {dataset_name}: {e}")
        else:
            print(f"Unknown file type '{file_type}' for dataset {dataset_name}")

        print(f"Finished processing dataset: {dataset_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Hugging Face datasets for audio extraction and conversion to FLAC."
    )
    parser.add_argument("--datasets_file", type=str, required=True,
                        help="Path to a text file containing dataset names and file types (comma-separated, one per line).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where output files will be stored.")
    args = parser.parse_args()

    main(args.datasets_file, args.output_dir)
