#!/usr/bin/env python3
"""
A file for downloading audio recordings from the Tarteel V1 dataset.
Contributed by @kareemn.

Example usage 1: download only the audio related to surah 1 (Al-Fatiha) 140 Mb
python download.py -s 1 --use-cache --keep-downloaded-audio

Example usage 2: download the entire audio dataset
python download.py --use-cache --keep-downloaded-audio
"""

from argparse import ArgumentParser
import logging

import requests
from tqdm import tqdm

from utils import convert_to_bool
import utils.files as file_utils
from utils.recording import download_recording_from_url

TARTEEL_V1_CSV_URL = 'https://tarteel-frontend-static.s3-us-west-2.amazonaws.com/datasets' \
                     '/tarteel_v1.0.csv'
DATASET_CSV_CACHE = 'csv'
DOWNLOADED_AUDIO_CACHE = 'downloaded_audio'
RAW_AUDIO_CACHE = 'raw_audio'

parser = ArgumentParser(description='Tarteel Audio Downloader')
parser.add_argument('--csv-url', type=str, default=TARTEEL_V1_CSV_URL)
parser.add_argument('--local-csv-filename', type=str, default='local.csv')
parser.add_argument('--cache-dir', type=str, default='.cache')
parser.add_argument('-u', '--use-cache', action='store_true')
parser.add_argument('-s', '--surah', type=int, default=0)
parser.add_argument('-k', '--keep-downloaded-audio', action='store_true')
parser.add_argument(
    '--log', choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL'], default='INFO',
    help='Logging level.'
)
args = parser.parse_args()

numeric_level = getattr(logging, args.log, None)
logging.basicConfig(level=numeric_level)


def get_correctly_labeled_entries(all_entries):
    """Get entries that are labeled and evaluated as correct."""
    return [
        entry for entry in all_entries if convert_to_bool(entry[9]) and convert_to_bool(entry[10])
    ]


def download_csv_dataset(csv_url, dataset_csv_path):
    logging.info(f"Downloading CSV from {csv_url} to {dataset_csv_path}.")
    with requests.Session() as request_session:
        response = request_session.get(csv_url)
        decoded_dataset = response.content.decode('utf-8')
        with open(dataset_csv_path, "w") as file:
            file.write(decoded_dataset)


def download_entry_audio(entry, download_audio_dir, raw_audio_dir, use_cache=True):
    """Download an audio recording given its entry in the Tarteel dataset csv."""
    surah_num, ayah_num, url, _, _, _, _, _, _, _, _, _ = entry

    # Convert data to correct type from string.
    surah_num = int(surah_num)
    ayah_num = int(ayah_num)

    # Ensure the proper surah directory structure for the downloaded audio.
    downloaded_ayah_audio_dir = file_utils.prepare_ayah_directory(
        download_audio_dir, surah_num, ayah_num)

    # Download and save the initially downloaded audio recording to the given path.
    download_recording_from_url(url, downloaded_ayah_audio_dir, use_cache)

    # Ensure the proper surah directory structure for the raw audio.
    file_utils.prepare_ayah_directory(raw_audio_dir, surah_num, ayah_num)


if __name__ == "__main__":

    # Parse the arguments.
    cache_directory = args.cache_dir
    use_cache = args.use_cache
    surah_to_download = args.surah

    # Prepare all requisite cache directories.
    subcache_directory_names = (DATASET_CSV_CACHE, DOWNLOADED_AUDIO_CACHE, RAW_AUDIO_CACHE)
    csv_cache_dir, downloaded_audio_dir, raw_audio_dir = file_utils.prepare_cache_directories(
        subcache_directory_names,
        cache_directory,
        use_cache)

    # Create path to dataset csv.
    path_to_dataset_csv = file_utils.get_path_to_dataset_csv(
        csv_cache_dir, args.local_csv_filename)

    # If we have decided not to use the cache, download the dataset CSV.
    if not use_cache:
        file_utils.clean_cache_directories(cache_directory)

    # If csv is not in specified location, then throw an error.
    if not file_utils.does_cached_csv_dataset_exist(path_to_dataset_csv):
        logging.info('Dataset CSV not found at {}. Downloading to location...'.format(
            path_to_dataset_csv))
        download_csv_dataset(args.csv_url, path_to_dataset_csv)
    else:
        logging.info("Using cached copy of dataset csv at {}.".format(path_to_dataset_csv))

    # Read the rows of dataset csv.
    header_row, entries = file_utils.get_dataset_entries(path_to_dataset_csv)

    # Filter out recordings that have been evaluated and labeled falsely.
    labeled_entries = get_correctly_labeled_entries(entries)

    # Download the audio in the dataset.
    for entry in tqdm(labeled_entries, desc='Audio Files'):
        if surah_to_download == 0 or entry[0] == str(surah_to_download):
            download_entry_audio(entry, downloaded_audio_dir, raw_audio_dir, use_cache)

    # If we don't want to keep the raw audio, remove it from the cache.
    if not args.keep_downloaded_audio:
        file_utils.delete_cache_directories(downloaded_audio_dir)
