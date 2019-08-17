#!/usr/bin/env python3
"""
A file for downloading audio recordings from tarteel.io.

Originally created by kareemn on Jan. 1, 2019.
https://github.com/kareemn/Tarteel-ML/blob/master/download.py

Modified by Hamzah Khan (khanh111) and added to the tarteel.io/Tarteel-ML on Jan. 12, 2019.

Refactored completely by Hamzah Khan (khanh111) on Aug. 03, 2019.

Example command: python download.py -s 1 --use_cache
"""

import utils.audio as audio_utils
import utils.files as file_utils
import utils.recording as recording_utils
import csv
import os
import requests
import wave

from urllib.parse import urlparse
from argparse import ArgumentParser

# Define argument constants.
TARTEEL_V1_CSV_URL = "https://d2sf46268wowyo.cloudfront.net/datasets/tarteel_v1.0.csv"
TARTEEL_LIVE_CSV_URL = "https://www.tarteel.io/download-full-dataset-csv"
TARTEEL_V1_LABELED_CSV_URL = "https://tarteel-static.s3-us-west-2.amazonaws.com/labels/tarteel_v1.0_labeled.csv"


parser = ArgumentParser(description='Tarteel Audio Downloader')
parser.add_argument('--csv_url', type=str, default=TARTEEL_V1_LABELED_CSV_URL)
parser.add_argument('--local_csv_filename', type=str, default='local.csv')
parser.add_argument('--cache_dir', type=str, default='.cache')
parser.add_argument('-u', '--use_cache', action='store_true')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-s', '--surah', type=int)
parser.add_argument('-k', '--keep_downloaded_audio', action='store_true')
args = parser.parse_args()


# Define constants.
DATASET_CSV_CACHE = 'csv'
DOWNLOADED_AUDIO_CACHE = 'downloaded_audio'
RAW_AUDIO_CACHE = 'raw_audio'


def convert_to_bool(str):
    """
    Converts string versions of true or false to bools.
    """
    upper_string = str.upper()
    if upper_string == "TRUE":
        return True
    elif upper_string == "FALSE":
        return False
    else:
        raise TypeError("Provided string is neither 'true' nor 'false'.")


def download_csv_dataset(csv_url, dataset_csv_path, verbose=False):
    if verbose:
        print("Downloading CSV from ", csv_url, " to ", dataset_csv_path, ".")
    with requests.Session() as request_session:
        response = request_session.get(csv_url)

        decoded_dataset = response.content.decode('utf-8')
        file = open(dataset_csv_path, "w")
        file.write(decoded_dataset)
        file.close()

def get_dataset_entries(path_to_dataset_csv):
    """
    A function to parse and return the header and rows of the dataset csv.
    """
    with open(path_to_dataset_csv, 'r') as file:
        content = file.read()
        csv_data = csv.reader(content.splitlines(), delimiter=',')
        entries = list(csv_data)
        header_row = entries[0]
        dataset = entries[1:]
        return header_row, dataset

def download_entry_audio(entry, download_audio_dir, raw_audio_dir, use_cache=True, verbose=False):
    """
    Downloads an audio recording given its entry in a Tarteel dataset csv.
    """
    surah_num, ayah_num, url, _, _, _, _, _, _, _, _ = entry

    # Convert data to correct type from string.
    surah_num = int(surah_num)
    ayah_num = int(ayah_num)

    # Ensure the proper surah directory structure for the downloaded audio.
    downloaded_ayah_audio_dir = file_utils.prepare_ayah_directory(download_audio_dir, surah_num, ayah_num)

    # Download and save the initially downloaded audio recording to the given path.
    downloaded_audio_path = recording_utils.download_recording_from_url(url, downloaded_ayah_audio_dir, use_cache=use_cache, verbose=verbose)

    # Ensure the proper surah directory structure for the raw audio.
    raw_ayah_audio_dir = file_utils.prepare_ayah_directory(raw_audio_dir, surah_num, ayah_num)

    # Convert the downloaded audio to a standard raw audio format and move it to another directory.
    # This audio is being converted to 44.1 MHz, 16bps audio (Google Speech Recognition's requirements).
    raw_audio_path = audio_utils.convert_audio(downloaded_audio_path, raw_ayah_audio_dir, use_cache=use_cache, verbose=verbose)

if __name__ == "__main__":

    # Parse the arguments.
    cache_directory = args.cache_dir
    dataset_csv_url = args.csv_url
    dataset_csv_filename = args.local_csv_filename
    use_cache = args.use_cache
    verbose = args.verbose
    surah_to_download = args.surah
    keep_downloaded_audio = args.keep_downloaded_audio

    # Prepare all requisite cache directories.
    subcache_directory_names = (DATASET_CSV_CACHE, DOWNLOADED_AUDIO_CACHE, RAW_AUDIO_CACHE)
    csv_cache_dir, downloaded_audio_dir, raw_audio_dir = file_utils.prepare_cache_directories(
                                                  subcache_directory_names,
                                                  cache_directory=cache_directory,
                                                  use_cache=use_cache,
                                                  verbose=verbose)

    # Create path to dataset csv.
    path_to_dataset_csv = os.path.join(csv_cache_dir, dataset_csv_filename)

    # If we have decided not to use the cache, download the dataset CSV.
    if not use_cache:
        file_utils.clean_cache_directories(cache_directory=cache_directory)

    # If csv is not in specified location, then throw an error.
    if not file_utils.does_cached_csv_dataset_exist(path_to_dataset_csv):
        if verbose:
            print('Dataset CSV not found at {}. Downloading to location...'.format(path_to_dataset_csv))
        download_csv_dataset(dataset_csv_url, path_to_dataset_csv, verbose=verbose)
    else:
        if verbose:
            print("Using cached copy of dataset csv at {}.".format(path_to_dataset_csv))

    # Read the rows of dataset csv.
    header_row, entries = get_dataset_entries(path_to_dataset_csv)

    # Filter out recordings that have been evaluated and labeled falsely.
    labeled_entries = [entry for entry in entries if convert_to_bool(entry[9]) and convert_to_bool(entry[10])]

    # Download the audio in the dataset.
    for entry in labeled_entries:
        if entry[0] == str(surah_to_download):
            sample_rate_hz = download_entry_audio(entry, downloaded_audio_dir, raw_audio_dir, use_cache=use_cache, verbose=verbose)

    # If we don't want to keep the raw audio, remove it from the cache.
    if not keep_downloaded_audio:
        file_utils.delete_cache_directories(cache_directory=downloaded_audio_dir)
