#!/usr/bin/env python3
"""
A file for downloading audio recordings from tarteel.io.

Originally created by kareemn on Jan. 1, 2019.
https://github.com/kareemn/Tarteel-ML/blob/master/download.py

Modified by Hamzah Khan (khanh111) and added to the tarteel.io/Tarteel-ML on Jan. 12.
"""

import utils.files as file_utils
import utils.recording as recording_utils
import csv
import os
import requests
import wave
import webrtcvad

from urllib.parse import urlparse
from argparse import ArgumentParser

# Define argument constants.
TARTEEL_V1_CSV_URL = "https://d2sf46268wowyo.cloudfront.net/datasets/tarteel_v1.0.csv"
TARTEEL_LIVE_CSV_URL = "https://www.tarteel.io/download-full-dataset-csv"


parser = ArgumentParser(description='Tarteel Audio Downloader')
parser.add_argument('--csv_url', type=str, default=TARTEEL_V1_CSV_URL)
parser.add_argument('--local_csv_filename', type=str, default='local.csv')
parser.add_argument('--cache_dir', type=str, default='.cache')
parser.add_argument('--vad_check', type=bool, default=True)
parser.add_argument('-u', '--use_cache', type=bool, default=False)
parser.add_argument('-v', '--verbose', type=bool, default=False)
parser.add_argument('-s', '--surah', type=int)
args = parser.parse_args()

DATASET_CSV_CACHE = 'csv'
AUDIO_CACHE = 'audio'

# Define constants.
NO_FRAMES_VALUE = 1.0

DEFAULT_NON_SPEECH_THRESHOLD_FRACTION = 0.5

WEBRTCVAD_SUPPORTED_SAMPLE_RATES_HZ = [8000, 16000, 32000, 48000]

def does_cached_csv_dataset_exist(path_to_dataset_csv):
    return os.path.isfile(path_to_dataset_csv)

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
    A function to parse and return the rows of the dataset csv.
    """
    file = open(path_to_dataset_csv, "r")
    content = file.read()
    csv_data = csv.reader(content.splitlines(), delimiter=',')
    entries = list(csv_data)
    return entries

def download_audio(row, local_download_dir, verbose=False):
    """
    Downloads an audio recording given its entry in a Tarteel dataset csv.
    """
    surah_num = int(row[0])
    ayah_num = int(row[1])
    url = row[2]
    parsed_url = urlparse(url)
    wav_filename = os.path.basename(parsed_url.path)
    local_download_path = recording_utils.get_path_to_recording(local_download_dir, surah_num, ayah_num, wav_filename)

    # Download and save the audio recording to the given path.
    try:
        with requests.Session() as s:
            if verbose:
                print("Downloading", url, "to", local_download_path, ".")
            download = s.get(url)
            dirname = os.path.dirname(local_download_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            file = open(local_download_path, "wb")
            file.write(download.content)
            file.close()

    except:
        # If the download fails, print an error and exit the function.
        print("Audio file", local_download_path, "could not be opened.")
        return

    # Check if the wave header is valid and if so, get the desired info. This deletes files with invalid headers.
    has_valid_header, wav_bytes, sample_rate_hz, num_channels = recording_utils.open_recording(local_download_path)

    if has_valid_header:
        # Note: webrtcVAD does not currently support 44.1MHz, so we have no way of checking those files for empty audio.
        recording_has_speech = recording_utils.has_speech(
            wav_bytes, 
            sample_rate_hz, 
            num_channels, 
            non_speech_threshold_fraction=DEFAULT_NON_SPEECH_THRESHOLD_FRACTION, 
            verbose=args.verbose)

        if args.vad_check and sample_rate_hz in WEBRTCVAD_SUPPORTED_SAMPLE_RATES_HZ and not recording_has_speech:
            print("Audio file", local_download_path, "does not have speech according to VAD. Removing.")
            os.remove(local_download_path)

    if sample_rate_hz not in WEBRTCVAD_SUPPORTED_SAMPLE_RATES_HZ+[44100] and sample_rate_hz != None:
        # If unsupported sample frequency, remove the file.
        print("File ", local_download_path, " has an unsupported sample frequency %d. Removing." % sample_rate_hz)
        if os.path.exists(local_download_path):
            os.remove(local_download_path)


if __name__ == "__main__":

    # Parse the arguments.
    cache_directory = args.cache_dir
    dataset_csv_url = args.csv_url
    dataset_csv_filename = args.local_csv_filename
    use_cache = args.use_cache
    verbose = args.verbose
    surah_to_download = args.surah

    # Prepare all requisite cache directories.
    subcache_directory_names = (DATASET_CSV_CACHE, AUDIO_CACHE)
    csv_cache_dir, audio_cache_dir = file_utils.prepare_cache_directories(subcache_directory_names,
                                                                          cache_directory=cache_directory,
                                                                          use_cache=use_cache,
                                                                          verbose=verbose)

    # Create path to dataset csv.
    path_to_dataset_csv = os.path.join(csv_cache_dir, dataset_csv_filename)

    # If we have decided not to use the cache, download the dataset CSV.
    if not use_cache:
        file_utils.clean_cache_directories(cache_directory=cache_directory)

    # If csv is not in specified location, then throw an error.
    if not does_cached_csv_dataset_exist(path_to_dataset_csv):
        if verbose:
            print('Dataset CSV not found at {}. Downloading to location...'.format(path_to_dataset_csv))
        download_csv_dataset(dataset_csv_url, path_to_dataset_csv, verbose=verbose)
    else:
        if verbose:
            print("Using cached copy of dataset csv at {}.".format(path_to_dataset_csv))

    # Read the rows of dataset csv.
    entries = get_dataset_entries(path_to_dataset_csv)

    # Download the audio in the dataset.
    for entry in entries:
        if entry[0] == str(surah_to_download):
            sample_rate_hz = download_audio(entry, audio_cache_dir, verbose=verbose)
