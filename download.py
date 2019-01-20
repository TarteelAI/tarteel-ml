#!/usr/bin/env python3
"""
A file for downloading audio recordings from tarteel.io.

Originally created by kareemn on Jan. 1, 2019.
https://github.com/kareemn/Tarteel-ML/blob/master/download.py

Modified by Hamzah Khan (khanh111) and added to the tarteel.io/Tarteel-ML on Jan. 12.
"""

import audio_preprocessing.recording_utils as recording_utils
import csv
import os
import requests
import wave
import webrtcvad

from urllib.parse import urlparse
from argparse import ArgumentParser

# Define argument constants.
TARTEEL_V1_CSV_URL = "https://www.tarteel.io/static/datasets/tarteel_v1.0.csv"
TARTEEL_LIVE_CSV_URL = "https://www.tarteel.io/download-full-dataset-csv"


parser = ArgumentParser(description='Tarteel Audio Downloader')
parser.add_argument('--csv_url', type=str, default=TARTEEL_V1_CSV_URL)
parser.add_argument('--local_csv_cache', type=str, default='.cache/local.csv')
parser.add_argument('--local_download_dir', type=str, default='.audio')
parser.add_argument('--vad_check', type=bool, default=True)
parser.add_argument('--verbose', type=bool, default=False)
parser.add_argument('-s', '--surah', type=int)
args = parser.parse_args()


# Define constants.
NO_FRAMES_VALUE = 1.0

DEFAULT_NON_SPEECH_THRESHOLD_FRACTION = 0.5

WEBRTCVAD_SUPPORTED_SAMPLE_RATES_HZ = [8000, 16000, 32000, 44100, 48000]

def download_csv_dataset():
    print("Downloading CSV from", args.csv_url)
    with requests.Session() as s:
        download = s.get(args.csv_url)

        os.makedirs(os.path.dirname(args.local_csv_cache))
        decoded_content = download.content.decode('utf-8')
        file = open(args.local_csv_cache, "w")
        file.write(decoded_content)
        file.close()
        print ("Done downloading CSV.")

def parse_csv():
    file = open(args.local_csv_cache, "r")
    content = file.read()
    cr = csv.reader(content.splitlines(), delimiter=',')
    rows = list(cr)
    return rows

def cached_csv_exists():
    return os.path.isfile(args.local_csv_cache)

def download_audio(row, local_download_dir):
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
            if args.verbose:
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

    if sample_rate_hz not in WEBRTCVAD_SUPPORTED_SAMPLE_RATES_HZ and sample_rate_hz != None:
        # If unsupported sample frequency, remove the file.
        print("File ", local_download_path, " has an unsupported sample frequency %d. Removing." % sample_rate_hz)
        if os.path.exists(local_download_path):
            os.remove(local_download_path)

if __name__ == "__main__":

    if not cached_csv_exists():
        download_csv_dataset()
    else:
        print("Using cached copy of csv at", args.local_csv_cache)

    rows = parse_csv()

    for row in rows:
        if row[0] == str(args.surah):
            sample_rate_hz = download_audio(row, args.local_download_dir)
        # else:
        #     download_audio(row)
