"""Create a CSV files for audio files formatted for DeepSpeech.

DeepSpeech format is:

| wav_filename | wav_filesize     | transcript |
| ------------ | ---------------- | ---------- |
| ...          | ...              | ...        |
"""

import argparse
import logging
import os
from pathlib import Path

from tqdm import tqdm
from typing import List, Tuple

from utils import files
from utils import quran_helper

DEEPSPEECH_FILENAME_HEADER = 'wav_filename'
DEEPSPEECH_FILESIZE_HEADER = 'wav_filesize'
DEEPSPEECH_TRANSCRIPT_HEADER = 'transcript'
DEEPSPEECH_CSV_HEADERS = [DEEPSPEECH_FILENAME_HEADER, DEEPSPEECH_FILESIZE_HEADER, DEEPSPEECH_TRANSCRIPT_HEADER]
DEEPSPEECH_CSV_FILENAME = 'tarteel_deepspeech_full.csv'

parser = argparse.ArgumentParser(description='Prepare a CSV file for Deepspeech')
parser.add_argument(
    '-i', '--audio-directory', type=str, required=True,
    help='Path to directory with audio files .'
)
parser.add_argument(
    '-o', '--output-directory', type=str,
    help='Output directory for CSV and alphabet.txt.'
)
parser.add_argument(
    '--filename', type=str, default=DEEPSPEECH_CSV_FILENAME,
    help='Output CSV filename.'
)
parser.add_argument(
    '--log', choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL'], default='INFO',
    help='Logging level.'
)
args = parser.parse_args()
numeric_level = getattr(logging, args.log, None)
logging.basicConfig(level=numeric_level)
quran = quran_helper.Quran()


def check_args() -> Tuple[str, str]:
    audio_directory = Path(args.audio_directory)
    output_directory = Path(args.output_directory)
    if not audio_directory.is_dir():
        raise ValueError("Audio directory is not a valid directory.")
    if not output_directory.is_dir():
        raise ValueError("Output directory is not a valid directory.")
    logging.info(
        "Parameters:\nAudio Directory: {}\nOutput Directory: {}".format(
            audio_directory, output_directory))
    return audio_directory, output_directory


def get_surah_ayah_from_file(filename: str) -> Tuple[int, int]:
    split_filename = filename.split('_')
    surah_number = int(split_filename[0])
    ayah_number = int(split_filename[1])
    return surah_number, ayah_number


def create_csv_file(file_names: List) -> List:
    csv_rows = [DEEPSPEECH_CSV_HEADERS]
    for wav_filename in tqdm(file_names):
        wav_filename = wav_filename.strip()  # Remove trailing characters
        file_path = Path(os.path.join(args.audio_directory, wav_filename))
        file_size = files.get_file_size(file_path.as_posix())
        if not file_size:
            logging.warning('Could not get {} file size skipping...'.format(wav_filename))
            continue

        surah_number, ayah_number = get_surah_ayah_from_file(wav_filename)
        text = quran.get_ayah_text(surah_number, ayah_number)
        csv_row = [file_path.as_posix(), file_size, text]
        csv_rows.append(csv_row)

    return csv_rows


def main():
    audio_directory, output_directory = check_args()  # Throws if invalid args

    file_names = files.get_all_files_in_directory(audio_directory)
    csv_rows, unique_alphabet = create_csv_file(file_names)

    csv_file_path = os.path.join(output_directory, args.filename)
    files.write_csv(csv_file_path, csv_rows)


if __name__ == '__main__':
    main()
