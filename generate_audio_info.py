"""Create a CSV file with info on all audio files in a directory."""

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

from pydub.utils import mediainfo_json
from tqdm import tqdm

from utils import files

parser = argparse.ArgumentParser(
  description='Create a CSV file with info on all audio files in a directory.')
parser.add_argument(
    '-i', '--input-directory', type=str
)
parser.add_argument(
    '-o', '--output-directory', type=str
)
args = parser.parse_args()

CSV_HEADERS = ['filename', 'duration', 'size', 'format', 'samplerate', 'channels', 'bits', 'codec']
CSV_FILENAME = 'audio_info.csv'


def check_args() -> Tuple[str, str]:
    input_directory = Path(args.input_directory)
    if not input_directory.is_dir():
        raise ValueError("Input directory is not a valid directory")
    output_directory = Path(args.output_directory)
    if not output_directory.is_dir():
        raise ValueError("Output directory is not a valid directory")
    logging.info(
        "Parameters\nAudio Directory: {}\nOutput Directory: {}".format(
            input_directory, output_directory))
    return input_directory, output_directory


def get_entry_from_mediainfo(filepath: str):
    info = mediainfo_json(filepath)
    stream = info.get('streams')[0]
    media_format = info.get('format')
    entry = [
        filepath,
        media_format.get('duration'),
        media_format.get('size'),
        media_format.get('format_name'),
        stream.get('sample_rate'),
        stream.get('channels'),
        stream.get('bits_per_sample'),
        stream.get('codec_name')
    ]
    return entry


def main():
    input_directory, output_directory = check_args()
    filename_list = files.get_all_files_in_directory(input_directory)
    csv_rows = [CSV_HEADERS]

    for filename in tqdm(filename_list):
        filepath = os.path.join(input_directory, filename)
        entry = get_entry_from_mediainfo(filepath)
        csv_rows.append(entry)

    csv_filename = os.path.join(output_directory, CSV_FILENAME)
    files.write_csv(csv_filename, csv_rows)


if __name__ == '__main__':
    main()
