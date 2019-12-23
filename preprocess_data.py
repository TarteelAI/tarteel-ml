"""Use this script to make sure all the audio files are valid and cleaned up."""

import argparse
import os
from pathlib import Path
import logging
from typing import Tuple

from tqdm import tqdm

from utils import audio
from utils import files

EPILOG = 'Pre-process audio files in a given directory.\n\n' \
         'Given an audio directory, iterate through all files and make sure that:' \
         '\t1. Files are not empty' \
         '\t2. Files are in the 16 bit WAVE format with a sample rate of 16000 Hz' \
         'Resulting files are placed in the output directory.'

parser = argparse.ArgumentParser(epilog=EPILOG)
parser.add_argument(
    '-i', '--audio-directory', type=str,
    help='Path to directory with audio files.'
)
parser.add_argument(
    '-o', '--output-directory', type=str,
    help='Path to put resulting files in.'
)
parser.add_argument(
    '-c', '--use-cache', action='store_true',
    help='Don\'t overwrite files in output directory and use cached files.'
)
parser.add_argument(
    '--log', choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL'], default='INFO',
    help='Logging level.'
)
args = parser.parse_args()

numeric_level = getattr(logging, args.log, None)
logging.basicConfig(level=numeric_level)


def check_args() -> Tuple[str, str]:
    audio_directory = Path(args.audio_directory)
    if not audio_directory.is_dir():
        raise ValueError("Audio directory is not a valid directory")
    output_directory = Path(args.output_directory)
    if not output_directory.is_dir():
        raise ValueError("Output directory is not a valid directory")
    logging.info("Parameters")
    return audio_directory, output_directory


def main():
    audio_directory, output_directory = check_args()
    filenames = files.get_all_files_in_directory(audio_directory)
    bad_files = []
    deleted_files = []

    with tqdm(filenames) as pbar:
        for filename in pbar:
            filepath = os.path.join(audio_directory, filename)
            new_filepath = os.path.join(output_directory, filename)

            if Path(new_filepath).exists() and args.use_cache:
                pbar.set_postfix_str("Using cache {}".format(filename))
                continue

            is_valid_file, wav_bytes, sample_rate_hz, _ = audio.open_wave_file(filepath)

            if not is_valid_file:
                pbar.set_postfix_str("Attempting to fix audio file {}".format(filename))
                try:
                    result = audio.convert_to_wav(filepath, new_filepath)
                    if result:
                        is_valid_file, wav_bytes, sample_rate_hz, _ = \
                            audio.open_wave_file(new_filepath)
                        assert is_valid_file
                    else:
                        deleted_files.append(filename)
                        continue
                except AssertionError:
                    pbar.set_postfix_str("Unable to fix {}. Skipping...".format(filename))
                    bad_files.append(filename)
                    continue

            audio.save_wave_file(new_filepath, wav_bytes, sample_rate_hz)

    if deleted_files:
        logging.info("Deleted the following empty files:")
        logging.info("\n".join(deleted_files))
    if bad_files:
        logging.warning("Unable to fix the following files:")
        logging.warning("\n".join(bad_files))


if __name__ == '__main__':
    main()
