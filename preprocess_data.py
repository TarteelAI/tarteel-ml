"""Use this script to make sure all the audio files are valid and cleaned up."""

import argparse
from multiprocessing import Pool
import os
from pathlib import Path
import logging
import warnings

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
    '--sample-rate', type=int, default=16000, choices=[48000, 44100, 16000],
    help='Set the sample rate of the output audio files'
)
parser.add_argument(
    '--audio-format', type=str, default='wav', choices=['wav', 'raw'],
    help='Audio file output format'
)
parser.add_argument(
    '--log', choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL'], default='INFO',
    help='Logging level.'
)
args = parser.parse_args()

numeric_level = getattr(logging, args.log, None)
logging.basicConfig(level=numeric_level)
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")


class MtAudioProcessor:
    def __init__(self,
                 audio_directory: str,
                 output_directory: str,
                 use_cache: bool,
                 num_processes: int = 8):
        self._audio_directory = audio_directory
        if not Path(audio_directory).is_dir():
            raise ValueError("Audio directory is not a valid directory")
        self._output_directory = output_directory
        if not Path(output_directory).is_dir():
            raise ValueError("Output directory is not a valid directory")
        self._use_cache = use_cache
        self._num_processes = num_processes
        self._filenames = files.get_all_files_in_directory(self._audio_directory)
        self._bad_files = []
        self._deleted_files = []

    def process_file(self, filename: str):
        input_filepath = os.path.join(self._audio_directory, filename)
        new_filepath = os.path.join(self._output_directory, filename)

        if Path(new_filepath).exists() and self._use_cache:
            return

        audio_file = audio.AudioFile(input_filepath)
        if audio_file.is_empty():
            audio_file.delete()
            self._deleted_files.append(input_filepath)
            return

        if audio_file.format_name != args.audio_format:
            audio_file.save_as(output_file=new_filepath, output_format=args.audio_format, sample_rate=16000)
        else:
            audio_file.save_as(output_file=new_filepath)
        return

    def run(self):
        with Pool(self._num_processes) as p:
            r = list(tqdm(
                p.imap_unordered(self.process_file, self._filenames), total=len(self._filenames)))

        if self._deleted_files:
            logging.info("Deleted the following empty files:")
            logging.info("\n".join(self._deleted_files))


if __name__ == '__main__':
    logging.info("Parameters\n")
    for k, v in vars(args).items():
        logging.info("{}: {}".format(k, v))
    logging.info("Starting audio pre-processing...")
    mtad = MtAudioProcessor(
        args.audio_directory, args.output_directory, args.use_cache)
    mtad.run()
