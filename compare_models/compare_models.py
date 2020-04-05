import argparse
import csv
import datetime
import logging
import os
import random
import sys
import time
from typing import List
from typing import Tuple

import deepspeech
from deepspeech.client import metadata_to_string
from google.cloud import speech
from google.cloud.speech import enums
from Levenshtein import distance
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

# TODO(piraka9011): Append to path until we decide whether tarteel_ml should be a package or not.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import bytes_to_array
from utils.files import load_yaml_config
from utils.files import read_file
from utils.quran_helper import Quran
from utils.recording import get_surah_ayah_number_from_file

LANGUAGE_CODE = 'ar-AE'
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_DIR = 'audio'
client = speech.SpeechClient()
logger = logging.getLogger(__name__)
quran = Quran()


class Result:
    def __init__(self,
                 is_exact_match: bool, is_partial_match: bool, score: float, norm_score: float):
        self.is_exact_match = is_exact_match
        self.is_partial_match = is_partial_match
        self.score = score
        self.norm_score = norm_score


def load_deepspeech_model(args: argparse.Namespace) -> deepspeech.Model:
    config = load_yaml_config(args.ds_config)
    model = deepspeech.Model(config.get('model_dir'), config.get('beam_width'))
    model.enableDecoderWithLM(
      config.get('lm_dir'), config.get('trie_dir'), config.get('lm_alpha'), config.get('lm_beta'))
    return model


def normalize_audio(file: str, sample_rate: int = 16000, channels: int = 1) -> AudioSegment:
    audio = AudioSegment.from_file(file)
    audio = audio.set_channels(channels)
    audio = audio.set_frame_rate(sample_rate)
    return audio


def get_gspeech_result(data: bytes) -> Tuple[str, float]:
    config = {
        'language_code'    : LANGUAGE_CODE,
        'sample_rate_hertz': SAMPLE_RATE,
        'encoding'         : enums.RecognitionConfig.AudioEncoding.LINEAR16
    }
    audio = {'content': data}
    start_time = time.time()
    response = client.recognize(config, audio)
    end_time = time.time()
    response_time = end_time - start_time
    response_result = response.results[0].alternatives[0].transcript if response.results else ''
    return response_result, response_time


def get_deepspeech_result(data: bytes, model: deepspeech.Model) -> Tuple[str, float]:
    audio_array = bytes_to_array(data)
    start_time = time.time()
    response = model.sttWithMetadata(audio_array)
    end_time = time.time()
    response_time = end_time - start_time
    return metadata_to_string(response), response_time


def compare_results(result1: str, result2: str, args: argparse.Namespace) -> Result:
    max_len = max(len(result1), len(result2))
    score = distance(result1, result2)
    # Normalize the score for comparison
    norm_score = (max_len - score) / max_len
    is_exact_match = result1 == result2
    is_partial_match = True if is_exact_match or norm_score >= args.max_score else False
    return Result(is_exact_match, is_partial_match, score, norm_score)


def calculate_results(labels: List[str], results: List) -> List:
    df = pd.DataFrame.from_records(results, columns=labels)
    num_cols = df.shape[0]
    # GoogleSpeech
    gs_exact_match = df['gs_exact_match'].value_counts().get(True, 0.0)
    gs_exact_match_percent = (gs_exact_match / num_cols) * 100.0
    gs_partial_match = df['gs_partial_match'].value_counts().get(True, 0.0)
    gs_partial_match_percent = (gs_partial_match / num_cols) * 100.0
    # DeepSpeech
    ds_exact_match = df['gs_exact_match'].value_counts().get(True, 0.0)
    ds_exact_match_percent = (ds_exact_match / num_cols) * 100.0
    ds_partial_match = df['gs_partial_match'].value_counts().get(True, 0.0)
    ds_partial_match_percent = (ds_partial_match / num_cols) * 100.0
    return [
        ['gs_exact_match_percent', gs_exact_match_percent,
         'ds_exact_match_percent', ds_exact_match_percent],
        ['gs_partial_match_percent', gs_partial_match_percent,
         'ds_partial_match_percent', ds_partial_match_percent],
        ['average_gs_time', df['gs_time'].mean(),
         'average_ds_time', df['ds_time'].mean()],
        ['average_gs_score', df['gs_score'].mean(),
         'average_ds_score', df['ds_score'].mean()],
        ['average_gs_norm_score', df['gs_norm_score'].mean(),
         'average_ds_norm_score', df['ds_norm_score'].mean()]
    ]


def write_results(args: argparse.Namespace, results: List) -> None:
    logger.info('Writing results to {}'.format(args.results_file))
    headers = [
        'file', 'surah', 'ayah',
        'gs_result', 'gs_time', 'gs_exact_match', 'gs_partial_match', 'gs_score', 'gs_norm_score',
        'ds_result', 'ds_time', 'ds_exact_match', 'ds_partial_match', 'ds_score', 'ds_norm_score',
    ]
    data = calculate_results(headers, results)
    with open(args.results_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(results)
        writer.writerows(data)
        writer.writerow(['max_score', args.max_score])


def get_audio_files(args: argparse.Namespace) -> List[str]:
    if args.text_file:
        audio_files = read_file(args.text_file)
    elif args.directory:
        # Only get the files in the top level directory (not recursive)
        _, _, audio_files = next(os.walk(args.directory))
        audio_files = [os.path.join(args.directory, f) for f in audio_files]
    else:
        audio_files = [
            f for f in os.listdir(AUDIO_DIR) if os.path.isfile(os.path.join(AUDIO_DIR, f))
        ]
    if args.random != 0:
        audio_files = random.sample(audio_files, args.random)
    return audio_files


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare ML models performance')
    parser.add_argument('--text-file', type=str,
                        help='Text file with recordings to use')
    parser.add_argument('--directory', type=str,
                        help='Directory with audio files')
    parser.add_argument('--ds-config', type=str, default='ds_config.yaml',
                        help='Path to DeepSpeech configuration yaml file')
    parser.add_argument('--max-score', type=float, default=0.80,
                        help='Max allowed score for a partial match')
    parser.add_argument('--log', type=str, default='info',
                        help='Log level')
    parser.add_argument('--random', type=int, default=0,
                        help='Sample n random files. 0 means no random sampling')
    default_file_name = datetime.datetime.now().strftime('compare-models-%Y-%m-%d_%H_%M') + '.csv'
    parser.add_argument('--results-file', type=str, default=default_file_name,
                        help='CSV file to write results to')
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    if args.directory and args.text_file:
        raise ValueError("Specify either --directory or --text-file but not both.")

    log_level = getattr(logging, args.log.upper(), None)
    logger.setLevel(log_level)

    ds_model = load_deepspeech_model(args)
    audio_files = get_audio_files(args)
    results = []
    for file in tqdm(audio_files):
        logger.debug('Normalizing {}'.format(file))
        audio_data = normalize_audio(file, SAMPLE_RATE, CHANNELS)

        logger.debug('Running GSpeech inference')
        gs_text_result, gs_time = get_gspeech_result(audio_data.raw_data)
        if gs_text_result == '':
            logger.warning('No result received from Google for {}'.format(file))

        logger.debug('Running DeepSpeech inference')
        ds_text_result, ds_time = get_deepspeech_result(audio_data.raw_data, ds_model)
        if ds_text_result == '':
            logger.warning('No result received from DeepSpeech for {}'.format(file))

        surah_number, ayah_number = get_surah_ayah_number_from_file(file)
        quran_text = quran.get_ayah_text(surah_number, ayah_number)
        gs_result = compare_results(gs_text_result, quran_text, args)
        ds_result = compare_results(ds_text_result, quran_text, args)

        logger.debug('Result:\n\tGS Score: {}\n\tDS Score: {}'.format(
          gs_result.norm_score, ds_result.norm_score))
        results.append(
          [file, surah_number, ayah_number,
           gs_text_result, gs_time, gs_result.is_exact_match, gs_result.is_partial_match,
           gs_result.score, gs_result.norm_score,
           ds_text_result, ds_time, ds_result.is_exact_match, ds_result.is_partial_match,
           ds_result.score, ds_result.norm_score]
        )

    write_results(args, results)


if __name__ == '__main__':
    main()
