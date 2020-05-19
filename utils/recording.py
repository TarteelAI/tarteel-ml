"""A file containing utils for dealing with Tarteel recordings.

Author: Hamzah Khan
"""

import logging
import os
from pathlib import Path
import re
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import requests
from urllib.parse import urlparse

DEFAULT_NON_SPEECH_THRESHOLD_FRACTION = 0.5
ALL_SURAHS = None
NUM_SURAHS = 114
NO_FRAMES_VALUE = 1.0


def get_paths_to_all_recordings(local_download_dir):
    """Return a list of paths to all recordings in the given directory."""
    return get_paths_to_surah_recordings(local_download_dir)


def get_paths_to_surah_recordings(local_download_dir: str, surahs: Optional[int] = ALL_SURAHS):
    """ Return a list of paths in the directory to recordings of ayahs for the specified surahs."""
    paths_to_audio = []
    if not os.path.isdir(local_download_dir):
        raise OSError('Local download directory {} not found'.format(local_download_dir))
    if not surahs:
        surahs = 1 + np.arange(NUM_SURAHS)
    for surah_num in surahs:
        local_surah_dir = os.path.join(local_download_dir, "s" + str(surah_num))
        for _, ayah_directories, _ in os.walk(local_surah_dir):
            for ayah_directory in ayah_directories:
                local_ayah_dir = os.path.join(local_surah_dir, ayah_directory)

                for _, _, recording_filenames in os.walk(local_ayah_dir):
                    for recording_filename in recording_filenames:
                        local_audio_path = os.path.join(local_ayah_dir, recording_filename)

                        # Add the fully constructed path name to the list of paths.
                        paths_to_audio.append(local_audio_path)

    return paths_to_audio


def get_paths_to_ayah_recordings(
        local_download_dir: str,
        ayahs: List[Tuple[int, int]]) -> List[str]:
    """Return a list of paths in given directory to recordings of the specified ayahs."""
    paths_to_audio = []

    if not ayahs:
        raise Exception('Invalid list of ayahs - should contain a tuples of surah-ayah pairs.')

    for surah_num, ayah_num in ayahs:
        local_ayah_dir = os.path.join(local_download_dir, "s" + str(surah_num), "a" + str(ayah_num))

        for _, _, recording_filenames in os.walk(local_ayah_dir):
            for recording_filename in recording_filenames:
                local_audio_path = local_audio_path = get_path_to_recording(local_download_dir,
                                                                            surah_num,
                                                                            ayah_num,
                                                                            recording_filename)

                # Add the fully constructed path name to the list of paths.
                paths_to_audio.append(local_audio_path)

    return paths_to_audio


def get_path_to_recording(
        local_download_dir: str,
        surah_num: int,
        ayah_num: int,
        filename: str) -> str:
    """Return the path of a single recording the surah and ayah numbers, and the filename."""
    local_path = os.path.join(
        local_download_dir, "s" + str(surah_num), "a" + str(ayah_num), filename)
    return local_path


def get_path_to_recording_by_id(
        local_download_dir: str,
        surah_num: int,
        ayah_num: int,
        recording_id: int,
        file_extension: str = 'raw') -> str:
    """Return the path of a single recording, the surah and ayah numbers, and the recording id."""
    filename = ("%d_%d_%d.%s" % (surah_num, ayah_num, recording_id, file_extension))
    return get_path_to_recording(local_download_dir, surah_num, ayah_num, filename)


def open_feature_file(path_to_audio: str) -> Any:
    """Opens and returns an .npy file at the passed in path.

    Returns:
        npy_array: A numpy ndarray.
    """
    return np.load(path_to_audio)


def download_recording_from_url(
        url: str,
        download_folder: str,
        use_cache: bool = True) -> str:
    """Downloads the audio from the given URL."""
    # Get wav filename from URL
    parsed_url = urlparse(url)
    wav_filename = os.path.basename(parsed_url.path)

    download_filepath = os.path.join(download_folder, wav_filename)

    # Download audio or use cached copy
    if not os.path.exists(download_filepath) or not use_cache:
        logging.debug('Downloading {}...'.format(wav_filename))
        r = requests.get(url, allow_redirects=True)
        open(download_filepath, 'wb').write(r.content)
    else:
        logging.debug('{} found in cache'.format(wav_filename))

    return download_filepath


def get_surah_ayah_number_from_file(path: str) -> Tuple[int, int]:
    p = Path(path)
    if not p.exists():
        raise ValueError('{} does not exist'.format(path))
    surah_ayah_regex = re.compile(r'^([0-9]+)_([0-9]+)')
    filename = p.stem
    matches = surah_ayah_regex.search(filename)
    if matches:
        return int(matches.group(1)), int(matches.group(2))
    raise ValueError(
      '{} is not of the format <surah_num>_<ayah_num>_<hash>.<file_extension>'.format(filename))


