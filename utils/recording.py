"""
A file containing utils for dealing with Tarteel recordings.

Author: Hamzah Khan
Date: Jan. 13, 2019
"""

import numpy as np
import os
import requests
from urllib.parse import urlparse
import wave

DEFAULT_NON_SPEECH_THRESHOLD_FRACTION = 0.5
ALL_SURAHS = None
NUM_SURAHS = 114
NO_FRAMES_VALUE = 1.0

def get_paths_to_all_recordings(local_download_dir):
    """
    Returns a list of paths to all recordings in the given directory.
    """
    return get_paths_to_surah_recordings(local_download_dir)

def get_paths_to_surah_recordings(local_download_dir, surahs=ALL_SURAHS):
    """
    Returns a list of paths, in the given directory, to recordings of ayahs in the specified surahs.
    """
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

def get_paths_to_ayah_recordings(local_download_dir, ayahs):
    """
    Returns a list of paths, in the given directory, to recordings of the specified ayahs.
    """
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

def get_path_to_recording(local_download_dir, surah_num, ayah_num, filename):
    """
    Returns the path of a single recording, given the local_download_dir, the surah and ayah numbers, and the filename.
    """
    local_path = os.path.join(local_download_dir, "s" + str(surah_num), "a" + str(ayah_num), filename)
    return local_path

def get_path_to_recording_by_id(local_download_dir, surah_num, ayah_num, recording_id, file_extension='raw'):
    """
    Returns the path of a single recording, given the local_download_dir, the surah and ayah numbers, and the recording
    id.
    """
    filename = ("%d_%d_%d.%s" % (surah_num, ayah_num, recording_id, file_extension))
    return get_path_to_recording(local_download_dir, surah_num, ayah_num, filename)

def open_recording(path_to_audio):
    """
    Returns a tuple (True, audio frames, sample rate in hz, number of channels) if the audio at the given path has a
    proper wave header. Returns (False, None, None, None) if the header is invalid.

    Note: As of now, proper is defined as whether the wave library can open the file.
    """
    # Check for valid WAVE header.
    try:
        wf = wave.open(path_to_audio, "rb")
        wav_bytes = wf.readframes(wf.getnframes())
        sample_rate_hz = wf.getframerate()
        num_channels = wf.getnchannels()
        wf.close()
        return (True, wav_bytes, sample_rate_hz, num_channels)
        
    # If wave can not load the file, print an error and exit the function.
    except wave.Error:
        print("Invalid wave header found", path_to_audio, ", removing.")

        if os.path.exists(path_to_audio):
            os.remove(path_to_audio)

        return (False, None, None, None)

def open_feature_file(path_to_audio):
    """
    Opens and returns the .npy file at the passed in path and returns a numpy ndarray of the feature in question and
    processes it.
    """
    return np.load(path_to_audio)

def download_recording_from_url(url, download_folder, use_cache=True, verbose=False):
    """
    Downloads the audio from the given URL.
    """
    # Get wav filename from URL
    parsed_url = urlparse(url)
    wav_filename = os.path.basename(parsed_url.path)

    download_filepath = os.path.join(download_folder, wav_filename)

    # Download audio or use cached copy
    if not os.path.exists(download_filepath) or not use_cache:
        if verbose:
            print('Downloading %s...' % (wav_filename))
        r = requests.get(url, allow_redirects=True)
        open(download_filepath, 'wb').write(r.content)
    else:
        if verbose:
            print('%s found in cache' % (wav_filename))

    return download_filepath
