"""Utility functions for processing audio."""
__author__ = 'Hamzah Khan'

import logging
import os
import re
import subprocess
from typing import Optional
from typing import Tuple
import wave

import librosa
import soundfile as sf
import tensorflow as tf


DEFAULT_AUDIO_FORMAT = 'raw'
DEFAULT_SAMPLE_RATE = 44100  # 44100 for Google STT, 16000 for DeepSpeech


def detect_audio_type(audio_filepath: str) -> str:
    """Returns the type of audio encoding. 
    
    Currently uses ffprobe to get this information.

    Args:
        audio_filepath: Absolute path to the audio file to process.

    Returns:
        audio_type: The audio encoding type.
    """
    command_args = ['ffprobe', '-hide_banner', audio_filepath]
    ffprobe_output = subprocess.check_output(command_args, stderr=subprocess.STDOUT, universal_newlines=True)
    m = re.search(r'Audio:\s(\w+)\s', ffprobe_output)

    if m:
        audio_type = m.group(1)
    else:
        audio_type = 'unknown'

    return audio_type


def convert_to_wav(
        input_file: str,
        output_file: str,
        sample_rate: int = 16000) -> bool:
    """Convert a file to wav format with librosa.

    Deletes empty audio files.

    Args:
        input_file: File to convert
        output_file: File to write to
        sample_rate: Desired audio sample rate in Hz (independent of original format)

    Returns:
        result: True if wrote successfully. False otherwise
    """
    try:
        x, sr = librosa.load(input_file, sr=sample_rate)
        sf.write(output_file, x, sr)
        return True
    except ValueError as e:
        if 'Input signal length=0' in str(e):
            logging.info("Found empty audio file. Removing...")
            os.remove(input_file)
            return False


def convert_to_raw(input_file: str, bitrate: int, sample_rate: int, output_file: str) -> str:
    """Convert audio file to `raw` format with sox CLI."""
    command_args = ['sox', input_file,
                    '-t', 'raw',             # Output type (raw)
                    '-b', str(bitrate),      # Bitrate
                    '-e', 'signed',          # Integer Encoding
                    '-r', str(sample_rate),  # Sampling Rate
                    '-c', '1',               # Number of channels (mono)
                    output_file]
    # TODO(piraka9011) Does the output of this need to be validated?
    return subprocess.check_output(command_args, stderr=subprocess.STDOUT, universal_newlines=True)


def open_wave_file_tf(path_to_audio: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """Opens a wave file using Tensorflow.

    Used to check if it's possible to read a wave file the the TF library.
    Throws on any errors.
    """
    audio_binary = tf.io.read_file(path_to_audio)
    return tf.audio.decode_wav(audio_binary, desired_channels=1)


def open_wave_file(path_to_audio: str) -> Tuple[bool, Optional[bytes], Optional[int], Optional[int]]:
    """Check if the audio file has a proper wave header.

    Proper is defined as whether the Python `wave` module can open the file.

    Returns:
        (False, None, None, None) if the header is invalid.
        (True, wav_bytes, sample_rate_hz, num_channels) if header is valid.
    """
    # TODO(piraka9011) Refactor to use wf.getparams() instead and return two objects instead.
    try:
        with wave.open(path_to_audio, 'rb') as wf:
            wav_bytes = wf.readframes(wf.getnframes())
            sample_rate_hz = wf.getframerate()
            num_channels = wf.getnchannels()
            return True, wav_bytes, sample_rate_hz, num_channels
    except wave.Error as e:
        logging.warning(
            "File with invalid wave header found:\n{}\nException: {}".format(path_to_audio, e))
        return False, None, None, None


def save_wave_file(
        filename: str,
        audio_data: bytes,
        sample_rate_hz: int,
        num_channels: int = 1) -> None:
    """Write a WAVE file to disk."""
    with wave.open(filename, 'wb') as file:
        file.setnchannels(num_channels)
        file.setframerate(sample_rate_hz)
        file.setsampwidth(2)
        file.writeframes(audio_data)
