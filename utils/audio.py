"""Utility functions for processing audio."""

import logging
import os
import subprocess
from typing import Optional
from typing import Tuple

import librosa
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import mediainfo_json
import soundfile as sf
import tensorflow as tf

DEFAULT_SAMPLE_RATE = 16000  # 44100 for Google STT, 16000 for DeepSpeech


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


class AudioFile:
    def __init__(self, filepath: str):
        self._filepath = filepath
        self.duration = None
        self.size = None
        self.format_name = None
        self.sample_rate = None
        self.channels = None
        self.bits = None
        self.codec = None
        self.data = None
        self._deleted = False
        self._get_info()

    def _get_info(self):
        info = mediainfo_json(self._filepath)
        stream = info.get('streams')[0]
        media_format = info.get('format')
        self.duration = float(media_format.get('duration'))
        self.size = int(media_format.get('size'))
        self.format_name = media_format.get('format_name')
        self.sample_rate = int(stream.get('sample_rate'))
        self.channels = stream.get('channels')
        self.bits = stream.get('bits_per_sample')
        self.bit_rate = int(media_format.get('bit_rate'))
        self.code = stream.get('codec_name')
        try:
            self.data, _ = librosa.load(self._filepath)
        except ValueError as e:
            if 'Input signal length=0' in str(e):
                logging.info("Found empty audio file.")
                self.data = None

    def is_wav(self):
        return self.format_name == 'wav'

    def is_empty(self):
        return (self.duration is None) or (self.size == 0)

    def is_deleted(self):
        return self._deleted

    def resample(self, sample_rate: int = DEFAULT_SAMPLE_RATE, inplace: bool = False):
        self.data, self.sample_rate = librosa.load(self._filepath, sample_rate)
        if not inplace:
            sf.write(self._filepath, self.data, sample_rate)

    def play(self):
        to_play = AudioSegment.from_file(self._filepath)
        play(to_play)

    def save_as(self,
                output_file: Optional[str] = None,
                sample_rate: Optional[int] = None,
                output_format: Optional[str] = None,
                subtype: str = 'PCM_16') -> None:
        output_path = output_file if output_file else self._filepath
        sr = sample_rate if sample_rate else self.sample_rate
        fmt = output_format if output_format else self.format_name
        sf.write(output_path, self.data, sr, subtype, format=fmt)

    def delete(self):
        os.remove(self._filepath)
        self._deleted = True

    def load_with_tf(self) -> Optional[Tuple[tf.Tensor, tf.Tensor]]:
        """Opens a wave file using Tensorflow.

        Used to check if it's possible to read a wave file the the TF library.
        Throws on any errors.
        """
        if self._deleted:
            logging.warning("File is deleted.")
            return

        audio_binary = tf.io.read_file(self._filepath)
        return tf.audio.decode_wav(audio_binary, desired_channels=self.channels)
