"""Utility functions for processing audio."""
__author__ = 'Hamzah Khan'

import os
import re
import subprocess

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

def convert_audio(audio_filepath: str, output_folder: str, bitrate: int = 16, 
                  sample_rate: int = 44100, use_cache: bool = True, verbose: bool = False) -> str:
    """Converts the given audio into raw audio with the given bitrate and sample rate. 
    
    Defaults are set to match Google Speech Recognition's requirements.

    Args:
        audio_filepath: Relative file name of the audio file to process.
        output_folder: Directory to put raw audio in
        bitrate: Audio bitrate
        sample_rate: Audio sampling rate in Hz
        use_cache: Use the audio file already in the output directory if found
        verbose: Talk a lot.
    
    Returns:
        raw_audio_path: Absolute filepath to the converted audio file.
    """
    audio_filename = os.path.basename(audio_filepath)
    raw_audio_filename = audio_filename[:-4] + '.raw'
    raw_audio_path = os.path.join(output_folder, raw_audio_filename)

    # Check if file is already available in cache.
    if os.path.exists(raw_audio_path) and use_cache:
        if verbose:
            print('%s found in cache' % (raw_audio_filename))
        return raw_audio_path

    # If not using cache, remove if file already exists.
    if os.path.exists(raw_audio_path):
        os.remove(raw_audio_path)

    # Check audio type
    audio_type = detect_audio_type(audio_filepath)

    # If file is not of type wav (pcm_s16le), convert to wav using ffmpeg.
    if audio_type != 'pcm_s16le':
        if verbose:
            print('Audio type is %s. Converting to wav...' % (audio_type))
        output_audio_path = os.path.join(output_folder, audio_filename)

        # Remove if file already exists
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)

        command_args = ['ffmpeg', '-i', audio_filepath, output_audio_path]
        ffmpeg_output = subprocess.check_output(command_args, stderr=subprocess.STDOUT, universal_newlines=True)
        matches = re.findall(r'Audio:\s(\w+)\s', ffmpeg_output)

        assert len(matches) == 2, 'ffmpeg conversion failed.'
        assert matches[1] == 'pcm_s16le', 'ffmpeg conversion failed.'

        wav_filepath = output_audio_path
    else:
        if verbose:
            print('Audio type is wav')
        wav_filepath = audio_filepath

    # Convert wav to raw audio
    if verbose:
        print('Converting to raw audio')
    command_args = ['sox', wav_filepath,
                    '-t', 'raw',             # Output type (raw)
                    '-b', str(bitrate),      # Bitrate
                    '-e', 'signed',          # Integer Encoding
                    '-r', str(sample_rate),  # Sampling Rate
                    '-c', '1',               # Number of channels (mono)
                    raw_audio_path]
    sox_output = subprocess.check_output(command_args, stderr=subprocess.STDOUT, universal_newlines=True)

    # Remove intermediate wav file
    if audio_type != 'pcm_s16le':
        os.remove(wav_filepath)

    return raw_audio_path
