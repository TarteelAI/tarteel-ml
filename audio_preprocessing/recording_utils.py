"""
A file containing utils for dealing with Tarteel audio recordings.

Author: Hamzah Khan
Date: Jan. 13, 2019
"""

import numpy as np
import os
import wave
import webrtcvad

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

def get_path_to_recording_by_id(local_download_dir, surah_num, ayah_num, recording_id, file_extension='wav'):
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

def has_speech(wav_bytes, 
               sample_rate_hz, 
               num_channels, 
               non_speech_threshold_fraction=DEFAULT_NON_SPEECH_THRESHOLD_FRACTION,
               verbose=False):
    """
    Returns true if at least (1 - non_speech_threshold_fraction) percentage of frames contain voice activity.
    Note: webrtc VAD does not currently support 44.1MHz, so we have no way of checking those files for empty audio.
    """

    # Use webrtc's VAD with the lowest level of aggressiveness.
    mono_channel_bytes = wav_bytes

    if num_channels == 2:
        # just take the left channel for simplicity purposes.
        # We're just trying to get a quick sanity check, no need
        # to mix the two channels.
        mono_channel_bytes = b"".join([wav_bytes[i:i+2] for i in range(0, len(wav_bytes), 4)])

    vad = webrtcvad.Vad(1)
    frame_duration = 10  # ms
    bytes_per_sample = 2 # assuming 16-bit PCM.
    samples_per_vaded_chunk = (sample_rate_hz * frame_duration / 1000)
    bytes_per_vaded_chunk = int(samples_per_vaded_chunk*bytes_per_sample)
    num_speech_frames = 0
    num_non_speech_frames = 0

    for i in range(0, len(mono_channel_bytes)-bytes_per_vaded_chunk, bytes_per_vaded_chunk):
        chunk_to_vad = mono_channel_bytes[i:i+bytes_per_vaded_chunk]
        vad_frame_length = int(len(chunk_to_vad) / bytes_per_sample)
        if (webrtcvad.valid_rate_and_frame_length(sample_rate_hz, vad_frame_length)
            and vad.is_speech(chunk_to_vad, sample_rate_hz)):
            num_speech_frames += 1
        else:
            num_non_speech_frames += 1

    has_frames = (num_speech_frames + num_non_speech_frames > 0)
    emptyAudio = (num_speech_frames == 0 or (num_speech_frames and num_non_speech_frames == 0))

    if has_frames:
        percentage_non_speech = (float(num_non_speech_frames) / float(num_non_speech_frames+num_speech_frames))
    else:
        # If there are no frames, return a default (positive > 0.5) number.
        percentage_non_speech = NO_FRAMES_VALUE

    if verbose:
        print ("percentage non-speech:", percentage_non_speech,
               "num_speech_frames", num_speech_frames,
               "num_non_speech_frames", num_non_speech_frames)

    return not emptyAudio and percentage_non_speech < non_speech_threshold_fraction
