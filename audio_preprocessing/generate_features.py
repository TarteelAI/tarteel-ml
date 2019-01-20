#!/usr/bin/env python3
"""
A file for preprocessing audio recordings from tarteel.io for input into
TensorFlow models.

Filter bank and MFCC background referenced from
[1] https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
and
http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/.

Tensorflow implementation details inspired by API at
https://www.tensorflow.org/api_guides/python/contrib.signal.

Author: Hamzah Khan
Date: Jan. 12, 2019
"""

import functools
import json
import numpy as np
import os
import recording_utils
import scipy.io.wavfile
import tensorflow as tf
from tensorflow.contrib import signal as tf_signal

from argparse import ArgumentParser

# Argument constants.
ALL_SURAHS = 0
NUM_SURAHS = 114

OUTPUT_MFCC = "mfcc"
OUTPUT_MEL_FILTER_BANK = "mel_filter_bank"
OUTPUT_LOG_MEL_FILTER_BANK = "log_mel_filter_bank"

# Set up parameters.
parser = ArgumentParser(description='Tarteel Audio Recording Tensorizer')
parser.add_argument('-f', '--format', help="Valid options are 'mfcc' or 'mel_filter_bank'.")
parser.add_argument('-o', '--output_dir', type=str, default='.outputs', help="A path to an output directory.")
parser.add_argument('-s', '--surah', type=int, help="Select an integer in [1, 114] to specify surahs with recordings" \
                                                    "to tensorize. None downloads all surahs.", default=ALL_SURAHS)
parser.add_argument('-a', '--ayah', type=int, help="Select an integer to specify an ayah in the surah with recordings" \
                                                    "to tensorize. None ignores this argument.", default=None)
parser.add_argument('--local_download_dir', type=str, default='.audio')
parser.add_argument('--local_csv_cache', type=str, default='.cache/local.csv')
parser.add_argument('-v', '--verbose', type=bool, default=False)
args = parser.parse_args()


# Define constants.

# Unsupported sampling frequencies.
SUPPORTED_FREQUENCIES = [8000, 16000, 32000, 44100, 48000]

# Select a pre_emphasis coefficient. 
"""
Typical values for the [pre-emphasis] filter coefficient are 0.95 or 0.97.
"""
pre_emphasis_factor_1 = 0.95
pre_emphasis_factor_2 = 0.97
PRE_EMPHASIS_FACTOR = pre_emphasis_factor_1

# Select the frame splitting constants. Note that frames can and should overlap.
"""
"Typical frame sizes in speech processing range from 20 ms to 40 ms with 50% (+/-10%) overlap between consecutive
frames. Popular settings are 25 ms for the frame size, frame_size = 0.025 and a 10 ms stride (15 ms overlap),
frame_stride = 0.01" [1].
"""
FRAME_SIZE_S = 0.025
FRAME_STRIDE_S = 0.01

"""
For the "Short-Time Fourier-Transform (STFT) (over N points),... N is typically 256 or 512." [1]
"""
# Select the number of points used in the Short-Time Fourier Transform.
STFT_NUM_POINTS_1 = 256
STFT_NUM_POINTS_2 = 512
STFT_NUM_POINTS = STFT_NUM_POINTS_2


"""
"typically 40 filters... The Mel-scale aims to mimic the non-linear human ear perception of sound,
by being more discriminative at lower frequencies and less discriminative at higher frequencies.
[1]"
"""
# Select the number of triangular filters to apply to the power spectrum for frequency band extraction.
NUM_TRIANGULAR_FILTERS = 40

# Select the default number of mel-frequency cepstral coefficents to reduce to from filter banks. This number must be
# less than the number of filters.
NUM_MFCCS = 13

########################################################################################################################
################################################ FEATURE CALCULATIONS ##################################################
########################################################################################################################
def generate_mel_filter_banks(signal, sample_rate_hz, frame_size_s=FRAME_SIZE_S, frame_stride_s=FRAME_STRIDE_S,
                              window_fn=functools.partial(tf_signal.hamming_window, periodic=True),
                              fft_num_points=STFT_NUM_POINTS, lower_freq_hz=0.0, num_mel_bins=NUM_TRIANGULAR_FILTERS,
                              log_offset=1e-6, should_log_weight=False):

    # Convert the signal to a tf tensor in case it is in an np array.
    signal = tf.convert_to_tensor(signal, dtype=tf.float32)

    # Compute the remaining parameters for this calculation.
    frame_length = int(sample_rate_hz * frame_size_s)
    frame_step   = int(sample_rate_hz * frame_stride_s)

    # The upper frequency is bounded by half the sample rate by Nyquist's Law.
    upper_freq_hz = sample_rate_hz / 2.0

    # Package the signal into equally-sized, overlapping subsequences (padded with 0s if necessary).
    frames = tf_signal.frame(signal, frame_length=frame_length, frame_step=frame_step, pad_end=True, pad_value=0)

    # Apply a Short-Term Fourier Transform (STFT) to convert into the frequency domain (assuming each window has a
    # constant frequency snapshot).
    stfts = tf_signal.stft(frames,
                           frame_length=frame_length,
                           frame_step=frame_step,
                           fft_length=fft_num_points,
                           window_fn=window_fn)

    # Compute the magnitude and power of the frequencies (the magnitude spectrogram).
    magnitude_spectrograms = tf.abs(stfts)
    power_spectograms = tf.real(stfts * tf.conj(stfts))

    # Warp the linear-scale spectrograms into the mel-scale.
    num_spectrogram_bins = 1 + int(fft_num_points/2)

    # Compute the conversion matrix to mel-frequency space.
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(num_mel_bins=num_mel_bins,
                                                                                num_spectrogram_bins=num_spectrogram_bins,
                                                                                sample_rate=sample_rate_hz,
                                                                                lower_edge_hertz=lower_freq_hz,
                                                                                upper_edge_hertz=upper_freq_hz,
                                                                                dtype=tf.float32)

    # Apply the conversion to complete the calculation of the filter-bank
    mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    if should_log_weight:
        return tf.log(mel_spectrograms + log_offset)
    else:
        return mel_spectrograms

def generate_log_mel_filter_banks(signal, sample_rate_hz, frame_size_s=FRAME_SIZE_S, frame_stride_s=FRAME_STRIDE_S,
                                  window_fn=functools.partial(tf_signal.hamming_window, periodic=True),
                                  fft_num_points=STFT_NUM_POINTS, lower_freq_hz=0.0,
                                  num_mel_bins=NUM_TRIANGULAR_FILTERS, log_offset=1e-6):

    # Calculate the filter banks.
    return generate_mel_filter_banks(signal, sample_rate_hz, frame_size_s=frame_size_s, frame_stride_s=frame_stride_s,
                                     window_fn=window_fn, fft_num_points=fft_num_points,
                                     lower_freq_hz=lower_freq_hz, num_mel_bins=num_mel_bins, log_offset=log_offset,
                                     should_log_weight=True)

def generate_mfcc(signal, sample_rate_hz, num_mfccs=NUM_MFCCS, frame_size_s=FRAME_SIZE_S, frame_stride_s=FRAME_STRIDE_S,
                  window_fn=functools.partial(tf_signal.hamming_window, periodic=True), fft_num_points=STFT_NUM_POINTS,
                  lower_freq_hz=0.0, num_mel_bins=NUM_TRIANGULAR_FILTERS, log_offset=1e-6, should_log_weight=True):

    # Calculate the log-mel-filter banks.
    log_mel_filter_banks = \
    generate_log_mel_filter_banks(signal, sample_rate_hz, frame_size_s=frame_size_s, frame_stride_s=frame_stride_s,
                                  window_fn=window_fn, fft_num_points=fft_num_points, lower_freq_hz=lower_freq_hz,
                                  num_mel_bins=num_mel_bins, log_offset=log_offset)

    return tf_signal.mfccs_from_log_mel_spectrograms(log_mel_filter_banks)[..., :num_mfccs]

def generate_features(args):
    """
    Generate the desired features using this script.
    """
    # Select the feature to generate.
    if(args.format == OUTPUT_MEL_FILTER_BANK):
        feature_gen_fn = generate_mel_filter_banks
    elif(args.format == OUTPUT_LOG_MEL_FILTER_BANK):
        feature_gen_fn = generate_log_mel_filter_banks
    elif(args.format == OUTPUT_MFCC):
        feature_gen_fn = generate_mfcc
    else:
        raise ValueError('--format was passed an invalid value.')

    # Get paths to valid audio recordings for each surah.
    if args.surah:
        surahs_to_tensorize = [args.surah]
    else:
        # 1-index the surah numbers in the vector.
        surahs_to_tensorize = np.arange(NUM_SURAHS) + 1

    if not args.ayah:
        paths_to_tensorize = recording_utils.get_paths_to_surah_recordings(args.local_download_dir, surahs_to_tensorize)
    else:
        ayah = (args.surah, args.ayah)
        paths_to_tensorize = recording_utils.get_paths_to_ayah_recordings(args.local_download_dir, [ayah])

    # Generate the desired feature and calculate the features.
    output = None
    for recording_path in paths_to_tensorize:

        # Start a TensorFlow Session.
        # Note that encapsulating the loop in a session will result in exorbitant memory demands on large audiosets.
        with tf.Session() as sess:
            sample_rate_hz, signal = scipy.io.wavfile.read(recording_path)

            # Transpose the signal in order to make the 0th axis over the channels of the audio.
            signal = signal.transpose()

            if sample_rate_hz in SUPPORTED_FREQUENCIES: 
                output = feature_gen_fn(signal, sample_rate_hz)
            else:
                if(bool(args.verbose)):
                    print('Unsupported sampling frequency for recording at path %s: %d.' % (recording_path, sample_rate_hz))
                continue

            # Save the outputs to their own file(s).
            path1, base_filename = os.path.split(recording_path)
            filename, _          = os.path.splitext(base_filename)

            path2, ayah_folder   = os.path.split(path1)
            _, surah_folder      = os.path.split(path2)

            # Join the directory, metric type, surah number, ayah number, and filename (with new extension).
            save_dir = os.path.join(args.output_dir, args.format, surah_folder, ayah_folder)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            save_path = os.path.join(save_dir, filename)
            output_np_array = tf.Session().run(output)
            np.save(save_path, output_np_array)

            # Close the session.
            sess.close()

if __name__ == "__main__":
    generate_features(args)

