"""
Contains helper methods that are used to train and infer Tarteel ML models
"""
import dill as pickle
import numpy as np
import os


def convert_list_of_arrays_to_padded_array(list_varying_sizes, pad_value=0):
    """
    Converts a list of 2D arrays of varying sizes to a single 3D numpy array. The extra elements are padded
    :param list_varying_sizes: the list of 2D arrays
    :param pad_value: the value with which to pad the arrays
    """
    max_shape = [0] * len(list_varying_sizes[0].shape)
    # first pass to compute the max size
    for arr in list_varying_sizes:
        shape = arr.shape
        max_shape = [max(s1, s2) for s1, s2 in zip(shape, max_shape)]
    padded_array = pad_value * np.ones((len(list_varying_sizes), *max_shape))

    # second pass to fill in the values in the array:
    for a, arr in enumerate(list_varying_sizes):
        r, c = arr.shape  # TODO(abidlabs): maybe make more general to more than just 2D arrays.
        padded_array[a, :r, :c] = arr

    return padded_array


def preprocess_encoder_input(arr):
    """
    Simple method to handle the complex MFCC coefs that are produced during preprocessing. This means:
    1. (For now), discarding one of the channels of the MFCC coefs
    2. Collapsing any empty dimensions
    :param arr: the array of MFCC coefficients.
    """
    return arr.squeeze()[0]


# Load every one-hot-encoded output as a dictionary
def get_one_hot_encodings(filepath='../data/one-hot.pkl'):
    """
    Gets the one_hot encodings of the verses of the Quran, along with mappings of characters to ints
    :param filepath: the filepath to the one_hot encoding pickled file
    :return:
    """
    with open(filepath, 'rb') as one_hot_quran_pickle_file:
        one_hot_obj = pickle.load(one_hot_quran_pickle_file)
    return one_hot_obj


def get_one_hot_encoded_verse(surah_num, ayah_num):
    """
    Converts a one-hot-encoded verse into forms that can be used by the LSTM decoder
    :param surah_num: an int designating the chapter number, one-indexed
    :param ayah_num: an int designating the verse number, one-indexed
    """
    # Load the preprocessed one-hot encoding
    one_hot_obj = get_one_hot_encodings()
    one_hot_verse = one_hot_obj['quran']['surahs'][surah_num - 1]['ayahs'][ayah_num - 1]['text']
    num_chars_in_verse, num_unique_chars = one_hot_verse.shape

    # Generate decoder_input_data
    decoder_input = np.zeros((num_chars_in_verse + 2, num_unique_chars + 2))
    decoder_input[0, :] = [0] * num_unique_chars + [1, 0]  # START token
    decoder_input[1:num_chars_in_verse + 1, :-2] = one_hot_verse  # original verse
    decoder_input[-1, :] = [0] * num_unique_chars + [0, 1]  # STOP token

    # Generate decoder_target_data
    decoder_target = np.zeros((num_chars_in_verse + 2, num_unique_chars + 2))
    decoder_target[:num_chars_in_verse, :-2] = one_hot_verse  # original verse
    decoder_target[-2, :] = [0] * num_unique_chars + [0, 1]  # STOP token

    return decoder_input, decoder_target


def shuffle_together(*arrays):
    """
    A helper method to randomly shuffle the order of an arbitrary number of arrays while keeping their relative orders
    the same.
    :param arrays A list of passed-in arrays.
    :return:
    """
    array_sizes = [array.shape[0] for array in arrays]

    # All arrays should be of equal size.
    first_size = array_sizes[0]
    assert all([array_size == first_size for array_size in array_sizes])

    # Permute the arrays and return them as a tuple.
    order = np.random.permutation(first_size)
    return tuple([array[order] for array in arrays]])

  
def get_seq2seq_data(local_coefs_dir='../.outputs/mfcc', surahs=[1], n=100, return_filenames=False):
    """
    Builds a dataset to be used with the sequence-to-sequence network.
    :param local_coefs_dir: a string with the path of the coefficients for prediction
    """

    def get_encoder_and_decoder_data(n=100):
        count = 0
        encoder_input_data = []
        decoder_input_data = []
        decoder_target_data = []
        filenames = []
        for surah_num in surahs:
            local_surah_dir = os.path.join(local_coefs_dir, "s" + str(surah_num))
            for _, ayah_directories, _ in os.walk(local_surah_dir):
                for ayah_directory in ayah_directories:
                    ayah_num = ayah_directory[1:]
                    local_ayah_dir = os.path.join(local_surah_dir, ayah_directory)
                    for _, _, recording_filenames in os.walk(local_ayah_dir):
                        for recording_filename in recording_filenames:
                            local_coefs_path = os.path.join(local_ayah_dir, recording_filename)
                            encoder_input = np.load(local_coefs_path)
                            encoder_input = preprocess_encoder_input(encoder_input)
                            encoder_input_data.append(encoder_input)

                            decoder_input, decoder_target = get_one_hot_encoded_verse(int(surah_num), int(ayah_num))
                            decoder_input_data.append(decoder_input)
                            decoder_target_data.append(decoder_target)
                            filenames.append(recording_filename)
                            count += 1
                            if count == n:
                                return encoder_input_data, decoder_input_data, decoder_target_data, filenames
        return encoder_input_data, decoder_input_data, decoder_target_data, filenames

    encoder_input_data, decoder_input_data, decoder_target_data, filenames = get_encoder_and_decoder_data(n=n)
    encoder_input_data = convert_list_of_arrays_to_padded_array(encoder_input_data)
    decoder_input_data = convert_list_of_arrays_to_padded_array(decoder_input_data)
    decoder_target_data = convert_list_of_arrays_to_padded_array(decoder_target_data)
    encoder_input_data, decoder_input_data, decoder_target_data, filenames = shuffle_together(
        encoder_input_data, decoder_input_data, decoder_target_data, np.array(filenames))

    if return_filenames:
        return encoder_input_data, decoder_input_data, decoder_target_data, filenames
    else:
        return encoder_input_data, decoder_input_data, decoder_target_data


def decode_sequence(input_seq, num_decoder_tokens, encoder_model, decoder_model, max_decoder_seq_length):
    """
    A method that performs basic inference from an audio coefficients by making predictions one character at a time and
    then feeding the previous predicted characters back into the model to get the next character.
    :param input_seq: the sequence of MFCC coefficients to use for prediction.
    :param num_decoder_tokens: the total number of distinct decoder tokens.
    :param encoder_model: the model used for encoding MFCC coefficients into a latent representation.
    :param decoder_model: the model used to decode a latent representation into a sequence of characters.
    :param max_decoder_seq_length: the longest possible sequence of predicted text, in number of characters, after which
        inference necessary ends even if the STOP token is not produced.
    :return: the inferred character sequence.
    """
    one_hot_obj = get_one_hot_encodings()
    reverse_target_char_index = one_hot_obj['int_to_char']
    reverse_target_char_index[num_decoder_tokens-2] = '->'
    reverse_target_char_index[num_decoder_tokens-1] = '<-'

    target_char_index = {v: k for k, v in reverse_target_char_index.items()}

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_char_index['->']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '<-' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence
