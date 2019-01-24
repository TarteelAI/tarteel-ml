#!/usr/bin/env python3
"""
A file for creating a one-hot encoding of all characters, including madd and harakat, in Tarteel's Qur'an dataset.

The output pickle file will contain an object with the one-hot encoded Qur'an, an encoding function, and a decoding
function.

Author: Hamzah Khan
Date: Jan. 12, 2019
"""

import copy
import dill as pickle
import json
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser(description='Tarteel Arabic One Hot Encoding Generator')
parser.add_argument('-i', '--input_json_path', type=str)
parser.add_argument('-o', '--output_pickle_path', type=str)
parser.add_argument('-v', '--verbose', type=bool, default=False)
args = parser.parse_args()

# Define constants.
QURAN_KEY  = "quran"
SURAHS_KEY = "surahs"
AYAHS_KEY  = "ayahs"
TEXT_KEY   = "text"

NUM_KEY       = "num"
NAME_KEY      = "name"
BISMILLAH_KEY = "bismillah"

ENCODING_MAP_KEY = "encoding_map"
DECODING_MAP_KEY = "decoding_map"
CHAR_TO_INT_MAP_KEY = "char_to_int"
INT_TO_CHAR_MAP_KEY = "int_to_char"

def create_list_of_quranic_chars(quran_obj, surahs_key=SURAHS_KEY, ayahs_key=AYAHS_KEY, text_key=TEXT_KEY):
    """
    Create a sorted list containing every character in the Qur'an text provided and return it.
    :param quran_obj: An object containing surah objects.
    :type quran_obj: object
    :param surahs_key: The key in quran_obj to the list of surah objects.
    :type surahs_key: string
    :param ayahs_key: The key in each surah object to the list of ayah objects in that surah.
    :type ayahs_key: string
    :param text_key: The key to the actual Qur'anic text in each ayah object.
    :type text_key: string
    :returns: A sorted list containing every Arabic character in the Qur'an exactly once.
    :rtype: list string
    """
    quranic_char_set = set()

    for surah_obj in quran_obj[surahs_key]:
        for ayah_obj in surah_obj[ayahs_key]:
            ayah_text = ayah_obj[text_key]

            for char in ayah_text:
                quranic_char_set.add(char)

    return sorted(list(quranic_char_set))

def create_one_hot_encoding(quranic_char_list):
    """
    Creates a one-hot encoding that associates each character in the argument list to a number and vice versa.
    :param quranic_char_list: A list of characters.
    :type quranic_char_list: list string
    :returns: A tuple containing the encoding and decoding functions for the alphabet.
    :rtype: tuple (function string => int, function int => string)
    """

    # Define an encoding of characters to integers.
    char_to_int = dict((c, i) for i, c in enumerate(quranic_char_list))
    int_to_char = dict((i, c) for i, c in enumerate(quranic_char_list))

    def encode_char_as_one_hot(string, char_to_int):
        """
        Converts a string of characters from our alphabet into a one_hot encoded string.
        """
        str_len = len(string)
        int_list = np.array([char_to_int[char] for char in string])

        one_hot_string = np.zeros((str_len, len(char_to_int)))
        one_hot_string[np.arange(str_len), int_list] = 1

        return one_hot_string

    def decode_one_hot_as_string(one_hot_string, int_to_char):
        """
        Converts a one_hot encoded numpy array back into a string of characters from our alphabet.
        """
        int_list = list(np.argmax(one_hot_string, axis=1))
        char_list = [int_to_char[integer] for integer in int_list]

        return str(char_list)

    return char_to_int, int_to_char, encode_char_as_one_hot, decode_one_hot_as_string


def generate_a_one_hot_encoded_script(quran_obj,
                                      encoding_fn,
                                      surahs_key=SURAHS_KEY,
                                      ayahs_key=AYAHS_KEY,
                                      text_key=TEXT_KEY,
                                      num_key=NUM_KEY,
                                      name_key=NAME_KEY,
                                      bismillah_key=BISMILLAH_KEY):
    """
    Translates each ayah in the given quran_obj into a vector of one-hot encoded characters using the given encoding.
    Create a sorted list containing every character in the Qur'an text provided and return it.
    :param quran_obj: An object containing surah objects.
    :type quran_obj: object
    :param quran_obj: A function that converts Arabic Qur'anic characters to a one-hot encoding.
    :type quran_obj: function (Arabic string => numpy 2darray)
    :param surahs_key: The key in quran_obj to the list of surah objects.
    :type surahs_key: string
    :param ayahs_key: The key in each surah object to the list of ayah objects in that surah.
    :type ayahs_key: string
    :param text_key: The key to the actual Qur'anic text in each ayah object.
    :type text_key: string
    :param num_key: The key in surah and ayah objects to the ordering of the surah or ayah.
    :type num_key: string
    :param name_key: The key in each surah object to the name of that surah.
    :type name_key: string
    :param bismillah_key: The key to the bismillah text in the first ayah object of each surah object.
    :type bismillah_key: string
    :returns: An object identical to the quran_obj but with one-hot encodings of all Arabic text (not names).
    :rtype: object
    """
    one_hot_quran_encoding = {}
    one_hot_quran_encoding[SURAHS_KEY] = []

    for surah_obj in quran_obj[surahs_key]:
        # Copy new surah object for one-hot Json container.
        one_hot_surah_obj            = {}
        one_hot_surah_obj[num_key]   = surah_obj[num_key]
        one_hot_surah_obj[name_key]  = surah_obj[name_key]
        one_hot_surah_obj[ayahs_key] = []

        for ayah_obj in surah_obj[ayahs_key]:
            ayah_text = ayah_obj[text_key]

            # Make new ayah object for one-hot Json container.
            one_hot_ayah_obj           = {}
            one_hot_ayah_obj[num_key]  = ayah_obj[num_key]
            one_hot_ayah_obj[text_key] = encoding_fn(ayah_text)

            if bismillah_key in ayah_obj:
                one_hot_ayah_obj[bismillah_key] = encoding_fn(ayah_obj[bismillah_key])

            one_hot_surah_obj[ayahs_key].append(one_hot_ayah_obj)
        one_hot_quran_encoding[surahs_key].append(one_hot_surah_obj)

    return one_hot_quran_encoding

def run_script(args):
    """
    Runs the script to find all characters, generate the encoding, and translate and store it in the output file.
    """
    try:
        with open(args.input_json_path) as quran_json_file:

            # Import json file.
            quran_obj = json.load(quran_json_file)[QURAN_KEY]

    except:
        print("Json file failed to open. Exiting script...")
        return

    # Get the list of every character in the Qur'an.
    quranic_char_list = create_list_of_quranic_chars(quran_obj)

    if args.verbose:
        print(quranic_char_list, ' has ', len(quranic_char_list), ' characters.')

    # Create the one-hot encodings.
    char_to_int_map, \
    int_to_char_map, \
    encode_char_as_one_hot, \
    decode_one_hot_as_string = create_one_hot_encoding(quranic_char_list)

    if args.verbose:
        print("encode!")
        x = encode_char_as_one_hot("".join(quranic_char_list))
        print(x)
        print("decode!")
        print(decode_one_hot_as_string(x))

    # Generate the Qur'anic text in one-hot encoding.
    one_hot_quran_encoding = generate_a_one_hot_encoded_script(
        quran_obj,
        lambda string: encode_char_as_one_hot(string, char_to_int_map))

    # Create an object with the encoding and the two functions.
    full_object = {
        QURAN_KEY: one_hot_quran_encoding,
        ENCODING_MAP_KEY: encode_char_as_one_hot,
        DECODING_MAP_KEY: decode_one_hot_as_string,
        CHAR_TO_INT_MAP_KEY: char_to_int_map,
        INT_TO_CHAR_MAP_KEY: int_to_char_map
    }

    try:
        with open(args.output_pickle_path, 'wb') as one_hot_quran_pickle_file:
            pickle.dump(full_object, one_hot_quran_pickle_file)
    except:
        print("One-hot Pickle file failed to save. Exiting script...")
        return

def load_data(pickle_file):
    """
    A sample function to demonstrate how to load the object.
    """
    try:
        with open(pickle_file, 'rb') as one_hot_pickle:
            one_hot_obj = pickle.load(one_hot_pickle)

            print('Now, we can do things with it! Keys: ', one_hot_obj.keys())

    except:
        print("Pickle file failed to open. Exiting...")
        return


if __name__ == "__main__":
    run_script(args)
