#!/usr/bin/env python3
"""
A file for creating the train-test-validation split for the quranic text file.

Author: Hamzah Khan
Date: Jan. 23, 2019
"""

from collections import defaultdict
import dill as pickle
import numpy as np
from sklearn.model_selection import train_test_split

from argparse import ArgumentParser

parser = ArgumentParser(description='Tarteel Data Train-Test-Validation Splitter')
parser.add_argument('-i', '--input_path', type=str, default="data/data-uthmani.pickle")
parser.add_argument('-o', '--output_path', type=str)
parser.add_argument('-u', '--unique_ayahs', type=bool, default=False)
parser.add_argument('-v', '--verbose', type=bool, default=False)
args = parser.parse_args()

# Define constants.
NUM_SURAHS = 114

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

# This gives us a 60-20-20 split.
RANDOM_SEED = 1
TRAIN_SPLIT = 0.6
TEST_SPLIT  = 0.2
VALIDATION_SPLIT = 0.2

def get_verse_in_quran_obj(one_hot_quran, surah_num, ayah_num):
    """
    Looks up and returns the (encoded or decoded) string in the quran object.
    """
    return one_hot_quran[SURAHS_KEY][surah_num][AYAHS_KEY][ayah_num][TEXT_KEY]

def create_split(input_path, split_unique_ayahs=False):
    """
    Create a train-test-validation split given the data and whether we should do it over unique ayahs or not.
    """
    try:
        with open(input_path, 'rb') as one_hot_pickle:
            one_hot_obj = pickle.load(one_hot_pickle)
    except:
        print("Pickle file failed to open. Exiting...")
        return

    one_hot_quran    = one_hot_obj[QURAN_KEY]
    str_to_onehot_fn = one_hot_obj[ENCODING_MAP_KEY]
    onehot_to_str_fn = one_hot_obj[DECODING_MAP_KEY]
    char_to_int_map  = one_hot_obj[CHAR_TO_INT_MAP_KEY]
    int_to_char_map  = one_hot_obj[INT_TO_CHAR_MAP_KEY]

    encoding_fn = lambda string: str_to_onehot_fn(string, char_to_int_map)
    decoding_fn = lambda one_hot: onehot_to_str_fn(one_hot, int_to_char_map)

    ayah_nums = []
    if split_unique_ayahs:
        unique_ayahs = defaultdict(list)
        for surah_num in range(NUM_SURAHS):
            for ayah_num in range(len(one_hot_quran[SURAHS_KEY][surah_num][AYAHS_KEY])):
                one_hot_ayah = get_verse_in_quran_obj(one_hot_quran, surah_num, ayah_num)
                string_ayah = decoding_fn(one_hot_ayah)
                unique_ayahs[string_ayah].append((surah_num, ayah_num))

        for ayah_string in unique_ayahs:
            identical_ayah_list = unique_ayahs[ayah_string]
            ayah_nums.append(identical_ayah_list)

    else:
        for surah_num in range(NUM_SURAHS):
            for ayah_num in range(len(one_hot_quran[SURAHS_KEY][surah_num][AYAHS_KEY])):
                ayah_nums.append(ayah_num)

    # Logic for splits
    split1_percent = TRAIN_SPLIT + VALIDATION_SPLIT
    split2_percent = 1.0 - (VALIDATION_SPLIT / split1_percent)

    X_train_valid, X_test = train_test_split(ayah_nums,
                                             train_size=split1_percent,
                                             random_state=RANDOM_SEED,
                                             shuffle=True)
    X_train, X_valid = train_test_split(X_train_valid,
                                        train_size=split2_percent,
                                        random_state=RANDOM_SEED,
                                        shuffle=True)

    return X_train, X_test, X_valid

    

if __name__ == '__main__':
    X_train, X_test, X_valid = create_split(args.input_path, args.unique_ayahs)
    print(len(X_train), len(X_test), len(X_valid))
