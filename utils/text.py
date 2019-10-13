"""
Utility functions for interacting with Qur'anic text.
Author: Hamzah Khan
Date: Oct. 13, 2019

A file for helper functions that read the Qur'anic json data and interact with the Arabic text.
"""

import json

# Define constants.
QURAN_KEY  = "quran"
SURAHS_KEY = "surahs"
AYAHS_KEY  = "ayahs"
TEXT_KEY   = "text"

NUM_KEY       = "num"
NAME_KEY      = "name"
BISMILLAH_KEY = "bismillah"

BISMILLAH_INDEX = 0

def load_quran_obj_from_json(input_json_path):
    """
    Loads the Json object containing Qur'anic data from the specific path. 
    """
    try:
        with open(input_json_path, 'rb') as quran_json_file:
            # Import json file and return Qur'an object.
            return json.load(quran_json_file)[QURAN_KEY]
    except:
        raise Exception("Json file not found at ", input_json_path)

def convert_quran_json_to_dict(quran_obj, should_include_bismillah=False):
    """
    Converts the Qur'anic Json file to a dict mapping from (surah number, ayah number) to the ayah text.
    If `should_include_bismillah` is True, then bismillah text will be stored under (surah num, 0).
    """
    ayah_to_text_dict = {}

    # Extract surah and ayah information and fill in text.
    for surah_obj in quran_obj[SURAHS_KEY]:
        surah_num = surah_obj[NUM_KEY]

        for ayah_obj in surah_obj[AYAHS_KEY]:
            ayah_num = ayah_obj[NUM_KEY]
            ayah_to_text_dict[(surah_num, ayah_num)] = ayah_obj[TEXT_KEY]

            if BISMILLAH_KEY in ayah_obj:
                ayah_to_text_dict[(surah_num, BISMILLAH_INDEX)] = ayah_obj[BISMILLAH_KEY]

    return ayah_to_text_dict
