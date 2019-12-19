import os
from os import path
from typing import Union

from util import file_handler
from util.logging import print_warn

QURAN_FILE = path.join(path.dirname(os.getcwd()), 'data/quran.json')
MAX_SURAH_NUMBER = 114
MIN_SURAH_NUMBER = 1


class Quran:
	"""An object used to get a Surah's Ayahs along with transcript loaded from JSON.

	The JSON file is ~12MB when loaded in memory which is acceptable. A better alternative
	is to use a database, however that is too much overhead for the simple operations we want to
	perform.
	"""
    def __init__(self, quran_file: str = QURAN_FILE):
        self._quran_json = file_handler.open_json(quran_file)
        self._quran_json = self._quran_json['quran']['surahs']

    def get_quran(self):
        return self._quran_json

    def surah(self, surah_number: int) -> Union[dict, None]:
        if (surah_number > MAX_SURAH_NUMBER) or (surah_number < MIN_SURAH_NUMBER):
            print_warn('Surah number out of bounds')
            return None
        try:
            surah = self._quran_json[surah_number-1]
            return surah
        except KeyError:
            print_warn('Surah number not found')
            return None

    def ayah(self, surah_number: int, ayah_number: int) -> Union[dict, None]:
        try:
            ayah = self.surah(surah_number).get('ayahs')[ayah_number-1]
            return ayah
        except KeyError:
            print_warn('Ayah not found')
            return None

    def get_ayah_text(self, surah_number: int, ayah_number: int) -> Union[str, None]:
        result = self.ayah(surah_number, ayah_number)
        if result:
            return result.get('text')
        else:
            return result

