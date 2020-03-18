import logging
from pathlib import Path
from typing import Union

from utils import files

QURAN_FILE = Path(__file__).parent.parent.absolute() / 'data/quran.json'
MAX_SURAH_NUMBER = 114
MIN_SURAH_NUMBER = 1


class Quran:
    """An object used to get a Surah's Ayahs along with transcript loaded from JSON.

    The JSON file is ~2MB when loaded in memory which is acceptable.
    A better alternative is to use a database, however that is too much overhead for the simple
    operations we perform.
    """

    def __init__(self, quran_file: str = QURAN_FILE):
        self._quran_json = files.open_json(quran_file)

    def get_quran(self):
        return self._quran_json

    def surah(self, surah_number: int) -> Union[dict, None]:
        if (surah_number > MAX_SURAH_NUMBER) or (surah_number < MIN_SURAH_NUMBER):
            logging.warning('Surah number out of bounds')
            return None
        try:
            surah = self._quran_json[str(surah_number)]
            return surah
        except KeyError:
            logging.warning('Surah number not found')
            return None

    def ayah(self, surah_number: int, ayah_number: int) -> Union[dict, None]:
        try:
            ayah = self.surah(surah_number)[str(ayah_number)]
            return ayah
        except KeyError:
            logging.warning('Ayah not found')
            return None

    def get_ayah_text(self,
                      surah_number: int,
                      ayah_number: int,
                      uthmani: bool = False) -> Union[str, None]:
        result = self.ayah(surah_number, ayah_number)
        if result:
            if uthmani:
                return result.get('displayText')
            else:
                return result.get('text')
        else:
            return result
