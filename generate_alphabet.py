"""Create a text file with all unique characters in the Quran."""

import argparse
import logging
import os

from tqdm import tqdm
from tqdm import trange

from utils import quran_helper

EPILOG = 'Create a text file with all unique characters in the Quran.'
ALPHABET_FILENAME = 'alphabet.txt'
parser = argparse.ArgumentParser(epilog=EPILOG)
parser.add_argument(
    '-o', '--output-dir', type=str, default='.',
    help='Directory to write text file to.'
)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


def main():
    quran = quran_helper.Quran()
    alphabet_path = os.path.join(args.output_dir, ALPHABET_FILENAME)
    alphabet = set()

    with trange(1, quran_helper.MAX_SURAH_NUMBER + 1) as surahs_pbar:
        for surah_number in surahs_pbar:
            surahs_pbar.set_postfix_str("Surah {}".format(surah_number))
            ayahs = quran.surah(surah_number)
            with tqdm(ayahs) as ayahs_pbar:
                for ayah_number in ayahs_pbar:
                    ayahs_pbar.set_postfix_str("Ayah {}".format(ayah_number))
                    text = quran.get_ayah_text(surah_number, ayah_number)
                    alphabet = alphabet | set(text)

    with open(alphabet_path, 'w') as file:
        for letter in alphabet:
            file.write("%s\n" % letter)

    logging.info("Wrote alphabet file to {}".format(alphabet_path))


if __name__ == '__main__':
    main()
