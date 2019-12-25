"""Create a text file with all ayahs line by line."""

import argparse
import logging
import os

from tqdm import tqdm
from tqdm import trange

from utils import quran_helper

DESCRIPTION = 'Create a text file with all unique ayahs in the Quran.'
VOCABULARY_FILENAME = 'vocabulary.txt'
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    '-o', '--output-dir', type=str, default='.',
    help='Directory to write text file to.'
)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


def main():
    quran = quran_helper.Quran()
    vocabulary_path = os.path.join(args.output_dir, VOCABULARY_FILENAME)

    with open(vocabulary_path, 'w') as file:
        with trange(1, quran_helper.MAX_SURAH_NUMBER + 1) as surahs_pbar:
            for surah_number in surahs_pbar:
                ayahs = quran.surah(surah_number)
                with tqdm(ayahs) as ayahs_pbar:
                    for ayah_number in ayahs_pbar:
                        text = quran.get_ayah_text(surah_number, ayah_number)
                        file.write("%s\n" % text)

    logging.info("Wrote vocabulary file to {}".format(vocabulary_path))


if __name__ == '__main__':
    main()
