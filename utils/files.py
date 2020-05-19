"""Utility functions for directory and file IO.

A file for helper functions that read and manipulate non-audio files and directories.
Helpers in this file should NOT require an internet connection to use.
"""

import csv
import json
import logging
import os
from typing import Any
from typing import List
from typing import Tuple
from typing import Union
import random
import shutil
import string
import yaml

DEFAULT_CACHE_DIRECTORY = '.cache'


def does_cached_csv_dataset_exist(path_to_dataset_csv: str) -> bool:
    return os.path.isfile(path_to_dataset_csv)


def get_path_to_dataset_csv(
        csv_cache_dir: str,
        dataset_csv_filename: str) -> str:
    return os.path.join(csv_cache_dir, dataset_csv_filename)


def get_dataset_entries(path_to_dataset_csv: str) -> Tuple[List, List]:
    """Parse and return the header and rows of the dataset csv."""
    with open(path_to_dataset_csv, 'r') as file:
        content = file.read()
        csv_data = csv.reader(content.splitlines(), delimiter=',')
        entries = list(csv_data)
        header_row = entries[0]
        dataset = entries[1:]
        return header_row, dataset


def prepare_cache_directories(
        subdirectory_names_tuple: Tuple,
        cache_directory: str = DEFAULT_CACHE_DIRECTORY,
        use_cache: bool = True) -> Tuple:
    """Create a set of directories to be used for caching different pieces of data."""
    # If we don't use the cache, create a common temporary cache directory.
    if not use_cache:
        cache_directory = create_temporary_cache_directory_name()

    # Create a list to hold the desired cache folders.
    subcache_directory_list = []

    for subdirectory_name in subdirectory_names_tuple:
        subcache_directory_path = create_cache_directory(
            subdirectory_name, cache_directory=cache_directory, use_cache=True)
        subcache_directory_list.append(subcache_directory_path)

    return tuple(subcache_directory_list)


def prepare_ayah_directory(directory_path, surah_num, ayah_num) -> str:
    """Prepare a directory structure for an ayah.

    Returns:
        ayah_directory_path The path to the directory.
    """
    ayah_directory_path = os.path.join(directory_path, 's{}'.format(surah_num), 'a{}'.format(ayah_num))
    os.makedirs(ayah_directory_path, exist_ok=True)
    return ayah_directory_path


def create_temporary_cache_directory_name() -> str:
    """Create a temporary cache directory name."""
    temp_directory_name = ''.join([random.choice(string.ascii_letters) for i in range(10)])
    return ".tmp-" + temp_directory_name


def create_cache_directory(
        directory_subpath: str,
        cache_directory: str = DEFAULT_CACHE_DIRECTORY,
        use_cache: bool = True) -> str:
    """Create the cache directory structure for the given directory."""
    # Ensure the cache_directory always exists.
    os.makedirs(cache_directory, exist_ok=True)

    # If we don't want to use the cache, create a temporary cache to place files in.
    if not use_cache:
        cache_directory = create_temporary_cache_directory_name()
        logging.info("Created temporary empty directory at {}.".format(cache_directory))

    # Create cache path.
    desired_cache_dir = os.path.join(cache_directory, directory_subpath)
    os.makedirs(desired_cache_dir, exist_ok=True)

    return desired_cache_dir


def clean_cache_directories(cache_directory: str = DEFAULT_CACHE_DIRECTORY) -> None:
    """Remove all cached files (but keeps the cache structure intact for future use)."""
    # If the cache directory doesn't exist, then just make an empty one.
    if not os.path.isdir(cache_directory):
        os.makedirs(cache_directory)
        
    for subdirectory in os.listdir(cache_directory):
        logging.info("Removing cache_subdirectory {}.".format(subdirectory))
        shutil.rmtree(os.path.join(cache_directory, subdirectory))


def delete_cache_directories(cache_directory: str = DEFAULT_CACHE_DIRECTORY) -> None:
    """Delete the entire cache (files and structure)."""
    shutil.rmtree(cache_directory)


def get_all_files_in_directory(directory: str) -> List:
    (_, _, filenames) = next(os.walk(directory))
    return filenames


def get_file_size(path: str) -> Union[int, None]:
    try:
        return os.path.getsize(path)
    except os.error as e:
        logging.warning(
            "Unable to get the file size for {}. Got the following error: {}".format(path, e))
        return None


def change_file_extension(filename: str, extension) -> str:
    audio_filename = os.path.basename(filename)
    return "{}.{}".format(audio_filename[:-4], extension)


def open_json(json_file: str) -> dict:
    with open(json_file) as file:
        return json.load(file)


def write_json(json_file: str, json_dict: dict) -> None:
    with open(json_file, 'w') as file:
        json.dump(json_dict, file)


def write_csv(csv_file: str, rows: List[Any], has_header: bool = True) -> None:
    with open(csv_file, 'w') as file:
        csv_writer = csv.writer(file)
        if has_header:
            csv_writer.writerow(rows[0])
            rows.pop(0)
        csv_writer.writerows(rows)


def load_yaml_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def read_file(path: str) -> List[str]:
    with open(path, 'r') as f:
        return f.read().splitlines()
