"""
Utility functions for directory and file IO.
Author: Fahim Dalvi, Hamzah Khan
Date: May 3, 2019
"""

import os
import random
import shutil
import string

DEFAULT_CACHE_DIRECTORY = '.cache'

def does_cached_csv_dataset_exist(path_to_dataset_csv):
    return os.path.isfile(path_to_dataset_csv)

def prepare_cache_directories(subdirectory_names_tuple, cache_directory=DEFAULT_CACHE_DIRECTORY, use_cache=True, verbose=False):
    """
    Creates a set of directories to be used for caching different pieces of data.
    """
    # If we don't use the cache, create a common temporary cache directory.
    if not use_cache:
        cache_directory = create_temporary_cache_directory_name()

    # Create a list to hold the desired cache folders.
    subcache_directory_list = []

    for subdirectory_name in subdirectory_names_tuple:
        subcache_directory_path = create_cache_directory(subdirectory_name, cache_directory=cache_directory, use_cache=True, verbose=verbose)
        subcache_directory_list.append(subcache_directory_path)

    return tuple(subcache_directory_list)

def prepare_ayah_directory(directory_path, surah_num, ayah_num):
    """
    Prepares a directory structure for an ayah. Returns the directory path.
    """
    ayah_directory_path = os.path.join(directory_path, 's{}'.format(surah_num), 'a{}'.format(ayah_num))
    os.makedirs(ayah_directory_path, exist_ok=True)
    return ayah_directory_path

def create_temporary_cache_directory_name():
    """
    Create a temporary cache directory anme.
    """
    temp_directory_name = ''.join([random.choice(string.ascii_letters) for i in range(10)])
    return ".tmp-" + temp_directory_name

def create_cache_directory(directory_subpath, cache_directory=DEFAULT_CACHE_DIRECTORY, use_cache=True, verbose=False):
    """
    Creates the cache directory structure for the given directory.
    """
    # Ensure the cache_directory always exists.
    os.makedirs(cache_directory, exist_ok=True)

    # If we don't want to use the cache, create a temporary cache to place files in.
    if not use_cache:
        cache_directory = create_temporary_cache_directory_name()
        if verbose:
            print("Created temporary empty directory at ", cache_directory, ".")

    # Create cache path.
    desired_cache_dir = os.path.join(cache_directory, directory_subpath)
    os.makedirs(desired_cache_dir, exist_ok=True)

    return desired_cache_dir

def clean_cache_directories(cache_directory=DEFAULT_CACHE_DIRECTORY, verbose=False):
    """
    Removes all cached files (but keeps the cache structure intact for future
    use).
    """
    # If the cache directory doesn't exist, then just make an empty one.
    if not os.path.isdir(cache_directory):
        os.makedirs(cache_directory)
        
    for subdirectory in os.listdir(cache_directory):
        if verbose:
            print("Removing cache_subdirectory %s." % (subdirectory))
        shutil.rmtree(os.path.join(cache_directory, subdirectory))

def delete_cache_directories(cache_directory=DEFAULT_CACHE_DIRECTORY):
    """
    Deletes the entire cache (files and structure).
    """
    shutil.rmtree(cache_directory)
