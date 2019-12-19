"""Utility funtions for reading from and writing to JSON, CSV, and other file types."""

import csv
import json
import os
from typing import Any
from typing import List
from typing import Union


def open_csv(csv_file: str, dict_reader: bool = False, delimiter: str = ',') -> List:
    """Delimiter can also be a pipe (|)"""
    with open(csv_file, 'r') as file:
        if dict_reader:
            reader = csv.DictReader(file, delimiter=delimiter)
        else:
            reader = csv.reader(file, delimiter=delimiter)
        return list(reader)


def write_csv(csv_file: str, rows: List[Any], has_header: bool = True) -> None:
    with open(csv_file, 'w') as file:
        csv_writer = csv.writer(file)
        if has_header:
            csv_writer.writerow(rows[0])
            rows.pop(0)
        csv_writer.writerows(rows)


def open_text(text_file: str):
    with open(text_file, 'r') as file:
        return file.readlines()


def get_file_size(path: str) -> Union[int, None]:
    try:
        return os.path.getsize(path)
    except os.error as e:
        print("util.file_handler got the following error: {}".format(e))
        return None


def open_json(json_file: str) -> dict:
    with open(json_file) as file:
        return json.load(file)


def write_json(json_file: str, json_dict: dict) -> None:
    with open(json_file, 'w') as file:
        json.dump(json_dict, file)
