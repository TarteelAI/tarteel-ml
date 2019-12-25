# Tarteel Machine Learning 

This repo is designed to house code related to Tarteel machine learning related tasks. :microscope:

Specifically, things like:

* Model selection :white_check_mark:
* Preprocessing of data :sound:
* Model training, validation, and and iteration :repeat:
* Demos :rocket:

Code here is mostly experimental so check back regularly for updates.

If you found this repo helpful, please keep it's contributors in your duaa :raised_hands:.

:fire: To see our technology live in action, visit [tarteel.io]. :fire:

## Getting Started :beginner:

### Prerequisites 

We use Python 3.7 for our development.
However, any Python above 3.6 should work.
For audio pre-processing, we use `ffmpeg` and `ffprobe`.
Make sure you install these using your system package manager.

**Mac OS**

```bash
brew install ffmpeg
```

**Linux**

```bash
sudo apt install ffmpeg
```

Then install the Python dependencies from [`requirements.txt`](requirements.txt).

```bash
pip3 install -r requirements.txt
```

### Usage

Use the `-h`/`--help` flag for more info on how to use each script.

This repo is structured as follows:

**Root**

* [`download.py`]: Download the Tarteel dataset
* [`create_train_test_split.py`]: Create train/test/validation split csv files.
* `generate_alphabet|vocabulary.py`: Generate all unique letters/ayahs in the Quran in a text file.
* [`generate_csv_deepspeech.py`]: Create a CSV file for training with DeepSpeech.

### Wiki :scroll:

Check out the [wiki] for instructions on how to download and pre-process the data, as well as how to start training models.

## Contributing :100:
Check out [`CONTRIBUTING.md`](CONTRIBUTING.md) to start contributing to Tarteel-ML!

[tarteel.io]: https://www.tarteel.io
[wiki]: https://github.com/Tarteel-io/Tarteel-ML/wiki
[`generate_csv_deepspeech.py`]: generate_csv_deepspeech.py
[`create_train_test_split.py`]: create_train_test_split.py
[`download.py`]: download.py
