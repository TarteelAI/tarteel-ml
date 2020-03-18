# Compare Models

Compare the accuracy of different ML Models.

## Usage

You will need to configure your GCP credentials before usage:

```sh
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

Then run the script:

```sh
# Detailed usage
$ python compare_models.py --help

# Run for 100 samples
$ python compare_models.py \
  --directory "/path/to/data" \
  --ds-config "/path/to/ds_config.yaml" \
  --random 100
```

## Results

Outputs the results in a `csv` format as:

```csv
'file', 'surah', 'ayah',
  'gs_result', 'gs_time', 'gs_exact_match', 'gs_partial_match', 'gs_score', 'gs_norm_score',
  'ds_result', 'ds_time', 'ds_exact_match', 'ds_partial_match', 'ds_score', 'ds_norm_score'
...
'percent_exact_match', ...
'percent_partial_match', ...
'average_gs_time', ...
'average_ds_time', ...
'average_score', ...
```
