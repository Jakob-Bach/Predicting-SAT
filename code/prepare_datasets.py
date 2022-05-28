"""Prepare datasets

Script to download databases from GBD and create datasets for the experimental pipeline.

Usage: python -m prepare_datasets --help
"""


import argparse
import pathlib
import urllib.request

import gbd_tool.gbd_api
import pandas as pd
import tqdm


DATABASE_NAMES = ['base', 'gate', 'meta', 'satzilla']

INSTANCE_FILTER_RULES = {
    'SC 2020': lambda dataset: dataset['meta.track'].fillna('').str.contains('main_2020'),
    'SC 2021': lambda dataset: dataset['meta.track'].fillna('').str.contains('main_2021'),
    'Solved': lambda dataset: dataset['meta.result'] != 'unknown'
}  # functions returning boolean pd.Series to select instances (rows) from the overall dataset

FEATURE_FILTER_RULES = {
    'Base + gate': lambda dataset: [x for x in dataset.columns if 'base.' in x or 'gate.' in x],
    'SATzilla 2012': lambda dataset: [x for x in dataset.columns if 'satzilla.' in x]
}  # functions returning list of strings to select features (columns) from the overall dataset


# Download database files and save them in original format + CSV in "data_dir".
def download_and_save_databases(data_dir: pathlib.Path) -> None:
    for db_name in tqdm.tqdm(DATABASE_NAMES, desc='Downloading'):
        urllib.request.urlretrieve(url=f'https://gbd.iti.kit.edu/getdatabase/{db_name}_db',
                                   filename=data_dir / f'{db_name}.db')
        with gbd_tool.gbd_api.GBD(db_list=[str(data_dir / f'{db_name}.db')]) as api:
            features = api.get_features()
            features.remove('hash')  # will be added to result anyway, so avoid duplicates
            database = pd.DataFrame(api.query_search(resolve=features), columns=['hash'] + features)
            database.to_csv(data_dir / f'{db_name}.csv', index=False)


# Create overall dataset file for experimental pipeline: read GBD databases from "databases_dir" and
# save result to "dataset_dir".
def merge_databases(databases_dir: pathlib.Path, dataset_dir: pathlib.Path) -> None:
    dataset = pd.read_csv(databases_dir / 'meta.csv')
    dataset.rename(columns=lambda x: f'meta.{x}' if x != 'hash' else x, inplace=True)
    numeric_cols = []
    for db_name in DATABASE_NAMES:
        if db_name != 'meta':
            database = pd.read_csv(databases_dir / (db_name + '.csv'))
            database.rename(columns=lambda x: f'{db_name}.{x}' if x != 'hash' else x, inplace=True)
            numeric_cols.extend([x for x in database.columns if x != 'hash'])
            dataset = dataset.merge(database, on='hash', how='left', copy=False)
    dataset[numeric_cols] = dataset[numeric_cols].transform(pd.to_numeric, errors='coerce')
    dataset.to_csv(dataset_dir / 'dataset.csv', index=False)


# Main-routine: download, pre-process, and save (to "data_dir") datasets.
def prepare_datasets(data_dir: pathlib.Path) -> None:
    if not data_dir.is_dir():
        print('Dataset directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if any(data_dir.iterdir()):
        print('Dataset directory is not empty. Files might be overwritten, but not deleted.')
    download_and_save_databases(data_dir=data_dir)
    merge_databases(databases_dir=data_dir, dataset_dir=data_dir)


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Retrieves databases from GBD, prepares datasets for the experiment pipeline,' +
        ' and stores them in the specified directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=pathlib.Path, default='data/', dest='data_dir',
                        help='Output directory for datasets.')
    print('Dataset preparation started')
    prepare_datasets(**vars(parser.parse_args()))
    print('Datasets prepared and saved.')
