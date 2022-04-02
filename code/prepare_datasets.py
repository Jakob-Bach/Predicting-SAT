"""Prepare datasets

Script to download databases from GBD and create datasets for the experimental pipeline.

Usage: python -m prepare_datasets --help
"""


import argparse
import pathlib


# Main-routine: download, pre-process, and save (to "data_dir") datasets.
def prepare_datasets(data_dir: pathlib.Path) -> None:
    if not data_dir.is_dir():
        print('Dataset directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if any(data_dir.iterdir()):
        print('Dataset directory is not empty. Files might be overwritten, but not deleted.')


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