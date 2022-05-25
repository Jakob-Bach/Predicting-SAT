"""Run experiments

Script to run the full experimental pipeline. Should be run after dataset preparation, as this
script requires the prediction datasets as inputs. Saves its results for evaluation.

Usage: python -m run_experiments --help
"""


import argparse
import multiprocessing
import pathlib
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import tqdm

import prepare_datasets
import prediction


# Define experimental design as the cross-product of cross-validation folds, instance sets, and
# feature sets for the full "dataset".
# Return a list of experimental settings (used for calling "run_experimental_setting()").
def define_experimental_settings(dataset: pd.DataFrame) -> Sequence[Dict[str, Any]]:
    return [{'dataset': dataset, 'fold_id': fold_id, 'instances_name': instances_name,
             'features_name': featureset_name}
            for fold_id in range(prediction.NUM_CV_FOLDS)
            for instances_name in prepare_datasets.INSTANCE_FILTER_RULES
            for featureset_name in prepare_datasets.FEATURE_FILTER_RULES]


# Evaluate predictions on "dataset" limited to one fold, one instance set, and one feature set.
# Return a table with evaluation metrics.
def run_experimental_setting(dataset: pd.DataFrame, fold_id: int, instances_name: str,
                             features_name: str) -> pd.DataFrame:
    instance_filter_func = prepare_datasets.INSTANCE_FILTER_RULES[instances_name]
    feature_filter_func = prepare_datasets.FEATURE_FILTER_RULES[features_name]
    dataset = dataset[instance_filter_func(dataset)]
    X = dataset[feature_filter_func(dataset)]
    y = dataset['meta.result']
    results = prediction.predict_and_evaluate(X=X, y=y, fold_id=fold_id)
    results['instances_name'] = instances_name
    results['features_name'] = features_name
    return results


# Main-routine: run complete experimental pipeline. To that end, read datasets from "data_dir" and
# save results to "results_dir". "n_processes" controls parallelization.
def run_experiments(data_dir: pathlib.Path, results_dir: pathlib.Path,
                    n_processes: Optional[int] = None) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError('Dataset directory does not exist.')
    if not results_dir.is_dir():
        print('Results directory does not exist. We create it.')
        results_dir.mkdir(parents=True)
    if any(results_dir.iterdir()):
        print('Results directory is not empty. Files might be overwritten, but not deleted.')
    dataset = pd.read_csv(data_dir / 'dataset.csv')
    experimental_settings = define_experimental_settings(dataset=dataset)
    progress_bar = tqdm.tqdm(total=len(experimental_settings))
    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(run_experimental_setting, kwds=setting,
                                        callback=lambda x: progress_bar.update())
               for setting in experimental_settings]
    process_pool.close()
    process_pool.join()
    progress_bar.close()
    results = pd.concat([x.get() for x in results])
    results.to_csv(results_dir / 'results.csv', index=False)


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs the complete experimental pipeline. Might take a while.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/', dest='data_dir',
                        help='Directory with input data, i.e., prediction datasets.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/', dest='results_dir',
                        help='Directory for output data, i.e., experimental results.')
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for multi-processing (default: all cores).')
    print('Experimental pipeline started.')
    run_experiments(**vars(parser.parse_args()))
    print('Experimental pipeline executed successfully.')
