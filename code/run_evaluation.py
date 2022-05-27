"""Run evaluation

Script to compute summary statistics and create plots for the paper. Should be run after the
experimental pipeline, as this script requires the pipeline's outputs as inputs.

Usage: python -m run_evaluation --help
"""


import argparse
import pathlib

import pandas as pd

import prepare_datasets


# Main-routine: run complete evaluation pipeline. To that end, read the dataset from "data_dir",
# results from the "results_dir", and save plots to the "plot_dir". Print some statistics.
def evaluate(data_dir: pathlib.Path, results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    if not results_dir.is_dir():
        raise FileNotFoundError('Results directory does not exist.')
    if not plot_dir.is_dir():
        print('Plot directory does not exist. We create it.')
        plot_dir.mkdir(parents=True)
    if any(plot_dir.glob('*.pdf')) > 0:
        print('Plot directory is not empty. Files might be overwritten, but not deleted.')

    dataset = pd.read_csv(data_dir / 'dataset.csv')
    results = pd.read_csv(results_dir / 'results.csv')

    print('\nHow is are the instance families and SAT instances in them distributed?')
    for instances_name, instance_filter_func in prepare_datasets.INSTANCE_FILTER_RULES.items():
        print('\n-- Instance set:', instances_name, '--')
        agg_data = dataset[instance_filter_func(dataset)].groupby('meta.family')['meta.result']
        agg_data = agg_data.agg([len, lambda x: (x == 'sat').sum() / len(x) * 100])
        agg_data.sort_values(by='len', ascending=False, inplace=True)
        agg_data.rename(columns={'len': 'instances', '<lambda_0>': '% sat'}, inplace=True)
        agg_data['% instances'] = agg_data['instances'] / agg_data['instances'].sum() * 100
        agg_data['cum % instances'] = agg_data['% instances'].cumsum()
        print(agg_data.round(2))

    print('\nHow do prediction results differ between models?')
    print(results[results['num_features'] == 'all'].groupby('model_name')[['train_mcc', 'test_mcc']].agg(
        ['mean', 'median', 'std']).transpose().round(2))

    print('\nHow do prediction results (test MCC) differ between instance sets?')
    print(results[results['num_features'] == 'all'].groupby('instances_name')['test_mcc'].agg(
        ['mean', 'median', 'std']).transpose().round(2))

    print('\nHow do prediction results (test MCC) differ between feature sets?')
    print(results[results['num_features'] == 'all'].groupby('features_name')['test_mcc'].agg(
        ['mean', 'median', 'std']).transpose().round(2))

    print('\nWhat are the most important features in predictions (on average)?')
    for features_name, features_filter_func in prepare_datasets.FEATURE_FILTER_RULES.items():
        print('\n-- Feature set:', features_name, '--')
        print(results[results['num_features'] == 'all'][features_filter_func(results)].mean(
            ).sort_values(ascending=False).agg([lambda x: x, 'cumsum']).rename(
                index=lambda x: x.replace('imp.', ''), columns={'<lambda>': 'importance'}).head(10).round(2))

    print('\nHow do prediction results (test MCC) change if filter feature-selection is applied?')
    print(results.groupby(['instances_name', 'model_name', 'num_features'])['test_mcc'].agg(
        ['mean', 'median', 'std']).round(2))


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the paper\'s plots and prints statistics.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/', dest='data_dir',
                        help='Directory with prediction datasets.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/',
                        dest='results_dir', help='Directory with experimental results.')
    parser.add_argument('-p', '--plots', type=pathlib.Path, default='../text/plots',
                        dest='plot_dir', help='Output directory for plots.')
    print('Evaluation started.')
    evaluate(**vars(parser.parse_args()))
    print('Plots created and saved.')
