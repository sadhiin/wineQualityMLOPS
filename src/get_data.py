import os
import yaml
import pandas as pd
import argparse


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def get_data(config_path):
    config = read_params(config_path)
    print(config)


if __name__=='__main__':
    args = argparse.ArgumentParser()
    # yaml_path = os.path.join("..",'params.yaml')
    args.add_argument('--config', default='params.yaml')

    parsed_args = args.parse_args()

    get_data(config_path= parsed_args.config)