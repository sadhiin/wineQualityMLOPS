import os
import yaml
import pytest
import json

@pytest.fixture
def config(config_path = "params.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def schema_input(schemapath = schema_path):
    with open(schemapath) as f:
        schema = json.load(f)
    return schema