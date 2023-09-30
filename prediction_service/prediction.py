import os
import yaml
import joblib
import json
import numpy as np

param_path = 'params.yaml'
schema_path = os.path.join('prediction_service', "input_schema.json")


class NotInRange(Exception):
    def __init__(self, message="Input value not in exceptable range.") -> None:
        self.message = message
        super().__init__(self.message)


class NotInCol(Exception):
    def __init__(self, message="Not in feature column") -> None:
        self.message = message
        super().__init__(self.message)


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def prediction(data):
    config = read_params(param_path)
    model_dir = config["webapp_model_dir"]
    model = joblib.load(model_dir)
    pred = model.predict(data).tolist()[0]
    try:
        if 3 <= pred <= 8:
            return pred
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected result"


def get_schema(schemapath=schema_path):
    with open(schemapath) as json_file:
        schema = json.load(json_file)
    return schema


def validate_input(dict_request):
    def _validate_keys(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInCol

    def _validate_values(col, val):
        schema = get_schema()
        if not (schema[col]['min'] <= float(dict_request[col]) <= schema[col]['max']):
            raise NotInRange

    for col, value in dict_request.items():
        _validate_keys(col)
        _validate_values(col, value)
    return True


def form_response(dict_request):
    print(dict_request)
    if validate_input(dict_request):
        data = dict_request.values()
        data = [list(map(float, data))]
        response = prediction(data)
        print(response)
        return response
    else:
        response = "Something went wrong!"
        return response

def api_response(dict_request):
    try:
        if validate_input(dict_request):
            data = np.array([list(dict_request.values())])
            response = prediction(data)
            response = {"response": response}
            return response

    # except NotInRange as e:
    #     response = {"the_expected_range": get_schema(), "response": str(e)}
    #     return response

    # except NotInCol as e:
    #     response = {"the_expected_columns": get_schema().keys(), "response": str(e)}
    #     return response

    except Exception as e:
        response = {"the_expected_range": get_schema(), "response": str(e)}
        return response