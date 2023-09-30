# load the train and test data
# training algo
# save the metrices, algorithms perameters

import os
import warnings
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import median_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import ElasticNet, ElasticNetCV
from urllib.parse import urlparse

from get_data import read_params
import argparse
import joblib
import json


def eval_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = median_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    return rmse, mae, r2


def train_and_eval_Elasticnet(config_path):
    config = read_params(config_path)

    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=',')
    test = pd.read_csv(test_data_path, sep=',')

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    train_y = train[target]
    test_y = test[target]

    model = ElasticNet(alpha=alpha,
                       l1_ratio=l1_ratio,
                       random_state=random_state,
                       warn_only=True)

    model.fit(X=train_x, y=train_y)

    predicted_qualities = model.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # reporting the informations
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]
    with open(scores_file, 'w') as f:
        matric = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(matric, f, indent=4)

    with open(params_file, 'w') as f:
        param = {
            "alpha": alpha,
            "l1_ratio": l1_ratio
        }
        json.dump(param, f, indent=4)

    # saving the model file
    os.makedirs(model_dir, exist_ok=True)

    # timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # model_filename = f"model_{timestamp}.joblib"
    # model_path = os.path.join(model_dir, model_filename)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(model, model_path)


def train_and_eval_ElasticnetCV(config_path):
    config = read_params(config_path)

    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    l1_ratio = config["estimators"]["ElasticNetCV"]["params"]["l1_ratio"]
    cv = config["estimators"]["ElasticNetCV"]["params"]["cv"]
    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=',')
    test = pd.read_csv(test_data_path, sep=',')

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    train_y = train[target]
    test_y = test[target]

    model = ElasticNetCV(l1_ratio=l1_ratio, cv=cv, random_state=random_state)
    model.fit(train_x, train_y)

    predicted_qualities = model.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("ElasticnetCV model (l1_ratio=%f, cv=%f):" % (l1_ratio, cv))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # reporting the informations
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]
    with open(scores_file, 'w') as f:
        matric = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(matric, f, indent=4)

    with open(params_file, 'w') as f:
        param = {
            "l1_ratio": l1_ratio,
            "cv": cv
        }
        json.dump(param, f, indent=4)

    # saving the model file
    os.makedirs(model_dir, exist_ok=True)

    # timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # model_filename = f"model_{timestamp}.joblib"
    # model_path = os.path.join(model_dir, model_filename)
    model_path = os.path.join(model_dir, "modelcv.joblib")

    joblib.dump(model, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")

    parsed_args = args.parse_args()

    train_and_eval_Elasticnet(config_path=parsed_args.config)
    # train_and_eval_ElasticnetCV(config_path= parsed_args.config)
