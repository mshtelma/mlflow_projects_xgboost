import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import matplotlib as mpl
from hyperopt import fmin, hp, tpe, rand, SparkTrials

import mlflow
import mlflow.xgboost

mpl.use("Agg")

def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost example")

    parser.add_argument(
        "--max-runs",
        type=int,
        default=2,
        help="number of runs",
    )
    return parser.parse_args()

def run_once(params):

    # prepare train „and test data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # enable auto logging
    mlflow.xgboost.autolog()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    with mlflow.start_run(nested=True):

        # train model
        p = {
            "objective": "multi:softprob",
            "num_class": 3,
            "learning_rate": params['lr'],
            "eval_metric": "mlogloss",
            "colsample_bytree": params['colsample_bytree'],
            "subsample": params['subsample'],
            "seed": 42,
        }
        model = xgb.train(p, dtrain, evals=[(dtrain, "train")])

        # evaluate model
        y_proba = model.predict(dtest)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})
        return acc

def main():
    # parse command-line arguments
    args = parse_args()

    space = [
        hp.uniform("lr", 0.0001, 0.9),
        hp.uniform("colsample_bytree", 0.1, 1),
        hp.uniform("subsample", 0.1, 1.0),
    ]
    with mlflow.start_run():
        best = fmin(
            fn=run_once,
            space=space,
            algo=tpe.suggest,
            max_evals=args.max_runs,
            trials=SparkTrials(parallelism=2)
        )
        mlflow.log_param("best_parameters", best)


if __name__ == "__main__":
    main()
