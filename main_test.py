import argparse
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.6)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.6)
args = parser.parse_args()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    data = pd.read_csv("./winequality.csv", delimiter=";")
    data.to_csv("data/red-wine-quality.csv", index=False)

    train, test = train_test_split(data)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio
    mlflow.set_tracking_uri(
        uri="file:/Users/manikata/Desktop/projects/mlflow/experiments"
    )
    print(f"The set tracking uri is: {mlflow.get_tracking_uri()}")
    exp_id = mlflow.create_experiment(
        name="check_elastic_net_6",
        tags={"version": "v2", "priority": "p2"},
        # artifact_location = Path.cwd().joinpath("my_artifacts").as_uri()
    )
    get_exp = mlflow.get_experiment(exp_id)
    print("Name: {}".format(get_exp.name))
    print("Experiment_id: {}".format(get_exp.experiment_id))
    print("Artifact Location: {}".format(get_exp.artifact_location))
    print("Tags: {}".format(get_exp.tags))
    print("Lifecycle_stage: {}".format(get_exp.lifecycle_stage))
    print("Creation timestamp: {}".format(get_exp.creation_time))
    # exp = mlflow.set_experiment(experiment_name="run_1")
    # with mlflow.start_run(experiment_id=exp.experiment_id):
    with mlflow.start_run(experiment_id=exp_id):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        """ mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio) """
        mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(lr, "mymodel")

        # run = mlflow.active_run()
        run = mlflow.last_active_run()
        print("Active run id is {}".format(run.info.run_id))
        print("Active run name is{}".format(run.info.run_name))
