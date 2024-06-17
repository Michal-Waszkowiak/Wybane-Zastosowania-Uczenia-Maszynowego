import keras
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature


mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("MLflow Quickstart")
mlflow.set_experiment("/wine-quality")




# def TODO01():
#     # with mlflow.start_run():
#     #     mlflow.log_metric("foo", 1)
#     #     mlflow.log_metric("bar", 2)
#
#     # Load the Iris dataset
#     X, y = datasets.load_iris(return_X_y=True)
#
#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Define the model hyperparameters
#     params = {"solver": "lbfgs", "max_iter": 1000, "multi_class": "auto", "random_state": 8888}
#
#     # Train the model
#     lr = LogisticRegression(**params)
#     lr.fit(X_train, y_train)
#
#     # Predict on the test set
#     y_pred = lr.predict(X_test)
#
#     # Calculate accuracy as a target loss metric
#     accuracy = accuracy_score(y_test, y_pred)
#
#     # Start an MLflow run
#     with mlflow.start_run():
#         # Log the hyperparameters
#         mlflow.log_params(params)
#
#         # Log the loss metric
#         mlflow.log_metric("accuracy", accuracy)
#
#         # Set a tag that we can use to remind ourselves what this run was for
#         mlflow.set_tag("Training Info", "Basic LR model for iris data")
#
#         # Infer the model signature
#         signature = infer_signature(X_train, lr.predict(X_train))
#
#         # Log the model
#         model_info = mlflow.sklearn.log_model(
#             sk_model=lr,
#             artifact_path="iris_model",
#             signature=signature,
#             input_example=X_train,
#             registered_model_name="tracking-quickstart",
#         )
#
#         loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
#
#         predictions = loaded_model.predict(X_test)
#
#         iris_feature_names = datasets.load_iris().feature_names
#
#         # Convert X_test validation feature data to a Pandas DataFrame
#         result = pd.DataFrame(X_test, columns=iris_feature_names)
#
#         # Add the actual classes to the DataFrame
#         result["actual_class"] = y_test
#
#         # Add the model predictions to the DataFrame
#         result["predicted_class"] = predictions
#
#         print(result[:4])



# Load dataset
data = pd.read_csv(
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
    sep=";",
)

# Split the data into training, validation, and test sets
train, test = train_test_split(data, test_size=0.25, random_state=42)
train_x = train.drop(["quality"], axis=1).values
train_y = train[["quality"]].values.ravel()
test_x = test.drop(["quality"], axis=1).values
test_y = test[["quality"]].values.ravel()
train_x, valid_x, train_y, valid_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)
signature = infer_signature(train_x, train_y)

def train_model(params, epochs, train_x, train_y, valid_x, valid_y, test_x, test_y):
    # Define model architecture
    model = keras.Sequential(
        [
            keras.Input([train_x.shape[1]]),
            keras.layers.Normalization(mean=np.mean(train_x), variance=np.var(train_x)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1),
        ]
    )

    # Compile model
    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=params["lr"], momentum=params["momentum"]
        ),
        loss="mean_squared_error",
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    # Train model with MLflow tracking
    with mlflow.start_run(nested=True):
        model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            epochs=epochs,
            batch_size=64,
        )
        # Evaluate the model
        eval_result = model.evaluate(valid_x, valid_y, batch_size=64)
        eval_rmse = eval_result[1]

        # Log parameters and results
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse", eval_rmse)

        # Log model
        mlflow.tensorflow.log_model(model, "model", signature=signature)

        return {"loss": eval_rmse, "status": STATUS_OK, "model": model}



def objective(params):
    # MLflow will track the parameters and results for each run
    result = train_model(
        params,
        epochs=3,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        test_x=test_x,
        test_y=test_y,
    )
    return result

space = {
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-1)),
    "momentum": hp.uniform("momentum", 0.0, 1.0),
}


with mlflow.start_run():
    # Conduct the hyperparameter search using Hyperopt
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=8,
        trials=trials,
    )

    # Fetch the details of the best run
    best_run = sorted(trials.results, key=lambda x: x["loss"])[0]

    # Log the best parameters, loss, and model
    mlflow.log_params(best)
    mlflow.log_metric("eval_rmse", best_run["loss"])
    mlflow.tensorflow.log_model(best_run["model"], "model", signature=signature)

    # Print out the best parameters and corresponding loss
    print(f"Best parameters: {best}")
    print(f"Best eval rmse: {best_run['loss']}")
