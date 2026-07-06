import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("test-experiment")

with mlflow.start_run(run_name="test-run"):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("model", "dummy")
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("roc_auc", 0.90)

    with open("artefact_test.txt", "w") as f:
        f.write("Test artefact MLflow - Examen MLOps")
    mlflow.log_artifact("artefact_test.txt")

print("Experience loggee ! Va voir http://127.0.0.1:5000")