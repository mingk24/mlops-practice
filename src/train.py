
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_CONFIGS = [
    {
        "name": "random-forest",
        "model": RandomForestClassifier,
        "params": {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    },
    {
        "name": "gradient-boosting",
        "model": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
    },
    {
        "name": "logistic-regression",
        "model": LogisticRegression,
        "params": {"max_iter": 500, "random_state": 42}
    },
]

def train_all(X_train, X_test, y_train, y_test):
    mlflow.set_experiment("wine-classification")
    best_run_id, best_acc = None, 0

    for cfg in MODEL_CONFIGS:
        with mlflow.start_run(run_name=cfg["name"]) as run:
            mlflow.log_params(cfg["params"])
            mlflow.log_param("model_type", cfg["name"])

            model = cfg["model"](**cfg["params"])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average="weighted")
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_weighted", f1)

            report = classification_report(y_test, y_pred)
            with open("classification_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("classification_report.txt")

            mlflow.sklearn.log_model(
                model, "model",
                registered_model_name=f"wine-{cfg['name']}"
            )

            print(f"[{cfg['name']}] accuracy={acc:.4f}, f1={f1:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_run_id = run.info.run_id

    print(f"\n🏆 최고 모델 run_id: {best_run_id} (accuracy={best_acc:.4f})")
    return best_run_id