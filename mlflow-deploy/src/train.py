import os
import json
import argparse
import warnings
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

def load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_data(url: str, sep: str) -> pd.DataFrame:
    # Carga desde URL; localmente puedes usar data/winequality-red.csv si ya lo tienes.
    return pd.read_csv(url, sep=sep)

def binarize_quality(df: pd.DataFrame, target_col: str, thr: int):
    y = (df[target_col] >= thr).astype(int)  # 1=buena calidad
    X = df.drop(columns=[target_col])
    return X, y

def build_pipeline(model_kind: str, cfg):
    # Todas las columnas del dataset son numéricas -> escalamos numéricas
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), "passthrough")],
        remainder="drop",
        verbose_feature_names_out=False
    )

    if model_kind == "logreg":
        clf = LogisticRegression(C=cfg["model"]["C"], max_iter=cfg["model"]["max_iter"])
    elif model_kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=cfg["model"]["n_estimators"],
            max_depth=cfg["model"]["max_depth"],
            random_state=cfg["data"]["random_state"]
        )
    else:
        raise ValueError("model.kind debe ser 'logreg' o 'rf'")

    pipe = Pipeline(steps=[("prep", pre), ("clf", clf)])
    return pipe

def main(config_path: str):
    cfg = load_config(config_path)

    # MLflow
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])

    # 1) Datos
    df = load_data(cfg["data"]["url"], cfg["data"]["sep"])
    X, y = binarize_quality(df, cfg["data"]["target"], cfg["data"]["binarize_threshold"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y
    )

    # 2) Modelo (pipeline)
    pipe = build_pipeline(cfg["model"]["kind"], cfg)

    with mlflow.start_run(run_name="train"):
        # Log parámetros
        mlflow.log_params({
            "model_kind": cfg["model"]["kind"],
            "test_size":  cfg["data"]["test_size"],
            "random_state": cfg["data"]["random_state"]
        })
        if cfg["model"]["kind"] == "logreg":
            mlflow.log_param("C", cfg["model"]["C"])
            mlflow.log_param("max_iter", cfg["model"]["max_iter"])
        else:
            mlflow.log_param("n_estimators", cfg["model"]["n_estimators"])
            mlflow.log_param("max_depth", cfg["model"]["max_depth"])

        # Entrenamiento
        pipe.fit(X_train, y_train)

        # 3) Evaluación
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)

        mlflow.log_metrics({"accuracy": acc, "f1": f1})

        # 4) Firma y ejemplo
        signature = mlflow.models.signature.infer_signature(X_train, pipe.predict(X_train))
        input_example = X_train.head(2)

        # 5) Log del modelo en MLflow
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=None  # opcional, para registry
        )

        # 6) Guardar artefactos locales para el workflow (subir como artifacts)
        Path("artifacts").mkdir(exist_ok=True)
        joblib.dump(pipe, "artifacts/model.pkl")
        with open("artifacts/metrics.json", "w", encoding="utf-8") as f:
            json.dump({"accuracy": acc, "f1": f1}, f, indent=2)

        print(f"Accuracy={acc:.4f}  F1={f1:.4f}")
        print("Run ID:", mlflow.active_run().info.run_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
