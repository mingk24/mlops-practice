import sys
import os
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from data_pipeline import load_and_prepare
from train import train_all
from monitor import detect_drift

print("━" * 50)
print("  MLOps 실습 파이프라인 시작")
print("━" * 50)

print("\n[1/3] 데이터 파이프라인...")
X_train, X_test, y_train, y_test, scaler, stats = load_and_prepare()

print("\n[2/3] 모델 학습 & 실험 추적...")
best_run_id = train_all(X_train, X_test, y_train, y_test)

print("\n[3/3] 데이터 드리프트 감지...")
print("\n--- 시나리오 A: 정상 신규 데이터 ---")
normal_new = np.random.normal(0, 1, size=(200, X_train.shape[1]))
detect_drift("data/raw/data_stats.json", normal_new)

print("\n--- 시나리오 B: 드리프트된 신규 데이터 ---")
drifted_new = np.random.normal(1.5, 2.0, size=(200, X_train.shape[1]))
detect_drift("data/raw/data_stats.json", drifted_new)

print("\n" + "━" * 50)
print("  완료! MLflow UI 실행: mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("━" * 50)
