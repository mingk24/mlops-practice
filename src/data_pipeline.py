import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
from datetime import datetime


def load_and_prepare(test_size=0.2, random_state=42):
    """데이터 로드 → 분할 → 스케일링까지 한 번에"""
    data = load_wine()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    stats = {
        "timestamp": datetime.now().isoformat(),
        "n_train": len(X_train),
        "n_test":  len(X_test),
        "feature_means": X_train_scaled.mean(axis=0).tolist(),
        "feature_stds":  X_train_scaled.std(axis=0).tolist(),
        "feature_names": list(data.feature_names),
    }

    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ 데이터 준비 완료: train={len(X_train)}, test={len(X_test)}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, stats