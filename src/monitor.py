import numpy as np
import json
import os
from scipy.stats import ks_2samp

def detect_drift(reference_stats_path, new_data, threshold=0.05):
    abs_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        reference_stats_path
    )
    with open(abs_path) as f:
        ref = json.load(f)

    ref_means     = np.array(ref["feature_means"])
    ref_stds      = np.array(ref["feature_stds"])
    feature_names = ref["feature_names"]

    np.random.seed(42)
    ref_samples = np.random.normal(ref_means, ref_stds, size=(500, len(ref_means)))

    print("=" * 55)
    print(f"{'피처':<28} {'p-value':>8}  {'상태':>8}")
    print("=" * 55)

    drift_count = 0
    for i, name in enumerate(feature_names):
        _, p = ks_2samp(ref_samples[:, i], new_data[:, i])
        drifted = p < threshold
        if drifted:
            drift_count += 1
        status = "⚠️  DRIFT" if drifted else "✅  OK"
        print(f"{name[:28]:<28} {p:>8.4f}  {status}")

    print("=" * 55)
    if drift_count > 0:
        print(f"🚨 드리프트 감지: {drift_count}/{len(feature_names)}개 피처 이상")
        print("   → 모델 재학습 트리거 권장")
    else:
        print("✅ 정상 범위 — 재학습 불필요")

    return drift_count > 0