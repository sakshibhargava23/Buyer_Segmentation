#!/usr/bin/env python3
"""Train buyer segmentation clustering models and export artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import (  # noqa: E402
    cluster_summary,
    map_clusters_to_segments,
    save_dendrogram,
    save_evaluation_plots,
    save_model,
    save_segment_distribution,
    train_clustering,
)
from src.data_cleaning import build_master_dataset  # noqa: E402
from src.feature_engineering import prepare_features, save_preprocessor  # noqa: E402
from src.utils import DATA_DIR, MODELS_DIR, OUTPUTS_DIR  # noqa: E402


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    clients_path = DATA_DIR / "clients.csv"
    properties_path = DATA_DIR / "properties.csv"

    print("Step 1: Cleaning and merging data...")
    master = build_master_dataset(str(clients_path), str(properties_path))
    master.to_csv(OUTPUTS_DIR / "master_dataset.csv", index=False)
    print(f"  Master dataset: {master.shape[0]} clients, {master.shape[1]} features")

    print("Step 2-3: Encoding and scaling features...")
    features, preprocessor, label_encoders = prepare_features(master)
    save_preprocessor(preprocessor, label_encoders, str(MODELS_DIR / "preprocessor.pkl"))
    print(f"  Feature matrix shape: {features.shape}")

    print("Step 4-5: Training clustering models...")
    result = train_clustering(features, n_clusters=4)
    print(f"  Optimal k (silhouette): {result.optimal_k}")
    print(f"  K-Means silhouette (k=4): {result.silhouette:.4f}")

    print("Step 6: Interpreting clusters...")
    segmented = map_clusters_to_segments(master, result.kmeans_labels)
    summary = cluster_summary(segmented)

    segmented.to_csv(MODELS_DIR / "segmented_clients.csv", index=False)
    summary.to_csv(MODELS_DIR / "cluster_summary.csv", index=False)
    save_model(result.kmeans_model, str(MODELS_DIR / "kmeans_model.pkl"))

    metrics = {
        "optimal_k_silhouette": result.optimal_k,
        "trained_k": 4,
        "silhouette_k4": round(result.silhouette, 4),
        "total_clients": int(len(segmented)),
        "segments": summary.to_dict(orient="records"),
    }
    with open(MODELS_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    save_evaluation_plots(result)
    save_dendrogram(features)
    save_segment_distribution(segmented)

    print("\nSegment Summary:")
    print(summary.to_string(index=False))
    print(f"\nArtifacts saved to {MODELS_DIR} and {OUTPUTS_DIR / 'figures'}")


if __name__ == "__main__":
    main()
