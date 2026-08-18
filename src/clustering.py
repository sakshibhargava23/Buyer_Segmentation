"""Step 4–6 – Clustering, evaluation, and segment interpretation."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

from src.utils import FIGURES_DIR, SEGMENT_NAMES


@dataclass
class ClusteringResult:
    optimal_k: int
    kmeans_labels: np.ndarray
    hierarchical_labels: np.ndarray
    silhouette: float
    kmeans_model: KMeans
    elbow_k: list[int]
    elbow_inertia: list[float]
    elbow_silhouette: list[float]


def find_optimal_k(
    features: np.ndarray, k_range: range = range(2, 11)
) -> tuple[int, list[int], list[float], list[float]]:
    """Use elbow method and silhouette scores to select optimal cluster count."""
    inertias: list[float] = []
    silhouettes: list[float] = []
    ks = list(k_range)

    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(features)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(features, labels))

    optimal_k = ks[int(np.argmax(silhouettes))]
    return optimal_k, ks, inertias, silhouettes


def train_clustering(features: np.ndarray, n_clusters: int = 4) -> ClusteringResult:
    """Train K-Means and hierarchical clustering models."""
    optimal_k, ks, inertias, silhouettes = find_optimal_k(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(features)
    silhouette = silhouette_score(features, kmeans_labels)

    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(features)

    return ClusteringResult(
        optimal_k=optimal_k,
        kmeans_labels=kmeans_labels,
        hierarchical_labels=hierarchical_labels,
        silhouette=silhouette,
        kmeans_model=kmeans,
        elbow_k=ks,
        elbow_inertia=inertias,
        elbow_silhouette=silhouettes,
    )


def map_clusters_to_segments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Assign human-readable segment names based on cluster profiles."""
    result = df.copy()
    result["cluster"] = labels

    profiles = (
        result.groupby("cluster")
        .agg(
            avg_age=("age", "mean"),
            investment_rate=("is_investor", "mean"),
            corporate_rate=("client_type", lambda x: (x == "Company").mean()),
            avg_investment=("total_investment", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            loan_rate=("loan_applied_flag", "mean"),
            avg_properties=("num_properties", "mean"),
        )
        .reset_index()
    )

    segment_map: dict[int, int] = {}
    used: set[int] = set()

    def assign_cluster(mask: pd.Series, target: int) -> None:
        candidates = profiles[~profiles["cluster"].isin(used)]
        if candidates.empty:
            return
        aligned = mask.reindex(candidates.index, fill_value=False)
        idx = candidates[aligned].sort_values("avg_investment", ascending=False)
        if idx.empty:
            idx = candidates.sort_values("avg_investment", ascending=False)
        cluster_id = int(idx.iloc[0]["cluster"])
        segment_map[cluster_id] = target
        used.add(cluster_id)

    assign_cluster(profiles["corporate_rate"] > profiles["corporate_rate"].median(), 2)
    assign_cluster(
        (profiles["investment_rate"] > profiles["investment_rate"].median())
        & (profiles["avg_investment"] > profiles["avg_investment"].median()),
        0,
    )
    assign_cluster(
        (profiles["avg_satisfaction"] >= profiles["avg_satisfaction"].median())
        & (profiles["avg_investment"] >= profiles["avg_investment"].median()),
        3,
    )
    assign_cluster(profiles["loan_rate"] > profiles["loan_rate"].median(), 1)

    for cluster_id in profiles["cluster"]:
        cid = int(cluster_id)
        if cid not in segment_map:
            remaining = [i for i in range(4) if i not in segment_map.values()]
            segment_map[cid] = remaining[0] if remaining else 0

    result["segment_id"] = result["cluster"].map(segment_map)
    result["segment_name"] = result["segment_id"].map(SEGMENT_NAMES)
    return result


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics per buyer segment."""
    summary = (
        df.groupby(["segment_id", "segment_name"])
        .agg(
            buyers=("client_id", "count"),
            avg_age=("age", "mean"),
            pct_investment=("is_investor", "mean"),
            pct_loan=("loan_applied_flag", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            avg_investment=("total_investment", "mean"),
            avg_properties=("num_properties", "mean"),
            top_country=("country", lambda x: x.mode().iloc[0] if len(x) else "N/A"),
            top_region=("region", lambda x: x.mode().iloc[0] if len(x) else "N/A"),
            top_referral=("referral_channel", lambda x: x.mode().iloc[0] if len(x) else "N/A"),
        )
        .reset_index()
    )

    for col in ["pct_investment", "pct_loan"]:
        summary[col] = (summary[col] * 100).round(1)

    numeric_cols = ["avg_age", "avg_satisfaction", "avg_investment", "avg_properties"]
    summary[numeric_cols] = summary[numeric_cols].round(2)
    return summary


def save_evaluation_plots(result: ClusteringResult, output_dir=FIGURES_DIR) -> None:
    """Save elbow and silhouette evaluation charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(result.elbow_k, result.elbow_inertia, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(result.elbow_k, result.elbow_silhouette, marker="o", color="green")
    axes[1].axvline(result.optimal_k, color="red", linestyle="--", label=f"Best k={result.optimal_k}")
    axes[1].set_title("Silhouette Score by k")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_dir / "cluster_evaluation.png", dpi=150)
    plt.close(fig)


def save_segment_profile_chart(df: pd.DataFrame, output_dir=FIGURES_DIR) -> None:
    """Save a simple segment comparison chart for the overview page."""
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = (
        df.groupby("segment_name")
        .agg(
            avg_investment=("total_investment", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            loan_rate=("loan_applied_flag", "mean"),
        )
        .reset_index()
    )
    profile["avg_investment"] = profile["avg_investment"] / 1000
    profile["loan_rate"] *= 100

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(profile))
    width = 0.25
    ax.bar([i - width for i in x], profile["avg_investment"], width, label="Avg Investment ($K)")
    ax.bar(x, profile["avg_satisfaction"], width, label="Avg Satisfaction (1-5)")
    ax.bar([i + width for i in x], profile["loan_rate"], width, label="Loan Rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(profile["segment_name"], rotation=12, ha="right")
    ax.set_title("Segment Profile Comparison")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "segment_profile.png", dpi=150)
    plt.close(fig)


def save_segment_distribution(df: pd.DataFrame, output_dir=FIGURES_DIR) -> None:
    """Save cluster distribution bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = df["segment_name"].value_counts().reset_index()
    counts.columns = ["segment_name", "count"]

    plt.figure(figsize=(8, 5))
    sns.barplot(data=counts, x="segment_name", y="count", hue="segment_name", legend=False, palette="viridis")
    plt.title("Buyer Segment Distribution")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "segment_distribution.png", dpi=150)
    plt.close()


def save_model(model: KMeans, path: str) -> None:
    joblib.dump(model, path)
