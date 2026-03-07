import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_customer_features(
    input_path: str = "../output/cleaned_data.csv",
) -> pd.DataFrame:
    """
    Aggregate order-level data into per-customer behavioral features.
    """
    df = pd.read_csv(input_path, parse_dates=["order_date"])

    agg = (
        df.groupby("customer_id")
        .agg(
            total_spending=("order_value", "sum"),
            order_count=("order_id", "count"),
        )
        .reset_index()
    )

    agg["avg_order_value"] = agg["total_spending"] / agg["order_count"]

    return agg


def run_kmeans_segmentation(
    features_df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Apply K-Means clustering on numeric customer features.
    """
    feature_cols = ["total_spending", "order_count", "avg_order_value"]

    X = features_df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(X_scaled)

    result = features_df.copy()
    result["segment"] = cluster_labels

    return result


def segment_customers(
    input_path: str = "./output/cleaned_data.csv",
    output_path: str = "./output/customer_segments.csv",
    n_clusters: int = 4,
) -> pd.DataFrame:
    """
    Build customer features and run K-Means segmentation, then save the result.
    """
    features_df = build_customer_features(input_path=input_path)
    segments_df = run_kmeans_segmentation(
        features_df=features_df, n_clusters=n_clusters
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    segments_df.to_csv(output_file, index=False)

    return segments_df


def main() -> None:
    segments_df = segment_customers()
    print(
        f"Customer segments written with shape {segments_df.shape} "
        f"and {segments_df['segment'].nunique()} segments."
    )


if __name__ == "__main__":
    main()

