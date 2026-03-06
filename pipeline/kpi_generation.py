import pandas as pd
from pathlib import Path


def compute_kpis(
    input_path: str = "./output/cleaned_data.csv",
    output_path: str = "./output/kpi_metrics.csv",
) -> pd.DataFrame:
    """
    Compute core business KPIs from the cleaned retail dataset and
    write them to a flat metrics table suitable for BI tools.
    """
    df = pd.read_csv(input_path, parse_dates=["order_date"])

    records: list[dict] = []

    # Overall KPIs
    total_revenue = df["order_value"].sum()
    avg_order_value = df["order_value"].mean()
    total_orders = len(df)
    unique_customers = df["customer_id"].nunique()

    records.extend(
        [
            {
                "metric_type": "overall",
                "group_key": "ALL",
                "metric_name": "total_revenue",
                "metric_value": float(total_revenue),
            },
            {
                "metric_type": "overall",
                "group_key": "ALL",
                "metric_name": "average_order_value",
                "metric_value": float(avg_order_value),
            },
            {
                "metric_type": "overall",
                "group_key": "ALL",
                "metric_name": "total_orders",
                "metric_value": float(total_orders),
            },
            {
                "metric_type": "overall",
                "group_key": "ALL",
                "metric_name": "unique_customers",
                "metric_value": float(unique_customers),
            },
        ]
    )

    # Monthly revenue
    monthly = (
        df.set_index("order_date")["order_value"]
        .resample("M")
        .sum()
        .rename("revenue")
        .reset_index()
    )
    for _, row in monthly.iterrows():
        records.append(
            {
                "metric_type": "monthly_revenue",
                "group_key": row["order_date"].strftime("%Y-%m"),
                "metric_name": "revenue",
                "metric_value": float(row["revenue"]),
            }
        )

    # Revenue and orders by product category
    cat_group = df.groupby("product_category").agg(
        revenue=("order_value", "sum"), orders=("order_id", "count")
    )
    cat_group = cat_group.sort_values("revenue", ascending=False)

    for category, row in cat_group.iterrows():
        records.append(
            {
                "metric_type": "category_revenue",
                "group_key": category,
                "metric_name": "revenue",
                "metric_value": float(row["revenue"]),
            }
        )
        records.append(
            {
                "metric_type": "category_orders",
                "group_key": category,
                "metric_name": "orders",
                "metric_value": float(row["orders"]),
            }
        )

    # Purchase frequency (average orders per customer)
    orders_per_customer = (
        df.groupby("customer_id")["order_id"].count().rename("orders")
    )
    avg_orders_per_customer = orders_per_customer.mean()

    records.append(
        {
            "metric_type": "purchase_frequency",
            "group_key": "ALL",
            "metric_name": "avg_orders_per_customer",
            "metric_value": float(avg_orders_per_customer),
        }
    )

    metrics_df = pd.DataFrame.from_records(records)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_file, index=False)

    return metrics_df


def main() -> None:
    metrics_df = compute_kpis()
    print(f"KPI metrics written with shape {metrics_df.shape}")


if __name__ == "__main__":
    main()

