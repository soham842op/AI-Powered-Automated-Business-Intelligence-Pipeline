import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_retail_dataset(
    n_orders: int = 5000,
    n_customers: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic retail orders dataset with some intentional missing
    values and duplicate rows to mimic real-world raw data.
    """
    rng = np.random.default_rng(seed)

    customer_ids = np.arange(1, n_customers + 1)

    product_categories = [
        "Grocery",
        "Clothing",
        "Electronics",
        "Home",
        "Beauty",
        "Sports",
    ]

    customer_regions = [
        "North",
        "South",
        "East",
        "West",
        "Central",
    ]

    order_ids = np.arange(1, n_orders + 1)
    sampled_customers = rng.choice(customer_ids, size=n_orders, replace=True)

    today = datetime.today().date()
    max_days_back = 18 * 30
    days_back = rng.integers(low=0, high=max_days_back, size=n_orders)
    order_dates = [today - timedelta(days=int(d)) for d in days_back]

    category_probs = np.array([0.25, 0.2, 0.15, 0.15, 0.15, 0.1])
    product_category = rng.choice(
        product_categories, size=n_orders, p=category_probs
    )

    region_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    customer_region = rng.choice(
        customer_regions, size=n_orders, p=region_probs
    )

    ages = rng.normal(loc=35, scale=10, size=n_customers).clip(18, 75)
    customer_age_lookup = dict(zip(customer_ids, ages.astype(int)))
    customer_age = np.array(
        [customer_age_lookup[c_id] for c_id in sampled_customers]
    )

    base_values = rng.lognormal(mean=3.0, sigma=0.6, size=n_orders)

    category_multipliers = {
        "Grocery": 0.6,
        "Clothing": 1.0,
        "Electronics": 2.0,
        "Home": 1.2,
        "Beauty": 0.9,
        "Sports": 1.1,
    }
    multipliers = np.array(
        [category_multipliers[cat] for cat in product_category]
    )
    order_values = base_values * multipliers

    order_values = np.round(order_values, 2)

    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": sampled_customers,
            "order_date": order_dates,
            "product_category": product_category,
            "order_value": order_values,
            "customer_age": customer_age,
            "customer_region": customer_region,
        }
    )

    n_missing_age = max(1, int(0.01 * n_orders))
    n_missing_region = max(1, int(0.01 * n_orders))

    missing_age_idx = rng.choice(df.index, size=n_missing_age, replace=False)
    missing_region_idx = rng.choice(
        df.index, size=n_missing_region, replace=False
    )

    df.loc[missing_age_idx, "customer_age"] = np.nan
    df.loc[missing_region_idx, "customer_region"] = np.nan

    n_duplicates = max(1, int(0.005 * n_orders))
    dup_idx = rng.choice(df.index, size=n_duplicates, replace=False)
    duplicates = df.loc[dup_idx].copy()
    df_with_dupes = pd.concat([df, duplicates], ignore_index=True)

    return df_with_dupes


def main() -> None:
    df = generate_retail_dataset()
    output_path = "./data/raw_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated dataset with shape {df.shape} at {output_path}")


if __name__ == "__main__":
    main()

