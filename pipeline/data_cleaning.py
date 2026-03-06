import pandas as pd
from pathlib import Path


def clean_retail_data(
    input_path: str = "./data/raw_dataset.csv",
    output_path: str = "./output/cleaned_data.csv",
) -> pd.DataFrame:
    """
    Load raw retail data, apply basic cleaning, and write a cleaned dataset.
    """
    df = pd.read_csv(input_path)

    # Ensure types
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_value"] = pd.to_numeric(df["order_value"], errors="coerce")

    # Standardize categorical text
    df["product_category"] = (
        df["product_category"].astype(str).str.strip().str.title()
    )
    df["customer_region"] = (
        df["customer_region"].astype(str).str.strip().str.title()
    )

    # Handle missing values
    if "customer_age" in df.columns:
        median_age = df["customer_age"].median()
        df["customer_age"] = df["customer_age"].fillna(median_age)

    if "customer_region" in df.columns:
        # Treat literal "nan" from astype(str) as missing
        df["customer_region"] = df["customer_region"].replace(
            {"Nan": pd.NA, "Na": pd.NA, "None": pd.NA}
        )
        mode_region = df["customer_region"].mode(dropna=True)
        if not mode_region.empty:
            df["customer_region"] = df["customer_region"].fillna(
                mode_region.iloc[0]
            )

    # Drop rows with invalid dates or order values
    df = df.dropna(subset=["order_date", "order_value"])

    # Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Sort for deterministic downstream behavior
    df = df.sort_values(["order_date", "order_id"]).reset_index(drop=True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return df


def main() -> None:
    df = clean_retail_data()
    print(f"Cleaned dataset written with shape {df.shape}")


if __name__ == "__main__":
    main()

