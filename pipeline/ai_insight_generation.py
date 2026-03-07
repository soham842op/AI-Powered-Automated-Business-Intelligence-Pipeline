import os
from pathlib import Path

import pandas as pd

try:
    import google.generativeai as genai
except ImportError:
    genai = None


def load_env_from_root(env_filename: str = ".env") -> None:
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / env_filename
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_summary_data(
    kpi_path: str = "./output/kpi_metrics.csv",
    segments_path: str = "./output/customer_segments.csv",
) -> str:
    """
    Load KPIs and segments, then build a concise text summary for the LLM.
    """
    kpi_df = pd.read_csv(kpi_path)
    seg_df = pd.read_csv(segments_path)

    overall = kpi_df[kpi_df["metric_type"] == "overall"]
    monthly = kpi_df[kpi_df["metric_type"] == "monthly_revenue"].sort_values(
        "group_key"
    )
    category = kpi_df[kpi_df["metric_type"] == "category_revenue"].sort_values(
        "metric_value", ascending=False
    )

    lines = ["=== RETAIL BUSINESS DATA SUMMARY ===\n"]

    for _, row in overall.iterrows():
        val = row["metric_value"]
        if "revenue" in row["metric_name"] or "value" in row["metric_name"]:
            lines.append(f"- {row['metric_name']}: {val:,.2f}")
        else:
            lines.append(f"- {row['metric_name']}: {int(val)}")

    lines.append("\n--- Monthly Revenue (last 6 months) ---")
    for _, row in monthly.tail(6).iterrows():
        lines.append(f"  {row['group_key']}: {row['metric_value']:,.2f}")

    lines.append("\n--- Top Product Categories by Revenue ---")
    for _, row in category.head(5).iterrows():
        lines.append(f"  {row['group_key']}: {row['metric_value']:,.2f}")

    seg_summary = (
        seg_df.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_spend=("total_spending", "mean"),
            avg_orders=("order_count", "mean"),
        )
        .round(2)
    )
    lines.append("\n--- Customer Segments (K-Means) ---")
    for seg_id, row in seg_summary.iterrows():
        lines.append(
            f"  Segment {seg_id}: {int(row['customers'])} customers, "
            f"avg spend {row['avg_spend']:.2f}, avg orders {row['avg_orders']:.2f}"
        )

    return "\n".join(lines)


def generate_insights_with_gemini(
    summary_text: str,
    api_key: str | None = None,
) -> list[dict]:
    """
    Call Gemini API to generate business insights from the summary.
    """
    if genai is None:
        raise ImportError("Install google-generativeai: pip install google-generativeai")

    load_env_from_root()
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "Set GEMINI_API_KEY in your environment or pass api_key to this function."
        )

    genai.configure(api_key=key)

    prompt = f"""You are a business analyst. Based on this retail data summary, provide 4-6 short, actionable insights in bullet points. Cover:
1. Sales trends (monthly pattern)
2. Top performing categories
3. Customer segment highlights (who to focus on)
4. Any unusual patterns or risks
5. One or two brief recommendations

Keep each insight to 1-2 sentences. Be specific with numbers when relevant.

DATA:
{summary_text}

OUTPUT FORMAT: Return only the bullet points, one per line, starting with a dash (-). No extra preamble."""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    if not response or not response.text:
        return [{"insight_type": "error", "insight_text": "No response from Gemini"}]

    raw_text = response.text.strip()
    bullets = [line.strip() for line in raw_text.split("\n") if line.strip() and line.strip().startswith("-")]

    records = []
    for i, bullet in enumerate(bullets):
        text = bullet.lstrip("- ").strip()
        if text:
            records.append(
                {
                    "insight_id": i + 1,
                    "insight_text": text,
                    "insight_type": "ai_generated",
                }
            )

    return records if records else [{"insight_id": 1, "insight_text": raw_text, "insight_type": "ai_generated"}]


def run_ai_insights(
    kpi_path: str = "./output/kpi_metrics.csv",
    segments_path: str = "./output/customer_segments.csv",
    output_path: str = "./output/ai_insights.csv",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Load data, call Gemini, and save insights to CSV.
    """
    summary = load_summary_data(kpi_path=kpi_path, segments_path=segments_path)
    records = generate_insights_with_gemini(summary, api_key=api_key)

    insights_df = pd.DataFrame.from_records(records)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    insights_df.to_csv(out_file, index=False)

    return insights_df


def main() -> None:
    insights_df = run_ai_insights()
    print(f"AI insights written with shape {insights_df.shape} to output/ai_insights.csv")


if __name__ == "__main__":
    main()
