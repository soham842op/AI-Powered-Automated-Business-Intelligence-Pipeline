## AI-Powered Automated Business Intelligence Pipeline

End-to-end learning project that simulates a retail business, builds an analytics pipeline, runs ML-based customer segmentation, generates AI insights with Gemini, and exposes everything to a Power BI dashboard.

### Project structure

- **data/**
  - `raw_dataset.csv` – synthetic raw retail orders (generated).
- **output/**
  - `cleaned_data.csv` – cleaned orders used for analysis.
  - `kpi_metrics.csv` – precomputed KPIs (optional for BI).
  - `customer_segments.csv` – per-customer features + K-Means segment.
  - `ai_insights.csv` – LLM-generated narrative insights.
- **pipeline/**
  - `generate_dataset.py` – create synthetic retail dataset.
  - `data_cleaning.py` – handle missing data, types, duplicates, categories.
  - `kpi_generation.py` – compute revenue KPIs and export metrics table.
  - `customer_segmentation.py` – build features and run K-Means clustering.
  - `ai_insight_generation.py` – summarize data and call Gemini for insights.
- **automation/**
  - `pipeline_workflow.py` – orchestrates the full pipeline end to end.
- **dashboard/**
  - `powerbi_dashboard_guide.md` – dashboard pages and visuals.

### Setup

1. **Create and activate a Python environment** (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your-gemini-api-key-here
```

Do not commit `.env` to version control.

### Running the pipeline

From the project root:

```bash
python automation/pipeline_workflow.py
```

This runs, in order:

1. `generate_dataset.py` → `data/raw_dataset.csv`
2. `data_cleaning.py` → `output/cleaned_data.csv`
3. `kpi_generation.py` → `output/kpi_metrics.csv`
4. `customer_segmentation.py` → `output/customer_segments.csv`
5. `ai_insight_generation.py` → `output/ai_insights.csv`

### Power BI dashboard

- Open **Power BI Desktop**.
- Use **Get data → Text/CSV** to load:
  - `output/cleaned_data.csv`
  - `output/customer_segments.csv`
  - `output/kpi_metrics.csv`
  - `output/ai_insights.csv`
- In **Model view**, relate:
  - `cleaned_data[customer_id]` → `customer_segments[customer_id]` (Many-to-One).
- Build report pages following `dashboard/powerbi_dashboard_guide.md`.

### Learning objectives

- Data pipeline architecture and orchestration.
- Data cleaning and feature engineering for analytics.
- KPI design for retail business monitoring.
- Unsupervised customer segmentation with K-Means.
- LLM-powered insight generation using Gemini.
- Building an executive-ready Power BI dashboard.

