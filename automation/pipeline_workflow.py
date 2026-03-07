import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def run_step(step_name: str, script_relative_path: str) -> None:
    log(f"START: {step_name}")
    script_path = PROJECT_ROOT / script_relative_path

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError as exc:
        log(f"ERROR in {step_name}: {exc}")
        raise

    log(f"END: {step_name}")


def run_full_pipeline() -> None:
    run_step("Generate synthetic retail dataset", "pipeline/generate_dataset.py")
    run_step("Clean data", "pipeline/data_cleaning.py")
    run_step("Compute KPI metrics", "pipeline/kpi_generation.py")
    run_step("Customer segmentation", "pipeline/customer_segmentation.py")
    run_step("AI insight generation", "pipeline/ai_insight_generation.py")


def main() -> None:
    log("AI-BI pipeline workflow started")
    try:
        run_full_pipeline()
    except Exception:
        log("Pipeline failed")
        raise
    else:
        log("AI-BI pipeline workflow completed successfully")


if __name__ == "__main__":
    main()

