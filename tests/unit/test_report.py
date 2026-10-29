
from unittest.mock import Mock

import pytest

from ml_platform.eda.report import EDAReportService


@pytest.fixture
def sample_df():
    import pandas as pd

    return pd.DataFrame({
        "age": [20, 21, 22, 23, 24],
        "score": [75, 80, 85, 90, 70],
        "gender": ["M", "F", "M", "F", "M"],
    })


@pytest.fixture
def report_service():
    return EDAReportService()


def test_collect_eda_results_structure(
    report_service, sample_df, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        report_service,
        "collect_profiling_results",
        Mock(return_value={"rows": 5}),
    )

    monkeypatch.setattr(
        report_service,
        "collect_statistical_results",
        Mock(return_value={"mean_score": 80}),
    )

    monkeypatch.setattr(
        report_service,
        "collect_visualization_results",
        Mock(return_value={
            "histogram": "artifacts/histogram.png"
        }),
    )

    result = report_service.collect_eda_results(
        sample_df, str(tmp_path)
    )

    assert result == {
        "profiling": {"rows": 5},
        "statistical_analysis": {"mean_score": 80},
        "visualizations": {
            "histogram": "artifacts/histogram.png"
        },
    }


def test_save_report_creates_json_file(
    report_service, tmp_path
):
    report_data = {
        "profiling": {"rows": 5},
        "statistical_analysis": {"mean_score": 80},
        "visualizations": {},
    }

    output_path = tmp_path / "eda_report.json"

    report_service.save_report(
        report_data, str(output_path)
    )

    assert output_path.exists()

    import json

    with open(output_path, "r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report == report_data


def test_save_report_handles_invalid_path(report_service):
    report_data = {
        "profiling": {},
        "statistical_analysis": {},
        "visualizations": {},
    }

    with pytest.raises(Exception):
        report_service.save_report(
            report_data,
            "\0/invalid/eda_report.json",
        )
