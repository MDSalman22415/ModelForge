
import pandas as pd
import pytest

from ml_platform.eda.profiling import DatasetProfiler


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [20, 21, 21, 23],
        "score": [80, 90, None, 70],
        "gender": ["Male", "Female", "Female", "Male"]
    })


def test_get_dataset_overview(sample_df):
    profiler = DatasetProfiler()

    result = profiler.get_dataset_overview(sample_df)

    assert result["Total_rows"] == 4
    assert result["Total_columns"] == 3
    assert result["Dataset_Shape"] == (4, 3)


def test_get_column_summary(sample_df):
    profiler = DatasetProfiler()

    result = profiler.get_column_summary(sample_df)

    assert len(result) == 3
    assert result[0]["Columns_name"] == "age"
    assert result[0]["Unique_Values"] == 3


def test_get_memory_usage(sample_df):
    profiler = DatasetProfiler()

    result = profiler.get_memory_usage(sample_df)

    assert result["Total_memory_bytes"] > 0
    assert result["Total_memory_kb"] > 0
    assert result["Total_memory_mb"] >= 0


def test_get_duplicate_summary(sample_df):
    profiler = DatasetProfiler()

    result = profiler.get_duplicate_summary(sample_df)

    assert len(result) == 3
    assert result[0]["Columns_Name"] == "age"
    assert result[0]["Duplicate_values"] == 1
    assert result[0]["Duplicate_percentage"] == 25.0


def test_get_missing_summary(sample_df):
    profiler = DatasetProfiler()

    result = profiler.get_missing_summary(sample_df)

    assert len(result) == 3
    assert result[1]["Column_Name"] == "score"
    assert result[1]["Missing_Values"] == 1
    assert result[1]["Missing_Percentage"] == 25.0
