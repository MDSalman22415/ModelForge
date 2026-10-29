
import pandas as pd
import pytest
import os

from ml_platform.eda.visualizations import DataVisualizationService


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "age": [20, 21, 22, 23, 24],
        "score": [60, 70, 80, 90, 100],
        "gender": ["Male", "Female", "Female", "Male", "Female"],
        "city": ["Ajmer", "Jaipur", "Ajmer", "Delhi", "Jaipur"]
    })


@pytest.fixture
def service():
    return DataVisualizationService()


def test_generate_histograms(service, sample_dataframe, tmp_path):
    output_dir = str(tmp_path / "histograms")

    result = service.generate_histograms(
        sample_dataframe, output_dir
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(os.path.exists(path) for path in result)


def test_generate_boxplots(service, sample_dataframe, tmp_path):
    output_dir = str(tmp_path / "boxplots")

    result = service.generate_boxplots(
        sample_dataframe, output_dir
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(os.path.exists(path) for path in result)


def test_generate_correlation_heatmap(service, sample_dataframe):
    result = service.generate_correlation_heatmap(sample_dataframe)

    assert isinstance(result, str)
    assert os.path.exists(result)


def test_generate_categorical_plots(service, sample_dataframe):
    result = service.generate_categorical_plots(sample_dataframe)

    assert isinstance(result, dict)
    assert "gender" in result
    assert "city" in result
    assert all(os.path.exists(path) for path in result.values())


def test_generate_scatter_plots(service, sample_dataframe):
    result = service.generate_scatter_plots(
        sample_dataframe,
        target_column="score"
    )

    assert isinstance(result, dict)
    assert len(result) == 1
    assert "age" in result

def test_generate_all_visualizations(service, sample_dataframe):
    result = service.generate_all_visualizations(
        sample_dataframe,
        target_column="score"
    )

    expected_keys = {
        "histograms",
        "boxplots",
        "correlation_heatmap",
        "categorical_plots",
        "scatter_plots"
    }

    assert isinstance(result, dict)
    assert expected_keys.issubset(result.keys())

    for name in expected_keys:
        assert result[name]["status"] == "success", (
            f"{name} failed: {result[name]['error']}"
        )


def test_visualization_does_not_modify_dataframe(
    service, sample_dataframe
):
    original = sample_dataframe.copy(deep=True)

    service.generate_all_visualizations(
        sample_dataframe,
        target_column="score"
    )

    pd.testing.assert_frame_equal(
        sample_dataframe, original
    )

def test_generate_histograms(service, sample_dataframe, tmp_path):
    result = service.generate_histograms(
        sample_dataframe,
        str(tmp_path / "histograms")
    )

    assert isinstance(result, list)
    assert len(result) == 2


def test_generate_boxplots(service, sample_dataframe, tmp_path):
    result = service.generate_boxplots(
        sample_dataframe,
        str(tmp_path / "boxplots")
    )

    assert isinstance(result, list)
    assert len(result) == 2
    
def test_generate_correlation_heatmap(service, sample_dataframe):
    result = service.generate_correlation_heatmap(sample_dataframe)

    assert isinstance(result, str)
    assert os.path.exists(result)

def test_generate_histograms(service, sample_dataframe, tmp_path):
    result = service.generate_histograms(
        sample_dataframe,
        str(tmp_path / "histograms")
    )

    assert isinstance(result, list)
    assert len(result) == 2


def test_generate_boxplots(service, sample_dataframe, tmp_path):
    result = service.generate_boxplots(
        sample_dataframe,
        str(tmp_path / "boxplots")
    )

    assert isinstance(result, list)
    assert len(result) == 2

def test_generate_correlation_heatmap(service, sample_dataframe):
    result = service.generate_correlation_heatmap(sample_dataframe)

    assert isinstance(result, str)
    assert os.path.exists(result)