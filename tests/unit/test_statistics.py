
import pandas as pd
import pytest

from ml_platform.eda.statistics import StatisticalAnalysisService


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "age": [20, 21, 22, 23],
        "score": [80, 90, 70, 100],
        "gender": ["Male", "Female", "Female", "Male"],
        "city": ["Ajmer", "Jaipur", "Ajmer", "Delhi"]
    })


@pytest.fixture
def service():
    return StatisticalAnalysisService()


def test_numerical_summary(service, sample_dataframe):
    result = service.get_numerical_summary(sample_dataframe)

    assert len(result) == 2
    assert result[0]["Column"] == "age"
    assert result[0]["Mean"] == 21.5
    assert result[0]["Median"] == 21.5
    assert result[0]["Min"] == 20
    assert result[0]["Max"] == 23


def test_categorical_summary(service, sample_dataframe):
    result = service.get_categorical_summary(sample_dataframe)

    assert len(result) == 2
    assert result[0]["Column"] == "gender"
    assert result[0]["Unique"] == 2
    assert result[1]["Column"] == "city"
    assert result[1]["Unique"] == 3


def test_correlation_matrix(service, sample_dataframe):
    result = service.get_correlation_matrix(sample_dataframe)

    assert "age" in result.columns
    assert "score" in result.columns
    assert "gender" not in result.columns
    assert result.shape == (2, 2)


def test_numerical_summary_ignores_missing_values(service):
    df = pd.DataFrame({
        "score": [10, 20, None, 30]
    })

    result = service.get_numerical_summary(df)

    assert result[0]["Mean"] == 20
    assert result[0]["Min"] == 10
    assert result[0]["Max"] == 30
