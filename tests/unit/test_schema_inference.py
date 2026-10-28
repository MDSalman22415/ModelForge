
import pandas as pd
import pytest

from ml_platform.ingestion.schema_inference import SchemaInterfaceService
from ml_platform.exception.excepton import ModelForge


@pytest.fixture
def schema_service():
    return SchemaInterfaceService()


def test_infer_schema_success(schema_service, tmp_path):
    file_path = tmp_path / "sample.csv"

    df = pd.DataFrame({
        "age": [20, 25, 30],
        "city": ["Delhi", "Jaipur", "Delhi"],
    })
    df.to_csv(file_path, index=False)

    result = schema_service.infer_schema(file_path)

    assert len(result) == 2

    assert result[0] == {
        "name": "age",
        "data_types": "int64",
        "null_count": 0,
        "unique_count": 3,
    }

    assert result[1] == {
        "name": "city",
        "data_types": "str",
        "null_count": 0,
        "unique_count": 2,
    }


def test_infer_schema_file_not_found(schema_service, tmp_path):
    file_path = tmp_path / "missing.csv"

    with pytest.raises(ModelForge) as exc_info:
        schema_service.infer_schema(file_path)

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_infer_schema_path_is_directory(schema_service, tmp_path):
    directory_path = tmp_path / "dataset.csv"
    directory_path.mkdir()

    with pytest.raises(ModelForge) as exc_info:
        schema_service.infer_schema(directory_path)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_infer_schema_unsupported_file_format(schema_service, tmp_path):
    file_path = tmp_path / "dataset.txt"
    file_path.write_text("age,city\n20,Delhi\n", encoding="utf-8")

    with pytest.raises(ModelForge) as exc_info:
        schema_service.infer_schema(file_path)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_infer_schema_empty_csv(schema_service, tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ModelForge) as exc_info:
        schema_service.infer_schema(file_path)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_infer_schema_headers_without_rows(schema_service, tmp_path):
    file_path = tmp_path / "headers_only.csv"
    file_path.write_text("age,city\n", encoding="utf-8")

    with pytest.raises(ModelForge) as exc_info:
        schema_service.infer_schema(file_path)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_infer_schema_malformed_csv(schema_service, tmp_path):
    file_path = tmp_path / "malformed.csv"
    file_path.write_text(
        'age,city\n20,"Delhi\n25,Jaipur\n',
        encoding="utf-8",
    )

    with pytest.raises(ModelForge) as exc_info:
        schema_service.infer_schema(file_path)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_infer_schema_counts_missing_values(schema_service, tmp_path):
    file_path = tmp_path / "missing_values.csv"

    df = pd.DataFrame({
        "age": [20, None, 30],
        "city": ["Delhi", None, "Delhi"],
    })
    df.to_csv(file_path, index=False)

    result = schema_service.infer_schema(file_path)

    assert result[0]["name"] == "age"
    assert result[0]["null_count"] == 1
    assert result[0]["unique_count"] == 2

    assert result[1]["name"] == "city"
    assert result[1]["null_count"] == 1
    assert result[1]["unique_count"] == 1
