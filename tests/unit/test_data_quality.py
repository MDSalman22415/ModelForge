
import pandas as pd
import pytest

from src.ml_platform.validation.data_quality import DataQualityService


@pytest.fixture
def service():
    return DataQualityService()


# 1. Missing values
def test_analyze_missing_values(service):
    df = pd.DataFrame({
        "age": [20, None, 22, 23],
        "city": ["Delhi", "Jaipur", None, "Kota"],
    })

    result = service.analyze_missing_values(df)

    assert any(
        item["column_name"] == "age"
        and item["missing"] == 1
        for item in result
    )
    assert any(
        item["column_name"] == "city"
        and item["missing"] == 1
        for item in result
    )


# 2. Duplicate rows
def test_analyze_duplicate_rows(service):
    df = pd.DataFrame({
        "student_id": [101, 102, 102, 103],
        "score": [80, 90, 90, 75],
    })
    original = df.copy()

    result = service.analyze_duplicate_rows(df)

    assert result is not None
    pd.testing.assert_frame_equal(df, original)


# 3. Constant columns
def test_constant_columns(service):
    df = pd.DataFrame({
        "city": ["Delhi"] * 4,
        "student_id": [101, 102, 103, 104],
        "status": ["Pass", "Fail", "Pass", "Fail"],
        "empty": [None] * 4,
    })

    result = service.constant_columns(df)

    assert "city" in result["constant_columns"]
    assert "student_id" not in result["constant_columns"]
    assert "status" not in result["constant_columns"]
    assert "empty" in result["all_null_cols"]


# 4. Nearly constant columns
def test_analyze_nearly_constant_columns(service):
    df = pd.DataFrame({
        "status": ["Active"] * 19 + ["Inactive"],
        "city": ["Delhi", "Jaipur"] * 10,
    })

    result = service.analyze_nearly_constant_columns(df)

    assert "status" in result
    assert result["status"]["dominate_value"] == "Active"
    assert result["status"]["frequecy_ratio"] == 0.95
    assert "city" not in result


# 5. Column types
def test_identify_column_types(service):
    df = pd.DataFrame({
        "age": [20, 25, 30, 35],
        "city": ["Delhi", "Jaipur", "Delhi", "Jaipur"],
        "passed": [True, False, True, False],
        "description": ["alpha", "bravo", "charlie", "delta"],
    })

    result = service.identify_column_types(df)

    assert "age" in result["numeric"]
    assert "city" in result["categorical"]
    assert "passed" in result["boolean"]
    assert "description" in result["categorical"]


# 6. Date-like columns
def test_detect_date_like_columns(service):
    df = pd.DataFrame({
        "transaction_date": [
            "2026-01-10",
            "2026-02-15",
            "2026-03-20",
            "2026-04-25",
        ],
        "city": ["Delhi", "Jaipur", "Mumbai", "Kota"],
        "student_id": [101, 102, 103, 104],
    })

    with pytest.warns(UserWarning, match="potential date"):
        result = service.detect_date_like_columns(df)

    assert "transaction_date" in result
    assert "city" not in result
    assert "student_id" not in result


# 7. Target validation
def test_validate_target_column_classification(service):
    df = pd.DataFrame({
        "passed": ["Yes", "No", "Yes", "No"],
    })

    result = service.validate_target_column(df, "passed")

    assert result["target_column"] == "passed"
    assert result["missing_values"] == 0
    assert result["task_type"] == "classification"
    assert result["class_counts"] == {"Yes": 2, "No": 2}


def test_validate_target_column_regression(service):
    df = pd.DataFrame({
        "score": [50, 60, 75, 90],
    })

    result = service.validate_target_column(
        df, "score", task_type="regression"
    )

    assert result["target_column"] == "score"
    assert result["task_type"] == "regression"
    assert result["missing_values"] == 0


# 8. Target imbalance
def test_detect_imbalanced_target(service):
    df = pd.DataFrame({
        "result": ["Pass"] * 90 + ["Fail"] * 10,
    })

    result = service.detect_imbalanced_target(df, "result")

    assert result["class_counts"] == {"Pass": 90, "Fail": 10}
    assert result["minority_class"] == "Fail"
    assert result["minority_ratio"] == pytest.approx(0.1)
    assert not bool(result["is_imbalanced"])


def test_detect_imbalanced_target_balanced(service):
    df = pd.DataFrame({
        "result": ["Pass"] * 50 + ["Fail"] * 50,
    })

    result = service.detect_imbalanced_target(df, "result")

    assert not bool(result["is_imbalanced"])
    assert result["minority_ratio"] == pytest.approx(0.5)


def test_detect_imbalanced_target_regression(service):
    df = pd.DataFrame({
        "score": list(range(25)),
    })

    result = service.detect_imbalanced_target(df, "score")

    assert result["is_imbalanced"] is False
    assert result["reason"] == "Regression target detected"


# 9. Identifier and high-cardinality columns
def test_detect_identifier_and_high_cardinality_columns(service):
    df = pd.DataFrame({
        "student_id": list(range(1001, 1021)),
        "description": [
            f"text_{i}" for i in range(18)
        ] + ["text_0", "text_1"],
        "category": ["A", "B"] * 10,
    })

    with pytest.warns(UserWarning):
        result = service.detect_identifier_and_high_cardinality_columns(df)

    assert "student_id" in result["potential_ids"]
    assert "description" in result["high_cardinality"]
    assert "category" not in result["high_cardinality"]


# 10. Complete dataset validation
def test_validate_dataset_valid(service):
    df = pd.DataFrame({
        "age": [20, 21, 22, 23],
        "passed": ["Yes", "No", "Yes", "No"],
    })

    result = service.validate_dataset(df, target_column="passed")

    assert result["is_valid"] is True
    assert result["details"]["shape"] == (4, 2)
    assert "target_validation" in result["details"]


def test_validate_dataset_empty(service):
    df = pd.DataFrame()

    result = service.validate_dataset(df)

    assert result["is_valid"] is False
    assert len(result["errors"]) > 0


def test_validate_dataset_invalid_target(service):
    df = pd.DataFrame({
        "age": [20, 21, 22, 23],
        "passed": ["Yes", "No", "Yes", "No"],
    })

    result = service.validate_dataset(
        df, target_column="missing_column"
    )

    assert result["is_valid"] is False
    assert any(
        "Target column validation failed" in error
        for error in result["errors"]
    )
