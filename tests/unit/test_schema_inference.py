
import pandas as pd

from ml_platform.ingestion.schema_inference import SchemaInterfaceService


def test_infer_schema():
    df = pd.DataFrame({
        "age": [20, 25, None],
        "salary": [25000, 30000, 35000],
    })

    service = SchemaInterfaceService()
    result = service.infer_schema(df)

    assert len(result) == 2
    assert result[0]["name"] == "age"
    assert result[0]["null_count"] == 1
    assert result[0]["unique_count"] == 2
    assert result[1]["name"] == "salary"
    assert result[1]["null_count"] == 0
    assert result[1]["unique_count"] == 3
