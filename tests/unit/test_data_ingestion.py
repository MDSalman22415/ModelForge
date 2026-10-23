import pandas as pd
from ml_platform.ingestion.data_ingestion import DataIngestionService


def test_load_csv(tmp_path):
    # Create a small CSV file for testing
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("name,age\nSalman,20\nRohan,21\n")

    service = DataIngestionService()
    df = service.load_csv(csv_file)

    # Verify the function works correctly
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["name", "age"]


