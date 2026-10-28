
import sys
from pathlib import Path

import pandas as pd

from ml_platform.logging.logger import logger
from ml_platform.exception.excepton import ModelForge


class DataIngestionService:

    def load_csv(self, file_path):
        try:
            path = Path(file_path)

            # 1. Validate file existence
            if not path.exists():
                raise FileNotFoundError(
                    f"Dataset file not found: {path}"
                )

            # 2. Validate file format
            if path.suffix.lower() != ".csv":
                raise ValueError(
                    "Unsupported file format. Please provide a CSV file."
                )

            # 3. Validate that the path points to a file
            if not path.is_file():
                raise ValueError(
                    f"The provided path is not a file: {path}"
                )

            logger.info("Loading CSV dataset: %s", path)

            # 4. Load CSV and handle common parsing errors
            try:
                dataframe = pd.read_csv(path)

            except pd.errors.EmptyDataError as exc:
                raise ValueError(
                    "The CSV file is empty or contains no readable data."
                ) from exc

            except pd.errors.ParserError as exc:
                raise ValueError(
                    "The CSV file could not be parsed. Check its structure."
                ) from exc

            # 5. Validate loaded dataset
            if dataframe.empty:
                raise ValueError(
                    "The CSV contains column headers but no data rows."
                )

            logger.info(
                "CSV loaded successfully: %s rows, %s columns",
                dataframe.shape[0],
                dataframe.shape[1],
            )

            return dataframe

        except Exception as e:
            logger.exception(
                "Failed to load CSV dataset: %s",
                file_path,
            )
            raise ModelForge(e, sys) from e
