
import sys
from pathlib import Path

import pandas as pd

from ml_platform.logging.logger import logger
from ml_platform.exception.excepton import ModelForge


class SchemaInterfaceService:

    def infer_schema(self, file_path):
        try:
            path = Path(file_path)

            logger.info("Starting schema inference for file: %s", path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Dataset file not found: {path}"
                )

            if not path.is_file():
                raise ValueError(
                    f"The provided path is not a file: {path}"
                )

            if path.suffix.lower() != ".csv":
                raise ValueError(
                    "Unsupported file format. Please provide a CSV file."
                )

            try:
                df = pd.read_csv(path)

            except pd.errors.EmptyDataError as exc:
                raise ValueError(
                    "The CSV file is empty or contains no readable data."
                ) from exc

            except pd.errors.ParserError as exc:
                raise ValueError(
                    "The CSV file could not be parsed."
                ) from exc

            if df.empty:
                raise ValueError(
                    "The CSV contains no data rows."
                )

            summary_list = []

            for col in df.columns:
                user_information = {
                    "name": str(col),
                    "data_types": str(df[col].dtype),
                    "null_count": int(df[col].isnull().sum()),
                    "unique_count": int(df[col].nunique()),
                }

                summary_list.append(user_information)

            logger.info(
                "Schema inference completed successfully for file: %s",
                path,
            )

            return summary_list

        except Exception as exc:
            logger.exception(
                "Schema inference failed for file: %s",
                file_path,
            )

            if isinstance(exc, ModelForge):
                raise

            raise ModelForge(exc, sys) from exc
