from ml_platform.exception.excepton import ModelForge
from ml_platform.logging.logger import logger
import pandas as pd
import sys

class DatasetProfiler:
    def get_dataset_overview(self, df: pd.DataFrame):
        try:
            logger.info("NOW WE ARE GOING TO START PHASE TWO OF THIS PROJECT")
            dataset_Overview = {
                "Total_rows" : len(df),
                "Total_columns" : len(df.columns),
                "Dataset_Shape" : (df.shape)
            }
            return dataset_Overview
        except Exception as e:
            raise ModelForge(e, sys)
    
    def get_column_summary(self, df: pd.DataFrame):
        try:
            logger.info("Now You can see columns summary")
            df_copy = df.copy()
            columns_summary = []
            for col in df_copy.columns:
                information = {
                    "Columns_name" : (col),
                    "Data_Types"   : (df_copy[col].dtypes),
                    "Unique_Values": (df_copy[col].nunique())
                }
                
                columns_summary.append(information)
            return columns_summary
        except Exception as e:
            raise ModelForge(e, sys)
    
    def get_memory_usage(self, df: pd.DataFrame):
        try:
            logger.info("NOW WE ARE CHECKING MEMORY USAGE")
            
            memory_bytes = df.memory_usage(deep=True).sum()
            memory_kb = round(memory_bytes / 1024, 2)
            memory_mb = round(memory_bytes / (1024 ** 2), 2)

            information = {
                "Total_memory_bytes": memory_bytes,
                "Total_memory_kb": memory_kb,
                "Total_memory_mb": memory_mb
            }

            logger.info("MEMORY USAGE CHECKED SUCCESSFULLY")

            return information

        except Exception as e:
            raise ModelForge(e, sys)
        
    def get_duplicate_summary(self, df: pd.DataFrame):
        try:
            logger.info("NOW WE CHECKING DUPLICATE SUMMARY")
            duplicate_summary = []
            for col in df.columns:
                total_rows = len(df)
                duplicate_values = df[col].duplicated().sum()
                if total_rows > 0:
                    Duplicate_Percentage = (duplicate_values / total_rows)  * 100
                else:
                    Duplicate_Percentage = 0
                information = {
                    "Columns_Name": (col),
                    "Duplicate_values" : duplicate_values,
                    "Duplicate_percentage" : Duplicate_Percentage
                }
                duplicate_summary.append(information)
            return duplicate_summary
        except Exception as e:
            raise ModelForge(e, sys)
    def get_missing_summary(self, df: pd.DataFrame):
        try:
            logger.info("NOW WE FIND MISSING VALUES ")
            missing_summary = []
            for col in df.columns:
                total_rows = len(df)
                missing_values = df[col].isnull().sum()
                if total_rows > 0:
                    Missing_Percentage = (missing_values / total_rows) * 100
                else:
                    Missing_Percentage = 0
                information = {
                    "Column_Name" : col,
                    "Missing_Values" :missing_values,
                    "Missing_Percentage" : Missing_Percentage
                }
                missing_summary.append(information)
            return missing_summary
        except Exception as e:
            raise ModelForge(e, sys)