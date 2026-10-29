from ml_platform.exception.excepton import ModelForge
from ml_platform.logging.logger import logger
import pandas as pd
import sys
class StatisticalAnalysisService:
    def get_numerical_summary(self, df : pd.DataFrame):
        try:
            numerical_columns = df.select_dtypes(include="number").columns
            numerical_summary = []
            for col in numerical_columns:
                information = {
                    "Column" : col,
                    "Mean" : df[col].mean(),
                    "Median" : df[col].median(),
                    "Std" : df[col].std(),
                    "Min" : df[col].min(),
                    "Max" : df[col].max(),
                }
                numerical_summary.append(information)
            return numerical_summary
        except Exception as e:
            raise ModelForge(e, sys)
    def get_categorical_summary(self, df: pd.DataFrame):
        try:
            categorical_columns = df.select_dtypes(include=['object','str']).columns
            categorical_summary = []
            for col in categorical_columns:
                information = {
                    "Column" : col,
                    "Unique" : df[col].nunique(),
                    "Value_Counts" : df[col].value_counts()
                }
                categorical_summary.append(information)
            return categorical_summary
        except Exception as e:
            raise ModelForge(e, sys)
    def get_correlation_matrix(self, df:pd.DataFrame):
        try:
            correlation = df.corr(numeric_only=True)
            return correlation
        except Exception as e:
            raise ModelForge(e, sys)