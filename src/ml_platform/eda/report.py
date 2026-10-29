
import os
import json
import sys
import pandas as pd

from ml_platform.exception.excepton import ModelForge
from ml_platform.logging.logger import logger

from ml_platform.eda.profiling import DatasetProfiler
from ml_platform.eda.statistics import StatisticalAnalysisService
from ml_platform.eda.visualizations import DataVisualizationService


class EDAReportService:
    def collect_profiling_results(self, df:pd.DataFrame):
        try:
            profiler = DatasetProfiler()
            profiling_results = {
                "get_data_overview" : profiler.get_dataset_overview(df),
                "get_columns_summary" : profiler.get_column_summary(df),
                "get_memory_usage" : profiler.get_memory_usage(df),
                "get_duplicate_summary" : profiler.get_duplicate_summary(df),
                "get_missing_values" : profiler.get_missing_summary(df)
            }
            
            return profiling_results
        except Exception as e:
            raise ModelForge(e, sys)
    def collect_statistical_results(self, df:pd.DataFrame):
        try:
            logger.info("COLLECCT STATISTICAL RESULT START")
            statical = StatisticalAnalysisService()
            
            result = {
                "get_numerical_summary" : statical.get_numerical_summary(df),
                "get_categorical_summary" : statical.get_categorical_summary(df),
                "get_correlation_matrix"  : statical.get_correlation_matrix(df)
            }
            
            return result
        except Exception as e:
            raise ModelForge(e, sys)
    
    def collect_visualization_results(self, df, output_dir):
        try:
            visualizer = DataVisualizationService()

            visualization_results = visualizer.generate_all_visualizations(
                df, output_dir
            )

            return visualization_results
        except Exception as e:
            logger.error(f"Error collecting visualization results: {e}")
            raise ModelForge(e, sys)
    
    def collect_eda_results(self, df, output_dir):
        try:
            profiling_results = self.collect_profiling_results(df)

            statistical_results = self.collect_statistical_results(df)

            visualization_results = self.collect_visualization_results(
                df, output_dir
            )

            report = {
                "profiling": profiling_results,
                "statistical_analysis": statistical_results,
                "visualizations": visualization_results
            }

            return report

        except Exception as e:
            logger.error(f"Error collecting EDA results: {e}")
            raise ModelForge(e, sys)

    
    def save_report(self, report, output_path):
        try:
            directory = os.path.dirname(output_path)

            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(report, file, indent=4, default=str)

            logger.info(f"EDA report saved: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error saving EDA report: {e}")
            raise ModelForge(e, sys)
