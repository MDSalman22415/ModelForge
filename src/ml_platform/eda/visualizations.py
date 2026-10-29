from ml_platform.exception.excepton import ModelForge
from ml_platform.logging.logger import logger
import pandas as pd
import sys
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
class DataVisualizationService:
    def generate_histograms(self, df: pd.DataFrame, output_dir: str):
        try:
            logger.info("STARTING NUMERICAL COLUMN HISTOGRAM GENERATION")

            if df.empty:
                logger.warning("DATAFRAME IS EMPTY")
                return []

            numerical_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

            if not numerical_columns:
                logger.warning("NO NUMERICAL COLUMNS FOUND")
                return []

            os.makedirs(output_dir, exist_ok=True)
            saved_files = []

            for col in numerical_columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(df[col].dropna(), bins=30)
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                fig.tight_layout()

                file_path = os.path.join(
                    output_dir,
                    f"{col}_histogram.png"
                )

                fig.savefig(file_path, dpi=150)
                plt.close(fig)
                saved_files.append(file_path)

                logger.info(f"HISTOGRAM SAVED: {file_path}")

            logger.info("ALL HISTOGRAMS GENERATED SUCCESSFULLY")
            return saved_files

        except Exception as e:
            raise ModelForge(e, sys)

    def generate_boxplots(self, df: pd.DataFrame, output_dir: str):
        try:
            logger.info("NOW WE ARE GENRATING BOXPLOTS")

            if df.empty:
                logger.warning("DATAFRAME IS EMPTY")
                return []

            numerical_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

            if not numerical_columns:
                logger.warning("NUMERICAL COLUMNS NOT FOUND")
                return []

            os.makedirs(output_dir, exist_ok=True)
            saved_files = []

            for col in numerical_columns:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.boxplot(df[col].dropna())
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                fig.tight_layout()

                file_path = os.path.join(
                    output_dir,
                    f"{col}_boxplot.png"
                )

                fig.savefig(file_path, dpi=150)
                plt.close(fig)

                saved_files.append(file_path)
                logger.info(f"BOXPLOT SAVED: {file_path}")

            logger.info("ALL BOXPLOT GENERATED SUCCESSFULLY")
            return saved_files

        except Exception as e:
            raise ModelForge(e, sys)

    def generate_correlation_heatmap(self, df: pd.DataFrame):
        try:
            logger.info("NOW WE ARE GENERATE CORRELATION HEATMAP")

            if df.empty:
                logger.warning("DATAFRAME IS EMPLTY")
                return []

            numerical_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

            if not numerical_columns:
                logger.warning("NUMERICAL NOT FOUND")
                return []

            if len(numerical_columns) < 2:
                logger.warning("AT LEAST TOW NUMERICAL COLUMNS ARE REQUIRED")
                return []

            correlation_matrix = df[numerical_columns].corr()

            fig, ax = plt.subplots(figsize=(12, 6))

            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                ax=ax
            )

            ax.set_title("Correlation Heatmap")
            fig.tight_layout()

            output_path = "artifacts/correlation_heatmap.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            fig.savefig(output_path, dpi=150)
            plt.close(fig)

            logger.info(f"CORRELATION HEATMAP SAVED: {output_path}")
            return output_path

        except Exception as e:
            raise ModelForge(e, sys)

    def generate_categorical_plots(
        self,
        df: pd.DataFrame,
        top_n: int = 10,
    ):
        try:
            logger.info("NOW WE ARE GOING TO MOVE GENERATE CATEGORICAL PLOTS")

            if df.empty:
                logger.warning("DATAFRAME IS EMPTY")
                return []

            categorical_columns = df.select_dtypes(
                include=["object", "category", "string", "bool"]
            ).columns.tolist()

            if not categorical_columns:
                logger.warning("CATEGORICAL COLUMNS IS NOT FOUND")
                return []

            output_dir = "artifacts/categorical_plots"
            os.makedirs(output_dir, exist_ok=True)

            saved_graphs = {}

            for col in categorical_columns:
                categorical_frequency = df[col].value_counts().head(top_n)

                if categorical_frequency.empty:
                    logger.warning(f"NO VALID CATEGORICAL FOUND IN {col}")
                    continue

                fig, ax = plt.subplots(figsize=(10, 6))

                ax.bar(
                    categorical_frequency.index.astype(str),
                    categorical_frequency.values
                )

                ax.set_title(f"Categorical frequency Distribution :{col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequecny")
                ax.tick_params(axis="x", rotation=45)

                fig.tight_layout()

                file_name = f"{col}_categorical_plot.png"
                output_path = os.path.join(output_dir, file_name)

                fig.savefig(output_path, dpi=150)
                plt.close(fig)

                saved_graphs[col] = output_path

                logger.info(f"CATEGORICAL PLOT SAVED: {output_path}")

            logger.info("CATEGORICAL PLOTS GENERATED SUCCESSFULLY")
            return saved_graphs

        except Exception as e:
            raise ModelForge(e, sys)

    def generate_scatter_plots(
        self,
        df: pd.DataFrame,
        target_column=None
    ):
        try:
            logger.info("NOW WE ARE GOING TO GENERATE SCATTER PLOTS")

            if df.empty:
                logger.warning("DATAFRAME IS EMPTY")
                return []

            numerical_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

            if len(numerical_columns) < 2:
                logger.warning("NOT ENOUGH NUMERICAL COLUMNS")
                return []

            if target_column is not None:
                if target_column not in numerical_columns:
                    logger.warning("TARGET COLUMN IS NOT NUMERIC")
                    return {}

                feture_columns = [
                    col for col in numerical_columns
                    if col != target_column
                ]
            else:
                feture_columns = numerical_columns[1:]
                target_column = numerical_columns[0]

            output_dir = "artifacts/scatter_plots"
            os.makedirs(output_dir, exist_ok=True)

            saved_graphs = {}

            for col in feture_columns:
                fig, ax = plt.subplots(figsize=(8, 5))

                ax.scatter(
                    df[col],
                    df[target_column],
                    alpha=0.6
                )

                ax.set_title(f"{col} vs {target_column}")
                ax.set_xlabel(col)
                ax.set_ylabel(target_column)

                fig.tight_layout()

                file_path = os.path.join(
                    output_dir,
                    f"{col}_vs_{target_column}.png"
                )

                fig.savefig(file_path, dpi=150)
                plt.close(fig)

                saved_graphs[col] = file_path

                logger.info(f"SCATTER PLOT SAVED: {file_path}")

            return saved_graphs

        except Exception as e:
            raise ModelForge(e, sys)

    def generate_all_visualizations(
        self,
        df: pd.DataFrame,
        target_column=None
    ):
        logger.info("STARTING ALL VISUALIZATIONS")

        visualization_manifest = {}

        visualization_methods = {
            "histograms": lambda: self.generate_histograms(
                df, "artifacts/histograms"
            ),
            "boxplots": lambda: self.generate_boxplots(
                df, "artifacts/boxplots"
            ),
            "correlation_heatmap": lambda: self.generate_correlation_heatmap(df),
            "categorical_plots": lambda: self.generate_categorical_plots(df),
            "scatter_plots": lambda: self.generate_scatter_plots(
                df, target_column
            ),
        }

        for visualization_name, method in visualization_methods.items():
            try:
                logger.info(f"GENERATING {visualization_name.upper()}")

                result = method()

                visualization_manifest[visualization_name] = {
                    "status": "success",
                    "result": result,
                    "error": None
                }

            except Exception as e:
                logger.exception(
                    f"FAILED TO GENERATE {visualization_name}"
                )

                visualization_manifest[visualization_name] = {
                    "status": "failed",
                    "result": None,
                    "error": str(e)
                }

        logger.info("ALL VISUALIZATION TASKS COMPLETED")

        return visualization_manifest
