from src.ml_platform.exception.excepton import ModelForge
from src.ml_platform.logging.logger import logger
import warnings
import sys
import pandas as pd

class  DataQualityService:
    def analyze_missing_values(self, df:pd.DataFrame):
        try:
            logger.info("Analyze missing values is Started")
            df_copy = df.copy()
            missing_values = []
            
            for col in df_copy.columns:
                missing_count = df_copy[col].isnull().sum()
                total_rows = len(df_copy[col])
                
                information = {
                    "column_name": col,
                    "missing": int(missing_count),
                    "missing_percentage": (
                    float(missing_count / total_rows) * 100
                    if total_rows > 0 else 0.0
                    )
                }
                
                missing_values.append(information)
                
            logger.info(f"Missing vlaues is {missing_values}")
            return missing_values
        except Exception as e:
            raise ModelForge(e, sys)
        
    def analyze_duplicate_rows(self, df: pd.DataFrame):
        try:
            logger.info("Checking Duplicate Rows is Started")
            df_copy = df.copy()
            total_rows = len(df_copy)
            duplicate_count = int(df_copy.duplicated().sum())
            duplicate_percentage = (
                float(duplicate_count / total_rows) * 100
                if total_rows > 0 else 0.0
            )

            duplicate_rows = {
                "duplicate_count": duplicate_count,
                "duplicate_percentage": duplicate_percentage
            }

            logger.info(f"Duplicate Rows is: {duplicate_rows}")
            return duplicate_rows
        except Exception as e:
            raise ModelForge(e, sys)

    def constant_columns(self, df:pd.DataFrame):
        try:
            logger.info("Now we are Started Checking constant columns")
            
            all_null_cols = df.columns[df.isna().all()].tolist()
            distinct_counts = df.nunique(dropna=True)
            constant_cols = distinct_counts[distinct_counts == 1].index.tolist()
            constant_cols = [col for col in constant_cols if col not in all_null_cols]
            
            return {
                "constant_columns": constant_cols,
                "all_null_cols": all_null_cols
            }
        except Exception as e:
            raise ModelForge(e, sys)
    def analyze_nearly_constant_columns(self, df:pd.DataFrame, threshold=0.95):
        try:
            df_copy = df.copy()
            logger.info("Start Analyze Nearly constant columns")
            
            nearly_constant_cols = {}
            
            for col in df_copy.columns:
                value_proporations = df_copy[col].value_counts(normalize=True, dropna=True)
                
                if not value_proporations.empty:
                    max_frequency = value_proporations.iloc[0]
                    most_frequent_value = value_proporations.index[0]
                    
                    if max_frequency >= threshold:
                        nearly_constant_cols[col] = {
                            "dominate_value": most_frequent_value,
                            "frequecy_ratio" : round(max_frequency, 4)
                        }
            return  nearly_constant_cols
        except Exception as e:
            raise ModelForge(e, sys)
    def identify_column_types(self, df: pd.DataFrame, card_threshold=20):
        try:
            logger.info("Now We are started Identify Columns type")
            df_copy = df.copy()
            columns_types = {
                "numeric": [],
                "categorical": [],
                "boolean": [],
                "datetime": [],
                "text": []
            }

            for col in df_copy.columns:
                non_null_values = df_copy[col].dropna()

                is_binary_like = (
                    not non_null_values.empty
                    and non_null_values.isin([True, False, 0, 1]).all()
                    and non_null_values.nunique() <= 2
                )

                if pd.api.types.is_bool_dtype(df_copy[col]) or is_binary_like:
                    columns_types['boolean'].append(col)
                elif pd.api.types.is_numeric_dtype(df_copy[col]):
                    columns_types['numeric'].append(col)
                elif pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    columns_types['datetime'].append(col)
                elif pd.api.types.is_object_dtype(df_copy[col]) or pd.api.types.is_string_dtype(df_copy[col]):
                    if df_copy[col].nunique(dropna=True) <= card_threshold:
                        columns_types['categorical'].append(col)
                    else:
                        columns_types['text'].append(col)
                else:
                    columns_types['text'].append(col)

            return columns_types
        except Exception as e:
            raise ModelForge(e, sys)

    def detect_date_like_columns(self, df: pd.DataFrame, threshold: float = 0.5, sample_size: int = 100):
        try:
            logger.info("Now we are going to Detect date like columns") 
            date_like_cols = []
            object_cols = df.select_dtypes(include=['object', 'string']).columns
            
            for col in object_cols:
                series = df[col].dropna()
                if series.empty:
                    continue
                sample = series.sample(n=min(len(series), sample_size), random_state=42)
                try:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    valid_count = parsed.notna().sum()
                    ratio = valid_count / len(sample)
                
                    is_pure_numeric = sample.astype(str).str.match(r'^\d+$').all()

                    if ratio >= threshold and not is_pure_numeric:
                        date_like_cols.append(col)
                        warnings.warn(
                            f"Warning: Column '{col}' contains potential date/datetime values "
                            f"({ratio * 100:.1f}% parsed successfully). Automatic conversion was skipped.",
                            UserWarning
                        )
                except Exception:
                    continue
                    
            return date_like_cols

        except Exception as e:
            raise ModelForge(e, sys)
    def validate_target_column(self, df: pd.DataFrame, target_column: str, task_type: str = 'classification'):
        try:
            logger.info(f"Validating target column '{target_column}' for task type: '{task_type}'")

            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' does not exist in the DataFrame.")

            missing_count = df[target_column].isnull().sum()
            if missing_count > 0:
                logger.warning(f"Warning: Target column '{target_column}' contains {missing_count} missing values.")
                
            validation_results = {
                "target_column": target_column,
                "missing_values": int(missing_count),
                "task_type": task_type
            }
            
            if task_type.lower() == 'classification':
                class_counts = df[target_column].value_counts().to_dict()
                validation_results["class_counts"] = class_counts
                logger.info(f"Class distribution for '{target_column}': {class_counts}")
               
                if len(class_counts) <= 1:
                    raise ValueError(f"Classification target '{target_column}' has only {len(class_counts)} unique class. At least 2 classes are required.")
                    
            elif task_type.lower() == 'regression':
                logger.info(f"Regression target '{target_column}' validated successfully.")
            else:
                logger.warning(f"Unknown task type '{task_type}'. Performing basic validation only.")
                
            return validation_results

        except Exception as e:
            raise ModelForge(e, sys)
    
    def detect_imbalanced_target(self, df: pd.DataFrame, target_column: str, imbalance_threshold: float = 0.1):

        try:
            logger.info(f"Detecting class imbalance for target column: '{target_column}'")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' does not exist in the DataFrame.")
                
            series = df[target_column].dropna()
            if series.empty:
                raise ValueError(f"Target column '{target_column}' contains only missing values.")
                
            # Check if target appears continuous (regression) rather than categorical/classification
            is_numeric = pd.api.types.is_numeric_dtype(series)
            if is_numeric and series.nunique() > 20:
                logger.info(f"Target column '{target_column}' appears to be continuous (regression). Skipping class imbalance detection.")
                return {"is_imbalanced": False, "reason": "Regression target detected"}
                
            # Compute class counts and proportions
            class_counts = series.value_counts()
            class_proportions = series.value_counts(normalize=True)
            
            min_class_ratio = class_proportions.min()
            min_class_name = class_proportions.idxmin()
            
            is_imbalanced = min_class_ratio < imbalance_threshold
            
            if is_imbalanced:
                warnings.warn(
                    f"Warning: Target column '{target_column}' is imbalanced. "
                    f"Minority class '{min_class_name}' represents only {min_class_ratio * 100:.2f}% of the data.",
                    UserWarning
                )
                logger.warning(f"Class imbalance detected in '{target_column}': {class_proportions.to_dict()}")
            else:
                logger.info(f"Class distribution for '{target_column}' is balanced based on the threshold.")
                
            return {
                "class_counts": class_counts.to_dict(),
                "class_proportions": class_proportions.to_dict(),
                "is_imbalanced": is_imbalanced,
                "minority_class": min_class_name,
                "minority_ratio": float(min_class_ratio)
            }

        except Exception as e:
            raise ModelForge(e, sys)
    def detect_identifier_and_high_cardinality_columns(self, df: pd.DataFrame, cardinality_threshold: float = 0.9):
            try:
                logger.info("Now we are going to detect identifier and high cardinality columns")
                
                flagged_cols = {
                    "potential_ids": [],
                    "high_cardinality": []
                }
                
                total_rows = len(df)
                if total_rows == 0:
                    return flagged_cols
                    
                # Common keywords for identifier columns
                id_keywords = ['id', 'identifier', 'guid', 'uuid', 'key', 'code', 'index']
                
                for col in df.columns:
                    series = df[col].dropna()
                    if series.empty:
                        continue
                        
                    unique_count = series.nunique()
                    cardinality_ratio = unique_count / total_rows
                    
                    col_lower = col.lower()
                    is_name_match = any(keyword in col_lower for keyword in id_keywords)
                    
                    # Check for potential ID columns (name match + decent uniqueness, or near 100% unique)
                    if (is_name_match and cardinality_ratio > 0.4) or (cardinality_ratio >= 0.99 and total_rows > 10):
                        flagged_cols["potential_ids"].append(col)
                        warnings.warn(
                            f"Warning: Column '{col}' is detected as a potential identifier (ID) "
                            f"(Cardinality ratio: {cardinality_ratio:.2f}). Automatic removal was skipped.",
                            UserWarning
                        )
                    # Check for general high cardinality columns (e.g., text descriptions or free-form strings)
                    elif cardinality_ratio >= cardinality_threshold and total_rows > 10:
                        flagged_cols["high_cardinality"].append(col)
                        warnings.warn(
                            f"Warning: Column '{col}' has high cardinality "
                            f"({unique_count} unique values, {cardinality_ratio * 100:.1f}% of rows). Automatic removal was skipped.",
                            UserWarning
                        )
                        
                return flagged_cols
    
            except Exception as e:
                raise ModelForge(e, sys)
    def validate_dataset(self, df: pd.DataFrame, target_column: str = None, task_type: str = 'classification'):
        try:
            logger.info("Starting comprehensive dataset validation orchestrator.")
            
            report = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "details": {}
            }
            
            # 1. Check for empty dataset
            if df is None or df.empty or df.shape[0] == 0 or df.shape[1] == 0:
                report["errors"].append("Dataset is empty, None, or has zero dimensions.")
                report["is_valid"] = False
                return report  # Critical failure, stop further processing
                
            report["details"]["shape"] = df.shape
            
            # 2. Check for duplicate column names
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            if duplicate_cols:
                report["errors"].append(f"Duplicate column names detected: {duplicate_cols}")
                report["is_valid"] = False
                
            # 3. Detect date-like columns (Warning)
            try:
                if hasattr(self, 'detect_date_like_columns'):
                    date_cols = self.detect_date_like_columns(df)
                    if date_cols:
                        report["warnings"].append(f"Potential date-like columns detected: {date_cols}")
                        report["details"]["date_like_columns"] = date_cols
            except Exception as sub_e:
                logger.warning(f"Date-like columns check encountered an issue: {sub_e}")
                
            # 4. Detect identifier and high cardinality columns (Warning)
            try:
                if hasattr(self, 'detect_identifier_and_high_cardinality_columns'):
                    id_card_results = self.detect_identifier_and_high_cardinality_columns(df)
                    if id_card_results.get("potential_ids"):
                        report["warnings"].append(f"Potential identifier (ID) columns detected: {id_card_results['potential_ids']}")
                    if id_card_results.get("high_cardinality"):
                        report["warnings"].append(f"High cardinality columns detected: {id_card_results['high_cardinality']}")
                    report["details"]["cardinality_analysis"] = id_card_results
            except Exception as sub_e:
                logger.warning(f"Identifier/Cardinality check encountered an issue: {sub_e}")
                
            # 5. Target Column Validations (if target_column is provided)
            if target_column:
                # Validate target existence and missing values
                try:
                    if hasattr(self, 'validate_target_column'):
                        target_val_res = self.validate_target_column(df, target_column, task_type)
                        report["details"]["target_validation"] = target_val_res
                except Exception as target_err:
                    report["errors"].append(f"Target column validation failed: {str(target_err)}")
                    report["is_valid"] = False
                
                # Detect class imbalance (Classification only)
                if task_type and task_type.lower() == 'classification':
                    try:
                        if hasattr(self, 'detect_imbalanced_target'):
                            imbal_res = self.detect_imbalanced_target(df, target_column)
                            if imbal_res.get("is_imbalanced"):
                                report["warnings"].append(
                                    f"Class imbalance detected in target '{target_column}' "
                                    f"(Minority class '{imbal_res.get('minority_class')}' ratio: {imbal_res.get('minority_ratio', 0) * 100:.2f}%)."
                                )
                            report["details"]["imbalance_analysis"] = imbal_res
                    except Exception as sub_e:
                        logger.warning(f"Class imbalance detection encountered an issue: {sub_e}")
                        
                # Detect potential target leakage
                try:
                    if hasattr(self, 'detect_potential_leakage'):
                        leakage_res = self.detect_potential_leakage(df, target_column)
                        if leakage_res.get("high_correlation"):
                            report["warnings"].append(f"Potential target leakage via high correlation: {leakage_res['high_correlation']}")
                        if leakage_res.get("suspicious_names"):
                            report["warnings"].append(f"Potential target leakage via suspicious names: {leakage_res['suspicious_names']}")
                        report["details"]["leakage_analysis"] = leakage_res
                except Exception as sub_e:
                    logger.warning(f"Target leakage detection encountered an issue: {sub_e}")
            
            logger.info(f"Dataset validation complete. Valid: {report['is_valid']} | Errors: {len(report['errors'])} | Warnings: {len(report['warnings'])}")
            return report

        except Exception as e:
            raise ModelForge(e, sys)
        