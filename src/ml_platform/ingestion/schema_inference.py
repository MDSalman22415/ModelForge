import pandas as pd


class SchemaInterfaceService:
    def infer_schema(self, df:pd.DataFrame):
        df_copy = df.copy()
        summary_list = []
        
        for col in df_copy.columns:
            user_information = {
                "name" : str(col),
                "data_types" : str(df_copy[col].dtypes),
                "null_count" : int(df_copy[col].isnull().sum()),
                "unique_count" : int(df_copy[col].nunique()),
            }
            
            summary_list.append(user_information)
        
        
        return summary_list