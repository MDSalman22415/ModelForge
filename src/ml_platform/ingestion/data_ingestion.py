import numpy as np
import pandas as pd

class DataIngestionService:
    def load_csv(self, file_path):
        dataframe = pd.read_csv(file_path)
        return dataframe