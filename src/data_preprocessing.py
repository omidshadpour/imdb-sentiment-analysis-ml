import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self , data_path , test_size = 0.2 , random_state = 42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

        self.df = None
        self.train_df = None
        self.test_df = None

    def load_data(self):

        # Read CSV into a DataFrame
        self.df = pd.read_csv(self.data_path)

        # Drop rows where review or sentiment is missing
        self.df = self.df.dropna(subset = ["review" , "sentiment"])

        # Reset index after dropping rows
        self.df = self.df.reset_index(drop = True)

        return self.df

    def split_data(self):

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data first.")

        self.train_df , self.test_df = train_test_split(
            self.df , test_size = self.test_size ,
            random_state = self.random_state , stratify = self.df["sentiment"]
        )
        return self.train_df, self.test_df