import pandas as pd

class DataHandler:
    def __init__(self, original_data_path):
        self.original_data_path = original_data_path
        self.data = pd.read_csv(original_data_path)

    def add_new_data(self, new_data):
        self.data = pd.concat([self.data, new_data], ignore_index=True)

    def save_updated_data(self):
        self.data.to_csv(self.original_data_path, index=False)

    def get_data(self):
        return self.data