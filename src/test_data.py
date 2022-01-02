import pandas as pd

from src.configurations import Configuration


class TestData:
    def __init__(self, config: Configuration):
        self.configuration = config
        self.raw_pd_df = pd.read_csv(config.test_data_path)

    def get_raw_pd_df(self):
        return self.raw_pd_df
