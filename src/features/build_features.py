import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class ZirconsDataProcessor:
    """Class for processing zircon data including feature engineering and preprocessing."""

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Constructor for ZirconsDataProcessor.

        Args:
            data (pd.DataFrame): The input data containing zircon information.
        """
        self.data = data.drop(columns=['price'])
        self.enc = OrdinalEncoder()
        self.scaler = StandardScaler()
        self.categorical = ['cut', 'color', 'clarity']
        self.numerical = ['depth', 'table', 'x', 'y', 'z']
        self.price = data.price

    def new_features(self):
        """Generate new features based on zircon dimensions and attributes."""
        self.data['volume'] = self.data['x'] * self.data['y'] * self.data['z']
        self.data['density'] = (self.data['carat'] * 0.2) / \
            (self.data['volume'] + 1e-6)
        self.data['depth_per_volume'] = self.data['depth'] / \
            (self.data['volume'] + 1e-6)
        self.data['depth_per_density'] = self.data['depth'] / \
            (self.data['density'] + 1e-6)
        self.data['depth_per_table'] = self.data['depth'] / \
            (self.data['table'] + 1e-6)
        self.data['ratio_xy'] = self.data['x'] / self.data['y']
        self.data['ratio_xz'] = self.data['x'] / self.data['z']
        self.data['ratio_yz'] = self.data['y'] / self.data['z']

    def data_processor(self) -> pd.DataFrame:
        """
        Process the zircon data by generating new features and preprocessing.

        Returns:
            pd.DataFrame: The processed zircon data.
        """
        self.new_features()
        self.data[self.categorical] = self.enc.fit_transform(
            self.data[self.categorical])

        self.data[self.numerical] = self.scaler.fit_transform(
            self.data[self.numerical])

        return self.data, self.price
