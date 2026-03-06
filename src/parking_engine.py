import pandas as pd
import numpy as np


class ParkingEngine:
    """
    Combine human flow prediction with parking parameters
    to estimate parking demand and parking availability probability.
    """

    def __init__(self, parking_param_path: str):
        """
        parking_param_path: path to grid_parking_params.parquet
        """
        self.df_parking = pd.read_parquet(parking_param_path)

    def merge_prediction(self, df_pred: pd.DataFrame):
        """
        Merge human flow prediction with parking parameters.

        df_pred:
            d, t, x, y, score
        """

        df = df_pred.merge(
            self.df_parking,
            on=["x", "y", "t"],
            how="left"
        )

        return df

    def compute_parking_demand(self, df: pd.DataFrame):
        """
        parking_demand = predicted_people * parking_ratio
        """

        df["parking_demand"] = df["score"] * df["parking_ratio"]
        return df

    def compute_parking_probability(self, df: pd.DataFrame):
        """
        parking_prob = probability of finding a parking spot
        """

        capacity = df["parking_capacity"]

        df["parking_prob"] = np.where(
            capacity > 0,
            (capacity - df["parking_demand"]) / capacity,
            0
        )

        df["parking_prob"] = df["parking_prob"].clip(0, 1)

        df["parking_prob"] = (capacity - df["parking_demand"]) / capacity
        df["parking_prob"] = df["parking_prob"].clip(0, 1)

        return df

    def run(self, df_pred: pd.DataFrame):
        """
        Full pipeline
        """

        df = self.merge_prediction(df_pred)
        df = self.compute_parking_demand(df)
        df = self.compute_parking_probability(df)

        return df