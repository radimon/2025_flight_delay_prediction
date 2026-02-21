import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


class ConfidenceEngine:
    """
    Convert predicted density dataframe into
    HIGH / MED / LOW spatial confidence circles.
    """

    def __init__(self, eps=5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    # --------------------------------------------------
    # Step 1: assign confidence level
    # --------------------------------------------------
    def assign_levels(self, df, value_col="pred_count"):
        thresholds = {
            "HIGH": df[value_col].quantile(0.90),
            "MED": df[value_col].quantile(0.75),
            "LOW": df[value_col].quantile(0.60),
        }

        def classify(v):
            if v >= thresholds["HIGH"]:
                return "HIGH"
            elif v >= thresholds["MED"]:
                return "MED"
            elif v >= thresholds["LOW"]:
                return "LOW"
            else:
                return "NONE"

        df = df.copy()
        df["level"] = df[value_col].apply(classify)

        return df, thresholds

    # --------------------------------------------------
    # Step 2: cluster by level
    # --------------------------------------------------
    def cluster_level(self, df, level):
        subset = df[df["level"] == level].copy()

        if len(subset) == 0:
            return pd.DataFrame()

        coords = subset[["x", "y"]].values

        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        ).fit(coords)

        subset["cluster_id"] = clustering.labels_

        clusters = []

        for cid in subset["cluster_id"].unique():
            if cid == -1:
                continue

            group = subset[subset["cluster_id"] == cid]

            center_x = group["x"].mean()
            center_y = group["y"].mean()

            radius = np.sqrt(
                ((group["x"] - center_x) ** 2 +
                 (group["y"] - center_y) ** 2).max()
            )

            clusters.append({
                "level": level,
                "center_x": center_x,
                "center_y": center_y,
                "radius_grid": radius,
                "points": len(group)
            })

        return pd.DataFrame(clusters)

    # --------------------------------------------------
    # Step 3: build all circles
    # --------------------------------------------------
    def build_confidence_circles(self, df):
        df, thresholds = self.assign_levels(df)

        result = []

        for level in ["HIGH", "MED", "LOW"]:
            clusters = self.cluster_level(df, level)
            result.append(clusters)

        final_df = pd.concat(result, ignore_index=True)

        return final_df, thresholds
