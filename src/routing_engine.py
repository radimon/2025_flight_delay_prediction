import numpy as np
import pandas as pd


class RoutingEngine:
    """
    Choose best parking grid based on
    parking probability + walking distance + driving distance
    """

    def __init__(self, mapper):
        self.mapper = mapper

    # -------------------------
    # 距離計算 (簡單版)
    # -------------------------

    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance (km) between two lat/lng
        """
        R = 6371

        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)

        dlat = lat2 - lat1
        dlon = np.radians(lon2 - lon1)

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return R * c

    # -------------------------
    # grid → lat/lng
    # -------------------------

    def grid_to_latlng(self, x, y):

        lat, lng = self.mapper.grid_to_latlng(x, y)

        return lat, lng

    # -------------------------
    # 計算 routing score
    # -------------------------

    def compute_score(
        self,
        drive_dist,
        walk_dist,
        parking_prob,
        alpha=1.0,
        beta=1.0,
        gamma=2.0
    ):
        """
        Lower score is better
        """

        return alpha * walk_dist + beta * drive_dist - gamma * parking_prob

    # -------------------------
    # 找候選停車 grid
    # -------------------------

    def find_candidate_parking(
        self,
        df_parking,
        dest_lat,
        dest_lng,
        top_k=50
    ):

        rows = []

        for _, r in df_parking.iterrows():

            lat, lng = self.grid_to_latlng(r.x, r.y)

            dist = self.haversine(dest_lat, dest_lng, lat, lng)

            rows.append(
                {
                    "x": r.x,
                    "y": r.y,
                    "lat": lat,
                    "lng": lng,
                    "parking_prob": r.parking_prob,
                    "walk_dist": dist
                }
            )

        df = pd.DataFrame(rows)

        # 先挑距離近 + 停車機率高的
        df = df.sort_values(["walk_dist", "parking_prob"], ascending=[True, False])

        return df.head(top_k)

    # -------------------------
    # 推薦停車位置
    # -------------------------

    def recommend_parking(
        self,
        df_parking,
        start_lat,
        start_lng,
        dest_lat,
        dest_lng,
        alpha=1.0,
        beta=1.0,
        gamma=2.0
    ):

        candidates = self.find_candidate_parking(df_parking, dest_lat, dest_lng)

        scores = []

        for _, r in candidates.iterrows():

            drive_dist = self.haversine(start_lat, start_lng, r.lat, r.lng)

            score = self.compute_score(
                drive_dist,
                r.walk_dist,
                r.parking_prob,
                alpha,
                beta,
                gamma
            )

            scores.append(score)

        candidates["score"] = scores

        best = candidates.sort_values("score").iloc[0]

        return best, candidates