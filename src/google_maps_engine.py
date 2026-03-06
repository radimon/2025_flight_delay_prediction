import requests
import polyline
import folium


class GoogleMapsEngine:

    def __init__(self, api_key: str):
        self.api_key = api_key

    # -----------------------------
    # 呼叫 Google Directions API
    # -----------------------------

    def get_route(self, origin_lat, origin_lng, dest_lat, dest_lng):

        url = "https://maps.googleapis.com/maps/api/directions/json"

        params = {
            "origin": f"{origin_lat},{origin_lng}",
            "destination": f"{dest_lat},{dest_lng}",
            "key": self.api_key
        }

        r = requests.get(url, params=params)
        data = r.json()

        if data["status"] != "OK":
            raise RuntimeError(f"Google API error: {data['status']}")

        route = data["routes"][0]

        poly = route["overview_polyline"]["points"]

        coords = polyline.decode(poly)

        leg = route["legs"][0]

        distance = leg["distance"]["value"]
        duration = leg["duration"]["value"]

        return coords, distance, duration

    # -----------------------------
    # 畫 route 在 folium map
    # -----------------------------

    def draw_route(self, fmap, coords, color="blue"):

        folium.PolyLine(
            coords,
            color=color,
            weight=5,
            opacity=0.8
        ).add_to(fmap)

    # -----------------------------
    # A → Parking → B
    # -----------------------------

    def route_via_parking(
        self,
        fmap,
        start_lat,
        start_lng,
        park_lat,
        park_lng,
        dest_lat,
        dest_lng
    ):

        # A → parking
        coords1, dist1, dur1 = self.get_route(
            start_lat,
            start_lng,
            park_lat,
            park_lng
        )

        # parking → destination
        coords2, dist2, dur2 = self.get_route(
            park_lat,
            park_lng,
            dest_lat,
            dest_lng
        )

        self.draw_route(fmap, coords1, color="blue")
        self.draw_route(fmap, coords2, color="green")

        total_dist = dist1 + dist2
        total_time = dur1 + dur2

        return {
            "distance_m": total_dist,
            "duration_s": total_time
        }