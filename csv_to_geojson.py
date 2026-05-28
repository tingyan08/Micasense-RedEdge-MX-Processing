import csv
import json
from pyproj import Transformer

HALF_SIZE = 6.0  # 12m x 12m -> ±6m
EPSG_UTM = "EPSG:32616"
INPUT_CSV = "./Data/PPAC-B3/PPAC-B3_aoi.csv"
OUTPUT_GEOJSON = "./Data/PPAC-B3/PPAC-B3_aoi_square.geojson"

transformer = Transformer.from_crs("EPSG:4326", EPSG_UTM, always_xy=True)

features = []
with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = int(row["Point_ID"])
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        cx, cy = transformer.transform(lon, lat)

        coords = [
            [cx + HALF_SIZE, cy - HALF_SIZE],
            [cx + HALF_SIZE, cy + HALF_SIZE],
            [cx - HALF_SIZE, cy + HALF_SIZE],
            [cx - HALF_SIZE, cy - HALF_SIZE],
            [cx + HALF_SIZE, cy - HALF_SIZE],
        ]

        features.append({
            "type": "Feature",
            "properties": {
                "id": pid,
                "center": f"({cx}, {cy})",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
        })

geojson = {
    "type": "FeatureCollection",
    "name": "aoi_square",
    "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:{EPSG_UTM.replace(':', '::')}"}},
    "features": features,
}

with open(OUTPUT_GEOJSON, "w") as f:
    json.dump(geojson, f, separators=(", ", ": "))

print(f"Written {len(features)} features to {OUTPUT_GEOJSON}")
