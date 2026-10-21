"""MCP server exposing the CPL library coverage model."""

import functools
import math
import os
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
import urllib3
from fastmcp import FastMCP

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

mcp = FastMCP("cpl-coverage")

MODEL_ENDPOINT = os.environ["MODEL_ENDPOINT"].rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "cov-1")

_SA_TOKEN = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")


def _model_token() -> str:
    """In-cluster the pod's own service account token satisfies the model's
    SubjectAccessReview, so no static credential is needed. MODEL_TOKEN is the
    escape hatch for running this outside the cluster."""
    if token := os.environ.get("MODEL_TOKEN"):
        return token
    if _SA_TOKEN.exists():
        return _SA_TOKEN.read_text().strip()
    raise RuntimeError(
        "No credential for the model endpoint: set MODEL_TOKEN, or run in a pod "
        "whose service account can 'get' the inferenceservice."
    )

# Reference data ships in the model image alongside the model itself, so the
# two can never drift apart and this server needs no bucket credentials.
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))

# Order is load-bearing: the server takes a bare array, not named fields.
FEATURES = [
    "distance_km", "bearing_sin", "bearing_cos", "tract_area_sqkm",
    "travel_time", "mode_driving", "mode_cycling", "mode_walking",
]

# Chicago bounding box, from the tract geometries the model was trained on.
LAT_RANGE = (41.60, 42.06)
LON_RANGE = (-87.98, -87.50)

@functools.cache
def _tracts() -> pd.DataFrame:
    """Tract centroids, areas and population. Centroids are precomputed at image
    build time so this process needs no geometry libraries."""
    return pd.read_parquet(DATA_DIR / "tracts.parquet")


@functools.cache
def _coverage_scores() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "coverage_scores.parquet")


def _current_coverage(mode: str, travel_time: int) -> dict:
    scores = _coverage_scores()
    scores = scores[(scores["mode"] == mode) & (scores["travel_time"] == travel_time)]
    return scores.set_index("geoid")["percent_covered"].to_dict()


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _infer(rows: list[list[float]]) -> list[float]:
    body = {"inputs": [{
        "name": "predict",
        "shape": [len(rows), len(FEATURES)],
        "datatype": "FP32",
        "data": [v for row in rows for v in row],
    }]}
    resp = requests.post(
        f"{MODEL_ENDPOINT}/v2/models/{MODEL_NAME}/infer",
        headers={"Authorization": f"Bearer {_model_token()}", "Content-Type": "application/json"},
        json=body,
        verify=False,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["outputs"][0]["data"]


@mcp.tool
def predict_branch_coverage(
    latitude: float,
    longitude: float,
    mode: Literal["walking", "cycling", "driving"] = "walking",
    travel_time: int = 10,
    top_n: int = 10,
) -> dict:
    """Estimate the coverage a new Chicago Public Library branch would provide.

    Predicts, for every census tract in Chicago, how much of it would fall
    within reach of a branch at the given coordinates, then compares that
    against current coverage to find who gains access.

    Args:
        latitude: Branch latitude. Must be within Chicago.
        longitude: Branch longitude. Must be within Chicago.
        mode: Travel mode to evaluate.
        travel_time: Travel time in minutes. The model was trained on
            driving/5, cycling/10 and walking/10; other values extrapolate.
        top_n: How many of the most-affected tracts to return in detail.

    Returns:
        Summary counts plus the top_n tracts ranked by coverage gained.
    """
    if not LAT_RANGE[0] <= latitude <= LAT_RANGE[1] or not LON_RANGE[0] <= longitude <= LON_RANGE[1]:
        raise ValueError(
            f"({latitude}, {longitude}) is outside Chicago "
            f"(lat {LAT_RANGE[0]}..{LAT_RANGE[1]}, lon {LON_RANGE[0]}..{LON_RANGE[1]}). "
            "The model has no training data there."
        )

    tracts = _tracts()
    current = _current_coverage(mode, travel_time)

    rows = []
    for t in tracts.itertuples():
        dist = _haversine_km(latitude, longitude, t.centroid_lat, t.centroid_lon)
        bearing = math.radians(_bearing_deg(latitude, longitude, t.centroid_lat, t.centroid_lon))
        rows.append([
            dist, math.sin(bearing), math.cos(bearing), t.area_sqkm, float(travel_time),
            1.0 if mode == "driving" else 0.0,
            1.0 if mode == "cycling" else 0.0,
            1.0 if mode == "walking" else 0.0,
        ])

    # The model is an unconstrained regressor and emits small negatives just
    # past the distance where coverage stops.
    predicted = [min(max(p, 0.0), 1.0) for p in _infer(rows)]

    results = []
    for t, pred, row in zip(tracts.itertuples(), predicted, rows):
        before = float(current.get(t.geoid, 0.0))
        # Additive rather than a geometric union, so a new branch next to an
        # existing one double-counts shared ground. A raw total above 1.0 means
        # the tract is already served by something else.
        raw_total = before + pred
        after = min(raw_total, 1.0)
        results.append({
            "geoid": str(t.geoid),
            "tract_name": t.tract_name,
            "population": int(t.population),
            "distance_km": round(row[0], 3),
            "current_coverage": round(before, 4),
            "predicted_overlap": round(pred, 4),
            "new_coverage": round(after, 4),
            "delta": round(after - before, 4),
            "overserved": raw_total > 1.0,
        })

    results.sort(key=lambda r: r["delta"], reverse=True)
    gained = [r for r in results if r["delta"] > 0.01]

    return {
        "location": {"latitude": latitude, "longitude": longitude},
        "mode": mode,
        "travel_time_minutes": travel_time,
        "tracts_evaluated": len(results),
        "tracts_gaining_coverage": len(gained),
        "population_served": sum(r["population"] for r in gained),
        "max_delta": max((r["delta"] for r in results), default=0.0),
        "mean_delta_affected": round(
            sum(r["delta"] for r in gained) / len(gained), 4
        ) if gained else 0.0,
        "overserved_tracts": sum(1 for r in gained if r["overserved"]),
        "top_tracts": results[:top_n],
    }


if __name__ == "__main__":
    # HTTP rather than stdio: this runs as a long-lived Service that agents
    # connect to over the network, not as a subprocess.
    mcp.run(
        transport="http",
        host=os.environ.get("MCP_HOST", "0.0.0.0"),
        port=int(os.environ.get("MCP_PORT", "8000")),
    )
