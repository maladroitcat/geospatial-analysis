from datetime import timedelta

from feast import FeatureView, Field
from feast.types import String, Int64, Float64

from entities import isochrone_entity
from data_sources import isochrone_push

isochrone_fv = FeatureView(
    name="isochrone_features",
    entities=[isochrone_entity],
    schema=[
        Field(name="branch", dtype=String),
        Field(name="mode", dtype=String),
        Field(name="travel_time", dtype=Int64),
        Field(name="latitude", dtype=Float64),
        Field(name="longitude", dtype=Float64),
        Field(name="geometry_wkt", dtype=String),
        Field(name="fetch_ts", dtype=Int64),
    ],
    source=isochrone_push,
    online=True,
    ttl=timedelta(0),
    description="Mapbox isochrone geometries per branch, mode, and travel time",
)
