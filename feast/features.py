from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.data_source import PushSource
from feast.types import String, Int64, Float64
from feast.value_type import ValueType

isochrone_entity = Entity(
    name="isochrone",
    join_keys=["isochrone_key"],
    value_type=ValueType.STRING,
)

isochrone_push = PushSource(
    name="isochrone_push",
    batch_source=FileSource(path="dummy", timestamp_field="event_timestamp"),
)

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
)
