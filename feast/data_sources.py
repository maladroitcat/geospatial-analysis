from feast import FileSource
from feast.data_source import PushSource

isochrone_batch = FileSource(
    name="isochrone_batch",
    path="s3://cpl-coverage-data/features/isochrones.parquet",
    timestamp_field="event_timestamp",
    description="Isochrone geometries stored in S3",
)

isochrone_push = PushSource(
    name="isochrone_push",
    batch_source=isochrone_batch,
    description="Real-time push source for isochrone data from Mapbox API",
)
