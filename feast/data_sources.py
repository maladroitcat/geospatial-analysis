from feast import FileSource
from feast.data_source import PushSource

# Offline store: the isochrone history the pipeline appends to on every run.
# The bucket is the project's ObjectBucketClaim (values.yaml s3.bucketName) and
# the endpoint is OpenShift Data Foundation's in-cluster S3 service. Clients
# that read it need AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from the cpl-s3
# secret, s3fs installed, and AWS_CA_BUNDLE pointing at the service CA.
isochrone_batch = FileSource(
    name="isochrone_batch",
    path="s3://cpl-coverage-data/features/isochrones.parquet",
    s3_endpoint_override="https://s3.openshift-storage.svc:443",
    timestamp_field="event_timestamp",
    description="History of every isochrone fetched from Mapbox, stored as parquet in the project bucket",
)

# Online writes go through this push source; materialization reads the batch
# source above to rebuild the online store, e.g. after the Feast pod restarts.
isochrone_push = PushSource(
    name="isochrone_push",
    batch_source=isochrone_batch,
    description="Pushes freshly fetched Mapbox isochrones straight to the online store",
)
