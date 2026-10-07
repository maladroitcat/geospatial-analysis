from feast import FeatureService

from feature_views import isochrone_fv

isochrone_service = FeatureService(
    name="isochrone_service",
    features=[isochrone_fv],
    description="All isochrone features for coverage analysis",
)
