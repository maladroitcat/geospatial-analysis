from feast import Entity
from feast.value_type import ValueType

isochrone_entity = Entity(
    name="isochrone",
    join_keys=["isochrone_key"],
    value_type=ValueType.STRING,
    description="Composite key of branch|mode|travel_time for isochrone lookups",
)
