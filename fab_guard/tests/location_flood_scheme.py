import pandera as pa
from pandera.typing import Series, String
from plugins.FabFlee.fab_guard.tests.location_scheme import LocationsScheme
class LocationsFloodScheme(LocationsScheme):
    # name: Series[pa.String] = pa.Field(nullable=False, alias='#"name"')
    location_type: Series[pa.String] = pa.Field(
        isin=["conflict_zone", "town", "camp", "forwarding_hub", "marker", "idpcamp", "flood_zone"])
