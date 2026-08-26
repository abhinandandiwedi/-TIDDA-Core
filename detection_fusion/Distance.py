from math import atan2, cos, radians, sin, sqrt


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return Haversine great-circle distance in meters."""
    earth_radius_m = 6_371_000.0
    lat1_rad, lat2_rad = radians(lat1), radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    value = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    value = max(0.0, min(1.0, value))
    return earth_radius_m * 2 * atan2(sqrt(value), sqrt(1 - value))
