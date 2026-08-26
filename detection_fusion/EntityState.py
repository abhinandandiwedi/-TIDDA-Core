from enum import Enum


class EntityState(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    EXPIRED = "EXPIRED"
