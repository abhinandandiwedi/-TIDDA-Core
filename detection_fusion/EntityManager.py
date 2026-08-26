from .Entity import Entity
from .Exceptions import EntityNotFoundError
from .EntityState import EntityState


class EntityManager:
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}

    def create_entity(self, entity: Entity) -> Entity:
        self.entities[entity.entity_id] = entity
        return entity

    def get_entity(self, entity_id: str) -> Entity:
        try:
            return self.entities[entity_id]
        except KeyError as exc:
            raise EntityNotFoundError(entity_id) from exc

    def update_entity(self, entity: Entity) -> Entity:
        if entity.entity_id not in self.entities:
            raise EntityNotFoundError(entity.entity_id)
        self.entities[entity.entity_id] = entity
        return entity

    def remove_entity(self, entity_id: str) -> None:
        self.entities.pop(entity_id, None)

    def get_all_entities(self): return list(self.entities.values())
    def get_active_entities(self): return [e for e in self.entities.values() if e.status == EntityState.ACTIVE]
    def get_stale_entities(self): return [e for e in self.entities.values() if e.status == EntityState.STALE]
    def get_expired_entities(self): return [e for e in self.entities.values() if e.status == EntityState.EXPIRED]
